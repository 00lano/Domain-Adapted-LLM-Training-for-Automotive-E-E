# Part 2: Continued Pre-Training — Design Write-up

Continued Pre-Training (CPT) adapts a general base model to a target domain by continuing next-token training on in-domain text. The goal is to turn a generic model into one that has internalized automotive E/E patterns before any instruction tuning.


## Training Objective

The training objective for CPT is standard next-token prediction: given a sequence of tokens x₁, x₂, …, xₙ₋₁, the model predicts xₙ.

The loss is cross-entropy averaged over all token positions in the chunk. Every token in the corpus simultaneously serves as both input context and training target and no manual labeling is needed. This is called **self-supervised learning**.

The model shifts its probability distribution toward automotive E/E text by repeatedly predicting the next token in domain-specific sequences.


## Model Choice

Running on a Kaggle T4 GPU, 4-bit quantization makes the model fit within the 15 GB VRAM constraint. Given its strong Unsloth support, availability as a pre-quantized 4-bit checkpoint, and a 128k token context window at only 1B parameters, Llama-3.2-1B was chosen as the base model.

**Why Base, not Instruct:** Llama-3.2-1B is available in both Base and Instruct variants. The Instruct variant has already been fine-tuned on instruction-following data.
CPT on raw domain text would degrade this instruction-following capability because the unstructured E/E documents as training data do not contain the prompt/response structure the model learned to expect (McCormick 2025). 

Therefore the pipeline is Base Model → CPT → SFT: first absorb domain knowledge from raw text, then teach instruction format on top.


## Why CPT Before SFT

Gururangan et al. (2020) show that Domain-Adaptive Pre-Training (DAPT) gains scale with domain distance from the pretraining corpus. Domains well-represented in web text benefit little, while structurally absent domains benefit substantially.

Regardless of the exact pretraining corpus, automotive E/E is a narrow engineering niche with its own terminology and specification formats that have no meaningful presence in general web text.

Concretely, without CPT, the model is more likely to interpret "CAN" as the English modal verb than as Controller Area Network.

Gururangan et al. (2020) further show that DAPT gains are largest precisely in low-resource settings, where few labeled examples are available. Fifteen instruction pairs cannot build domain knowledge but only teach instruction format on top of existing knowledge. When that knowledge is absent, the model hallucinates plausible-sounding but factually incorrect E/E content.


## Learning Rate Strategy

The learning rate strategy for CPT addresses different stability concerns.

CPT computes loss over every token in a 2048-token sequence, unlike SFT where loss is masked to output tokens only. This broader prediction target requires a more conservative learning rate to maintain training stability.

CPT also begins from a well-optimized pretrained checkpoint, not from random initialization. A full learning rate at step 0 risks destroying attention patterns that took hundreds of billions of training tokens to develop. A short linear warmup phase brings the learning rate to its target value gradually, keeping early updates small enough to preserve the pretrained structure. After warmup, a cosine decay schedule maintains a high learning rate during the main training phase for broad domain pattern absorption, then decays smoothly toward convergence.

Finally, not all model components require the same degree of adaptation. Vocabulary embeddings encode the most compressed form of semantic knowledge from pretraining. Applying the same update magnitude as attention and FFN layers risks destabilizing this structure. A decoupled, smaller embedding learning rate allows domain-specific token meaning to shift gradually without destroying the general semantic prior.


## Avoiding Catastrophic Forgetting

CPT must update the vocabulary embeddings so the model can learn domain-specific token meaning. But the embeddings hold the most compressed form of the model's general language knowledge, so modifying them is exactly what risks catastrophic forgetting. Two mechanisms keep this in check: LoRA and data mixing.


### LoRA

The 4-bit quantization used to save memory makes the base weights read-only. LoRA is therefore not just a regularization choice but necessary to make weight updates. LoRA inserts small trainable matrices alongside the frozen base weights, so the model retains everything encoded during pretraining while the adapters capture domain-specific shifts.

For CPT, embed_tokens and lm_head are included in the LoRA target modules alongside attention and FFN projections. In standard SFT, embeddings are frozen because vocabulary context is unchanged. For CPT, they need to be updated to shift domain-specific token meaning. 

Because the base weights remain frozen, LoRA structurally prevents overwriting the pretrained knowledge — the adapter adds to the base weights rather than replacing them. The low-rank constraint additionally limits how much the model's behavior can shift overall.

### Data Mixing

Freezing the base weights through LoRA largely prevents catastrophic forgetting, but training purely on domain text still risks degrading general language fluency. Gururangan et al. (2020) recommend mixing general-domain data into domain-specific CPT to preserve the pretraining prior.

Mixing a small percentage of a curated general-domain corpus into the training data alongside the E/E domain chunks keeps the domain signal dominant while providing a regularizing general-language signal.


## Toy Setup

| Setting | Value |
|---|---|
| Base model | `unsloth/Llama-3.2-1B-bnb-4bit` (4-bit) |
| Platform | Kaggle T4 (15 GB VRAM) |
| LoRA | r=64, α=32, rsLoRA; targets: attention + FFN + `embed_tokens` + `lm_head` |
| Learning rate | 1e-4 (embeddings decoupled at 1e-5), cosine schedule, 6 warmup steps |
| Training | 3 epochs / 174 steps, effective batch 8 (4×2), ~25 min |
| Data | 442 domain + 22 WikiText-2 (5%) = 464 train chunks; 50 validation |
| Trainable params | 316M / 1.81B (17.4%) — dominated by `embed_tokens` |
| Checkpoint | epoch 2 (`load_best_model_at_end` on validation loss) |


## Results

**Validation perplexity**

| | Pre-CPT | Post-CPT (epoch 2) |
|---|---|---|
| Perplexity | 10.27 | 8.89 |

**Training / validation loss**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 2.309 | 2.188 |
| 2 | 1.968 | **2.184** |
| 3 | 1.834 | 2.211 |

Training loss falls steadily while validation loss bottoms out at epoch 2 and rises at epoch 3 — the model begins overfitting, so the epoch-2 checkpoint is selected.

**Qualitative before/after.** On all three probe prompts the base model degenerates into repetition loops with no domain content. After CPT the loops disappear and the model produces coherent domain-style text using correct E/E vocabulary (ECUs, functional domains, service-oriented architecture, the functional-safety lifecycle). Factual precision, however, remains limited: the post-CPT CAN description correctly situates CAN among powertrain/chassis/body domains but wrongly attributes a "central controller" to what is a multi-master bus. This is the expected behavior of CPT — it shifts the token distribution toward domain style and vocabulary but does not reliably encode precise facts at this model and corpus scale.