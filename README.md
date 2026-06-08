# Domain-Adapted LLM Training for Automotive E/E

This project is an end-to-end demonstration of adapting a small open-source LLM to the automotive electrical/electronic (E/E) domain using only public data and free compute in a toy scenario.


## Motivation

General-purpose language models are pretrained on broad web corpora — automotive E/E text is structurally underrepresented in these sources.
CPT on curated in-domain data followed by SFT on targeted instruction pairs directly improves both domain knowledge and task behavior.


## Overview

| Part | Task | Deliverable |
|------|------|-------------|
| 1 | Data corpus construction | Strategy write-up + corpus assembly script |
| 2 | Continued Pre-Training (CPT) | Design write-up + training notebook |
| 3 | Supervised Fine-Tuning (SFT) | 15 instruction pairs (JSONL) + SFT write-up |
| 4 | Evaluation | Eval plan + eval questions with reference answers |

All steps use only public data and free compute.


## Model Choice  

**Llama-3.2-1B-Base** (`meta-llama/Llama-3.2-1B`)

| Property           | Value                         |
| ------------------ | ----------------------------- |
| Publisher          | Meta (2024)                   |
| Parameters         | 1.24B                         |
| Tokenizer          | TikToken-based                |
| Context window     | 128k tokens                   |

## Prerequisites
  
**HuggingFace Access (required for Part 1 + Part 2)**

Llama-3.2-1B is a gated model. Before running any script or notebook:

1. Accept the model license at: https://huggingface.co/meta-llama/Llama-3.2-1B
2. Create a HuggingFace access token (Read): https://huggingface.co/settings/tokens
3. Authenticate:
huggingface-cli login
    or set the environment variable:
export HF_TOKEN=your_token_here

## Repository Structure

```
assignment/
├── README.md
├── part1_data/
│   ├── corpus_strategy.md       # data corpus strategy write-up
│   └── build_corpus.py          # sample corpus assembly script
├── part2_cpt/
│   ├── cpt_design.md            # CPT design write-up
│   └── cpt_notebook.ipynb       # training notebook (Colab)
├── part3_sft/
│   ├── sft_approach.md          # SFT approach write-up
│   └── instruction_pairs.jsonl  # 15 instruction/response pairs
└── part4_eval/
    ├── eval_plan.md             # evaluation plan
    └── eval_questions.md        # eval questions + reference answers
```