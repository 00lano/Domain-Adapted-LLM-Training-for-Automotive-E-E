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


## Repository Structure

```
assignment/
├── README.md
├── part1_data/
│   ├── corpus_strategy.md       # Write-up: data selection and filtering rationale
│   └── build_corpus.py          # Script: citation-driven corpus assembly + preprocessing
├── part2_cpt/
│   ├── cpt_design.md            # Write-up: CPT objective, LoRA config, hyperparameters
│   └── cpt_notebook.ipynb       # Notebook: runnable CPT with Unsloth on Colab
├── part3_sft/
│   ├── sft_approach.md          # Write-up: instruction pair design, LoRA vs full FT
│   └── instruction_pairs.jsonl  # 15 E/E domain instruction/response pairs
└── part4_eval/
    ├── eval_plan.md             # Write-up: evaluation methodology, ROUGE-L rationale
    └── eval_questions.md        # 10 questions with reference answers + ROUGE-L results
...
```