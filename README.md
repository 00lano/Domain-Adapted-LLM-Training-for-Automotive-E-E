# Domain-Adapted LLM Training for Automotive E/E

This project is an end-to-end demonstration of adapting a small open-source LLM to the automotive electrical/electronic (E/E) domain using only public data and free computein a toy scenario.

---

## Motivation

General-purpose language models are pretrained on broad web corpora — automotive E/E text is structurally underrepresented in these sources.
CPT on curated in-domain data followed by SFT on targeted instruction pairs directly improves both domain knowledge and task behavior.

---

## Overview

| Part | Task | Deliverable |
|------|------|-------------|
| 1 | Data corpus construction | Strategy write-up + corpus assembly script |
| 2 | Continued Pre-Training (CPT) | Design write-up + training notebook |
| 3 | Supervised Fine-Tuning (SFT) | 15 instruction pairs (JSONL) + SFT write-up |
| 4 | Evaluation | Eval plan + ROUGE-L results (Base → CPT → CPT+SFT) |

All steps use only public data and free compute.

---

## Model Choice  

**Llama-3.2-1B-Base** (`meta-llama/Llama-3.2-1B`)

| Property           | Value                         |
| ------------------ | ----------------------------- |
| Publisher          | Meta (2024)                   |
| Parameters         | 1.24B                         |
| Pretraining tokens | ~9T                           |
| Context window     | 128k tokens                   |

?? Selected for the ≤1.5B parameter constraint: strong English pretraining, efficient vocabulary, and broad community support for fine-tuning tooling (Unsloth, PEFT). ??