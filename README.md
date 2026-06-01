# Domain-Adapted LLM Training for Automotive E/E

  

---

  

## Overview

  

This project demonstrates how to adapt a small open-source language model to the automotive E/E domain end-to-end in a toy szenario:

  

- Part 1: data corpus construction

- Part 2: continued pre-training (CPT)

- Part 3: supervised fine-tuning (SFT)

- Part 4: evaluation

  

All steps use only public data and free compute (Colab/local).

  

## Model Choice

  

**Llama-3.2-1B-Base** (`meta-llama/Llama-3.2-1B`)

  

Selected as the base model within the ≤1.5B parameter constraint:

  

- published by Meta (2024)

- trained on 9T tokens with a strong English focus

- 128k token context window

- Efficient 32k vocabulary

- Base variant (not instruct)