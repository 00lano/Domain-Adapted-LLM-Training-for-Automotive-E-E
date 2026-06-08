# Part 1: Data Corpus - strategy wirte-up

---

How would you build a training corpus for the E/E domain from public sources?
- Where does usable text come from? How do you ensure quality and filter out noise?
- How do you handle the domain-specific vocabulary that a general tokenizer may not cover well?
Deliverable: Strategy write-up + a script that assembles a small sample corpus.

---

## Why General LLMs Struggle with This Domain

General-purpose language models are pretrained on broad data corpora where Automotive E/E related data is structurally underrepresented in these sources. This creates two distinct gaps:

**Domain Knowledge Gap:** E/E-specific entities and their semantic relationships are absent or sparse in general pretraining data: AUTOSAR layer semantics, ASIL classification, UDS service IDs, CAN arbitration rules. This can lead base models to produce plausible-sounding but factually incorrect answers.

**Out-of-Distribution (OOD) Text Gap:** E/E documentation follows strict formatting conventions, like UDS byte sequences, DBC notation or ARXML schemas, which are rarely seen in web text. When format and style fall outside the model's training distribution, it assigns low probability to individual tokens: the text is out-of-distribution even if every word is individually known (McCormick, 2025).

Continued Pre-Training (CPT) on in-domain text addresses both gaps simultaneously, shifting the model's token distribution toward automotive E/E patterns.
  
## Corpus Construction: Citation-Driven Strategy

The corpus is built from three categories of English-language E/E sources: academic papers, Wikipedia articles, and domain-specific web pages. Rather than scraping broadly, I started with a citation-driven strategy, using Eder (TUM, 2022), "Automatic Exploration of Automotive E/E Architectures", as the seed document and its reference list as a curated map to the most relevant E/E literature.

Reference sections are parsed for arXiv IDs, which are downloaded directly.
Wikipedia articles were manually curated to cover the full E/E stack: communication protocols (CAN, LIN, FlexRay, Ethernet), system architecture (ECU, AUTOSAR), safety (ISO 26262, functional safety), and diagnostics (OBD-II, UDS). Domain-specific web pages complement the paper corpus with more accessible coverage of E/E concepts.

## Preprocessing Pipeline

### Text Extraction

PDFs are extracted with pypdf; HTML sources are cleaned with BeautifulSoup4 to strip navigation, scripts, and footers.

E/E documentation is schema-heavy: architecture diagrams, signal tables, and AUTOSAR layer figures are embedded as images or vector graphics. This creates a structural extraction problem:

  ┌────────────────────┬─────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
  │       Level        │          Approach           │                               Capability                               │
  ├────────────────────┼─────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Toy (this project) │ pypdf                       │ Plain text only; images and tables silently dropped                    │
  ├────────────────────┼─────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Pragmatic          │ pdfplumber                  │ Extracts embedded tables as structured text; images still lost         │
  ├────────────────────┼─────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Production         │ marker-pdf / multimodal OCR │ Renders pages; extracts text from figures, diagrams, and signal tables  │
  └────────────────────┴─────────────────────────────┴────────────────────────────────────────────────────────────────────────┘

  For this prototype, image and table loss is accepted and documented in the Limitations section below.

### Language Filtering
Documents are filtered to English using langdetect. At production scale, multilingual corpora — including German-language automotive standards and AUTOSAR specifications — would broaden domain coverage significantly.

### Quality Filtering

Quality filtering follows Gopher (Rae et al., 2021): successive pipeline stages remove low-quality documents before chunking. Filters include minimum/maximum word count, mean word length, symbol-to-word ratio, bullet-point ratio, and a stop-word check.

PDF documents carry an additional chunk-level filter: pypdf often extracts mathematical formulas as non-ASCII garbage (□◆(1−λ)·Ck); chunks with more than 10% non-ASCII characters are discarded.

### Deduplication

Exact duplicates are removed via SHA256 hashing. Near-duplicates — e.g. the same paper indexed under multiple URLs — are identified using MinHash with 13-gram Jaccard similarity above 0.8 and discarded.

### Chunking

Chunk size is set to 2048 tokens — a conservative choice that fits comfortably on a T4 (~15 GB VRAM) with Unsloth + LoRA and leaves room for a reasonable batch size. While FlashAttention reduces attention memory to O(n), compute still scales quadratically with sequence length (Vaswani et al., 2017), and longer sequences reduce effective batch size accordingly.
  
### Train/Val Split

The filtered, deduplicated corpus is split 90/10 into train and validation sets. Validation loss during CPT serves as an early convergence signal without waiting for the Part 4 evaluation.

### Domain-Specific Vocabulary

  General tokenizers (trained on web-scale text) tokenize E/E terms like AUTOSAR, CAN-FD, or SOME/IP into suboptimal subword fragments. The clean solution — vocabulary extension with retraining of embedding layers — is complex and risks training instability.

  The pragmatic approach used here: CPT progressively adjusts the embeddings of existing tokens toward the E/E context through continued exposure. This is sufficient for a toy prototype.

  Production-level fix: extend the tokenizer vocabulary with domain-specific tokens, initialize their embeddings as the mean of their constituent subword  fragments, and retrain embedding and LM head layers alongside CPT. This requires a sufficiently large corpus for stable embedding learning — infeasible at toy scale.
  
## Limitations

  ┌─────────────────────────────────────────┬─────────────────────────────────────────────────┬─────────────────────────────────────────────────────────┐
  │               Limitation                │                     Impact                      │                     Production fix                      │
  ├─────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Small corpus (~101 chunks, ~206k        │ 2–3 orders of magnitude below production        │ 10M–100M tokens minimum                                 │
  │ tokens)                                 │ minimum                                         │                                                         │
  ├─────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Image/table loss (pypdf)                │ Significant E/E content silently dropped        │ marker-pdf or multimodal OCR                            │
  ├─────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Formula loss (non-ASCII filter)         │ Mathematical notation discarded                 │ LaTeX-aware PDF extraction                              │
  ├─────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ No data mixing                          │ Risk of catastrophic forgetting during CPT      │ 10–20% general-domain text mixed in (addressed in Part  │
  │                                         │                                                 │ 2)                                                      │
  ├─────────────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┤
  │ Tokenizer vocabulary mismatch           │ Suboptimal subword splits for E/E terms         │ Vocabulary extension + embedding retraining             │
  └─────────────────────────────────────────┴─────────────────────────────────────────────────┴─────────────────────────────────────────────────────────┘