# Part 1: Data Corpus — Strategy Write-up

---
**Questions:**

How would you build a training corpus for the E/E domain from public sources?
- Where does usable text come from? How do you ensure quality and filter out noise?
- How do you handle the domain-specific vocabulary that a general tokenizer may not cover well?

**Deliverable:** Strategy write-up + a script that assembles a small sample corpus.

---

## Why General LLMs Struggle with This Domain

General-purpose language models are pretrained on broad data corpora where Automotive E/E related data is structurally underrepresented. This creates two distinct gaps:

**Domain Knowledge Gap:** E/E-specific entities and their semantic relationships are absent or sparse in general pretraining data: AUTOSAR layer semantics, ASIL classification, UDS service IDs, CAN arbitration rules. This can lead base models to produce plausible-sounding but factually incorrect answers.

**Out-of-Distribution (OOD) Text Gap:** E/E documentation follows strict formatting conventions, like UDS byte sequences, DBC notation or ARXML schemas, which are rarely seen in web text. When format and style fall outside the model's training distribution, it assigns low probability to individual tokens: the text is out-of-distribution even if every word is individually known (McCormick, 2025).

Continued Pre-Training (CPT) on in-domain text addresses both gaps simultaneously, shifting the model's token distribution toward automotive E/E patterns.

## Corpus Construction: Citation-Driven Strategy

The corpus is built from three categories of English-language E/E sources: academic papers, Wikipedia articles, and domain-specific web pages. Rather than scraping broadly, I started with a citation-driven strategy, using Eder (TUM, 2022), "Automatic Exploration of Automotive E/E Architectures", as the seed document and its reference list as a curated map to the most relevant E/E literature.

Reference sections are parsed and matched against the Semantic Scholar API to retrieve open-access PDFs where available. Wikipedia articles were manually curated to cover the full E/E stack: communication protocols (CAN, LIN, FlexRay, Ethernet), system architecture (ECU, AUTOSAR), safety (ISO 26262, functional safety), and diagnostics (OBD-II, UDS). Domain-specific web pages complement the paper corpus with more accessible coverage of E/E concepts.

## Preprocessing Pipeline

### Text Extraction

PDFs are extracted with pypdf. Wikipedia articles are fetched via the Wikipedia API and saved as plain text. HTML sources from domain-specific web pages are cleaned with BeautifulSoup4 to strip navigation, scripts, and footers.

E/E documentation is schema-heavy — embedded images, signal tables, and architecture diagrams cannot be extracted with pypdf and are silently dropped; see Limitations.

### Language Filtering

Documents are filtered to English using langdetect. At production scale, multilingual corpora — including German-language automotive standards and AUTOSAR specifications — would broaden domain coverage significantly.

### Quality Filtering

Quality filtering follows Gopher (Rae et al., 2021): successive pipeline stages remove low-quality documents before chunking. Filters include minimum/maximum word count, mean word length, symbol-to-word ratio, bullet-point ratio, and a stop-word check.

For this prototype, two filters are implemented: word count (50–100,000 words) and non-ASCII character ratio (< 10% per chunk). The non-ASCII filter catches PDF chunks where pypdf extracted mathematical formulas as garbage characters (□◆(1−λ)·Ck); the word count filter catches navigation fragments and short snippets from web sources. Additional Gopher heuristics are omitted as they are unlikely to trigger on the curated source types used here but should be implemented on production scale.

### Deduplication

Exact duplicates are removed via SHA256 hashing. Near-duplicate detection via MinHash is not implemented. At toy scale with a curated corpus, the same document appearing twice is unlikely. At production scale, MinHash with 13-gram Jaccard similarity above 0.8 should be added.

### Chunking

Chunk size is set to 2048 tokens — a conservative choice that fits comfortably on a T4 (~15 GB VRAM) with Unsloth + LoRA and leaves room for a reasonable batch size. While FlashAttention reduces attention memory to O(n), compute still scales quadratically with sequence length (Vaswani et al., 2017), and longer sequences reduce effective batch size accordingly.

### Train/Val Split

The filtered, deduplicated corpus is split 90/10 into train and validation sets. Validation loss during CPT serves as an early convergence signal without waiting for the Part 4 evaluation.


## Limitations