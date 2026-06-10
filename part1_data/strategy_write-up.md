# Part 1: Data Corpus — Strategy Write-up


The quality of the training corpus directly determines the quality of domain adaptation: a small, high-quality corpus consistently outperforms a large, noisy one for CPT (Gururangan et al., 2020).


## Why General LLMs Struggle with This Domain

General-purpose language models are pretrained on broad data corpora where automotive E/E related data is structurally underrepresented. This creates two distinct gaps:

**Domain Knowledge Gap:** E/E-specific entities and their semantic relationships are absent or sparse in general pretraining data: AUTOSAR layer semantics, ASIL classification, UDS service IDs, CAN arbitration rules. This can lead base models to produce plausible-sounding but factually incorrect answers.

**Out-of-Distribution (OOD) Text Gap:** E/E documentation follows strict formatting conventions, like UDS byte sequences, DBC notation or ARXML schemas, which are rarely seen in web text. When format and style fall outside the model's training distribution, it assigns low probability to individual tokens: the text is out-of-distribution even if every word is individually known (McCormick, 2025).

Continued Pre-Training (CPT) on in-domain text addresses both gaps simultaneously, shifting the model's token distribution toward automotive E/E patterns.


## Corpus Construction

Usable E/E text comes from three main categories of public sources: academic papers, reference articles (Wikipedia) and domain-specific web pages. Each contributes differently: papers provide precise technical depth, Wikipedia broad conceptual coverage and web pages accessible explanations.

Rather than scraping broadly, the corpus can be built citation-driven: starting from a small set of high-quality seed documents and following their reference lists to collect further domain literature. This keeps relevance density high because every document traces back to a curated, on-topic source and automates expansion from a few seeds instead of requiring each source to be found by hand.


## Preprocessing Pipeline

### Data Extraction

Different source types require different extraction approaches. Plain text sources can be used directly. HTML pages require stripping of navigation, scripts, and footers to isolate the main content. PDFs are the most common format for academic papers and technical standards but also the most problematic: text can be extracted via parser libraries, but embedded images, signal tables, architecture diagrams, and mathematical formulas are either lost or extracted as garbled characters. For formula-heavy documents, specialized extraction tools or manual preprocessing may be needed.

The extraction method determines what ends up in the corpus. Schema-heavy E/E documentation is particularly affected since much of its technical content is conveyed through diagrams and tables rather than prose.


### Language Filtering

Language filtering ensures corpus consistency by keeping only documents in the target language. The choice depends on the domain and intended training distribution. E/E documentation, for example, exists in both English and German, so filtering decisions should reflect which language coverage the model needs.


### Quality Filtering 

Quality filtering can be used to remove low-quality documents before chunking. Standard filters include word count thresholds to remove fragments and excessively long documents, mean word length to catch garbled text, symbol-to-word ratio to filter formula-heavy extractions, bullet-point ratio to remove low-information list pages and a stop-word check, since natural prose contains many common function words ("the", "and", "of") while keyword lists or garbled text do not.

Beyond general text quality, a relevance filter removes off-domain documents that slip in through citation lists, keeping only text clearly related to automotive E/E.


### Deduplication

Exact deduplication via hash comparison removes identical documents. Near-duplicate detection via MinHash with n-gram Jaccard similarity catches documents that are substantially similar but not identical.
Deduplication matters because repeated text causes the model to over-memorize duplicated passages rather than generalize and wastes training compute on redundant data.


### Chunking

Documents are split into fixed-length token sequences matching the model's training context window. Longer chunks preserve more document context but reduce effective batch size, since attention compute scales quadratically with sequence length (Vaswani et al., 2017).


### Train/Val Split

The final corpus can be split into train and validation sets. Validation loss during CPT serves as an early convergence signal without waiting for downstream task evaluation.


## Domain-Specific Vocabulary

A tokenizer splits text into subword units, with frequent words tending to form a single token and rarer words broken into several fragments. Because general tokenizers are trained on web text, E/E terminology falls into the second category. A clean split is harmless and the model reconstructs meaning from the pieces but the cost appears with terms that fragment into meaningless units: acronyms and identifiers like "AUTOSAR", "SOME/IP", or "0xF190" may split into pieces that carry no signal, and every multi-token term lengthens sequences and adds compute.

Two approaches handle this, differing mainly in effort:

- **Progressive embedding adjustment (used here).** The tokenizer is left unchanged; CPT gradually shifts the embeddings of the existing fragment tokens toward their E/E meaning in context. No structural change, and sufficient at this corpus scale.
- **Vocabulary extension.** Dedicated tokens for frequent domain terms are added to the tokenizer, each initialized as the mean of the fragments it replaces, then trained alongside CPT. More token-efficient and more principled, but a new token only earns its place after many occurrences — so it needs a large corpus.


## Toy Setup

| Step | Implementation | Detail |
|---|---|---|
| Data Sources | Wikipedia API, seed PDFs, web pages | 10 Wikipedia articles, 3 web pages, 7 seed PDFs + cited papers via OpenAlex API |
| Text Extraction | pypdf, wikipediaapi, BeautifulSoup4 | HTML: strips script, style, nav, footer, header tags |
| Language Filtering | langdetect | English only |
| Quality Filtering | Word count + non-ASCII ratio | < 50 or > 100,000 words removed; non-ASCII > 10% removed |
| Relevance Filtering | Keyword check | ≥ 3 E/E keywords required (AUTOSAR, CAN bus, ECU, ISO 26262, ...) |
| Deduplication | SHA256 hashing | Exact duplicates only |
| Chunking | 2048 tokens | Llama-3.2-1B tokenizer, no overlap |
| Train/Val Split | 90/10 | Fixed index split |


## Results

Running the pipeline end-to-end produced the following corpus:

| Stage | Documents |
|---|---|
| Raw collected (7 seeds + 70 reference PDFs + 13 text sources) | 90 |
| After filtering (language, quality, relevance) | 71 |
| After deduplication | 42 |

Filtering removed 19 documents: 14 below the relevance threshold (< 3 E/E keywords), 3 non-English, 1 too short, 1 above the non-ASCII ratio. Deduplication removed a further 29 documents — exact duplicates, mostly the same open-access papers retrieved under different reference entries via the OpenAlex API.

The 42 unique documents were chunked into 492 sequences of 2048 tokens (962,886 tokens total) and split 442 train / 50 validation.
