import requests # xrXiv / WEB-URLs
import wikipediaapi # Wikipedia
from bs4 import BeautifulSoup
from pathlib import Path
import pypdf
from langdetect import detect
import hashlib
from transformers import AutoTokenizer
import json
import re
import time

OUTPUT_DIR = Path(__file__).parent / "data_corpus"
CHUNK_SIZE = 2048
EE_KEYWORDS = ["autosar", "can bus", "ecu", "automotive", "e/e",
                  "powertrain", "adas", "iso 26262", "flexray", "lin bus",
                  "electronic control unit", "on-board diagnostic",
                  "functional safety", "asil", "some/ip"]

SEED_PDFS = list((Path(__file__).parent / "seed_documents").glob("*.pdf"))

WIKIPEDIA_ARTICLES = [
      "AUTOSAR",
      "CAN bus",
      "Electronic control unit",
      "ISO 26262",
      "Unified Diagnostic Services",
      "FlexRay",
      "LIN bus",
      "Automotive Ethernet",
      "Functional safety",
      "OBD-II",
  ]

WEB_URLS = [
      "https://flex-product.com/knowledge/can-bus/introdutory-guide-for-beginners",
      "https://automotivetechis.wordpress.com/autosar-concepts/",
      "https://mohamedayman23.medium.com/inside-the-autosar-can-communication-stack-a-layered-architecture-explained-8e92e0f265f9",
  ] 

def fetch_wikipedia(articles, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="ee-corpus")
    for article in articles:
        page = wiki.page(article)
        with open(output_dir / f"{article}.txt", "w", encoding="utf-8") as f:
            f.write(page.text)

def fetch_webpages(urls, output_dir):
      output_dir.mkdir(parents=True, exist_ok=True)
      for url in urls:
          try:
              response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
              soup = BeautifulSoup(response.text, "html.parser")
              for tag in soup(["script", "style", "nav", "footer", "header"]):
                  tag.decompose()
              text = soup.get_text(separator="\n")
              lines = [l.strip() for l in text.splitlines()]
              text = "\n".join(l for l in lines if l)
              slug = url.split("//")[-1].replace("/", "_").replace(".", "_")
              with open(output_dir / (slug[:60] + ".txt"), "w", encoding="utf-8") as f:
                  f.write(text)
          except Exception as e:
              print(f"Skipping {url}: {e}")
  
def fetch_papers_from_references(pdf_path, output_dir, max_papers=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not pdf_path.exists():
        print(f"Seed PDF not found: {pdf_path.name} — place it manually in part1_data/")
        return
    seed_slug = pdf_path.stem[:20]
    text = "".join(page.extract_text() or "" for page in pypdf.PdfReader(pdf_path).pages)
    bib_match = re.search(r'(Bibliography|References|REFERENCES)\s*\n', text)
    bib_text = text[bib_match.start():] if bib_match else text
    refs = re.findall(r'\[(\d+)\]\s+(.+?)(?=\[\d+\]|\Z)', bib_text, re.DOTALL)
    print(f"Found {len(refs)} references in {pdf_path.name}")
    downloaded = 0
    for num, ref_text in refs[:max_papers]:
        out_path = output_dir / f"{seed_slug}_ref{num.zfill(3)}.pdf"
        if out_path.exists():
            downloaded += 1
            continue
        clean = re.sub(r'[^\x00-\x7F]+', ' ', ref_text)
        query = " ".join(clean.split())[:200]
        print(f"  Searching [{num}]...", end="\r")
        try:
            r = requests.get(
                "https://api.openalex.org/works",
                params={"search": query, "filter": "open_access.is_oa:true", "select": "title,open_access", "per_page": 1},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            data = r.json().get("results", [])
            if not data or not data[0].get("open_access", {}).get("oa_url"):
                time.sleep(0.5)
                continue
            pdf_url = data[0]["open_access"]["oa_url"]
            pdf_r = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if pdf_r.status_code == 200:
                out_path.write_bytes(pdf_r.content)
                downloaded += 1
                print(f"  [{num}] Downloaded")
        except Exception as e:
            print(f"Failed ref [{num}]: {e}")
        time.sleep(0.2)
    print(f"Downloaded {downloaded}/{len(refs)} papers from references")
            
def extract_texts(output_dir):
  docs = []
  for datei in output_dir.iterdir():
    if datei.suffix == ".pdf":
        try:
            reader = pypdf.PdfReader(datei)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            docs.append(text)
        except Exception as e:
            print(f"Skipping {datei.name}: {e}")
    elif datei.suffix == ".txt":
        docs.append(open(datei, encoding="utf-8").read())
  return docs

def is_english(text):
    return detect(text) == "en"

def quality_ok(text):
    words = text.split()
    if len(words) < 50:
        return "too_short"
    if len(words) > 100000:
        return "too_long"
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / len(text) > 0.1:
        return "non_ascii"
    return None

def is_relevant(text):
    t = text.lower()
    if sum(1 for kw in EE_KEYWORDS if kw in t) < 3:
        return "not_relevant"
    return None

def filter_docs(docs):
    filtered = []
    n_not_english = 0
    n_too_short = 0
    n_too_long = 0
    n_non_ascii = 0
    n_not_relevant = 0
    for doc in docs:
        if not is_english(doc):
            n_not_english += 1
            continue
        reason = quality_ok(doc)
        if reason == "too_short":
            n_too_short += 1
            continue
        elif reason == "too_long":
            n_too_long += 1
            continue
        elif reason == "non_ascii":
            n_non_ascii += 1
            continue
        if is_relevant(doc) == "not_relevant":
            n_not_relevant += 1
            continue
        filtered.append(doc)
    print(f"Total: {len(docs)} | Passed: {len(filtered)} | Not English: {n_not_english} | Too short: {n_too_short} | Too long: {n_too_long} | Non-ASCII: {n_non_ascii} | Not relevant: {n_not_relevant}")
    return filtered
  
def deduplicate(docs):
    seen = set()
    unique = []
    for doc in docs:
        h = hashlib.sha256(doc.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique

def chunk_docs(docs):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    chunks = []
    for doc in docs:
        tokens = tokenizer.encode(doc)  # → Liste von Token-IDs
        for i in range(0, len(tokens), CHUNK_SIZE):
            chunk = tokens[i : i + CHUNK_SIZE]
            chunks.append(chunk)
    return chunks

def save_splits(chunks, output_dir):
    split = int(len(chunks) * 0.9)
    train = chunks[:split]
    val = chunks[split:]

    with open(output_dir / "train.json", "w") as f:
            json.dump(train, f)
    with open(output_dir / "val.json", "w") as f:
            json.dump(val, f)

def main():
    fetch_wikipedia(WIKIPEDIA_ARTICLES, OUTPUT_DIR / "raw")
    fetch_webpages(WEB_URLS, OUTPUT_DIR / "raw")
    for seed in SEED_PDFS:
        fetch_papers_from_references(seed, OUTPUT_DIR / "raw")
      
    docs = extract_texts(OUTPUT_DIR / "raw")
  
    for seed in SEED_PDFS:
        if seed.exists():
            seed_text = "".join(page.extract_text() or "" for page in pypdf.PdfReader(seed).pages)
            docs.append(seed_text)

    docs = filter_docs(docs) # Filter: language + quality
  
    docs = deduplicate(docs) # Deduplicate
    print(f"After dedup: {len(docs)} docs")
  
    chunks = chunk_docs(docs)
    total_tokens = sum(len(c) for c in chunks)
    print(f"Chunks: {len(chunks)} | Tokens: {total_tokens} | Train: {int(len(chunks)*0.9)} | Val: {len(chunks) - int(len(chunks)*0.9)}") # Chunks

    save_splits(chunks, OUTPUT_DIR) # Split & save

if __name__ == "__main__":
      main()