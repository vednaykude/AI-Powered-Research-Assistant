# src/data_processing.py
import argparse
from pathlib import Path
from datasets import load_dataset
import numpy as np
import faiss
import json
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "meta.json"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512  # Max tokens per chunk for embeddings


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks by word count.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def truncate_chunk(chunk: str, tokenizer, max_tokens=MAX_TOKENS):
    """
    Truncate text to max tokens using tokenizer.
    """
    tokens = tokenizer.encode(chunk, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens)


def build_from_cnn_dailymail(split="train", max_samples=5000, chunk_size=500, overlap=100):
    """
    Load CNN/DailyMail dataset, split into chunks, embed, and save FAISS index + meta.
    """
    ds = load_dataset("ccdv/cnn_dailymail", "3.0.0", split=split)
    texts = []
    metas = []

    # Load tokenizer for truncation
    tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL)

    for i, item in enumerate(tqdm(ds, total=min(len(ds), max_samples))):
        if i >= max_samples:
            break
        article = item["article"].strip()
        summary = item.get("highlights", "")
        pid = f"cnn_{split}_{i}"
        chunks = chunk_text(article, chunk_size=chunk_size, overlap=overlap)

        # Truncate each chunk
        chunks = [truncate_chunk(c, tokenizer, max_tokens=MAX_TOKENS) for c in chunks]

        for j, c in enumerate(chunks):
            texts.append(c)
            metas.append({
                "paper_id": pid,
                "chunk_id": j,
                "title": item.get("title") or f"cnn_article_{i}",
                "summary": summary
            })

    print(f"Prepared {len(texts)} chunks")

    # Embeddings
    model = SentenceTransformer(EMB_MODEL)
    batch_size = 64
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings).astype("float32")

    # Build FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity using inner product
    faiss.normalize_L2(embeddings)  # normalize embeddings
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"FAISS index saved to {INDEX_PATH}")

    # Save metadata
    with open(META_PATH, "w") as f:
        json.dump({"texts": texts, "metas": metas}, f)
    print(f"Meta saved to {META_PATH}")

    return INDEX_PATH, META_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()
    build_from_cnn_dailymail(
        split=args.split,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
