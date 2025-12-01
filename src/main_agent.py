# src/main_agent.py
"""
Main Agent backend for retrieval, summarization, and QA:
- Load FAISS index and metadata
- Semantic search (batched)
- Summarization (brief/detailed)
- Question answering over retrieved chunks
- Iterative RAG loop
"""

from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Tuple

# Directories
DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "meta.json"

# Default models
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUM_MODEL = "google/flan-t5-large"
QA_MODEL = "distilbert-base-uncased-distilled-squad"

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    def __init__(self, emb_model=EMB_MODEL, sum_model=SUM_MODEL, qa_model=QA_MODEL):
        """
        Initialize the agent with embedding, summarization, and QA models.
        Loads FAISS index and metadata.
        """
        self.embedder = SentenceTransformer(emb_model, device=device)

        # Load FAISS index and metadata
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                "Run src.data_processing to create FAISS index and meta.json"
            )
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        self.texts = meta["texts"]
        self.metas = meta["metas"]

        # Summarization pipeline
        self.summarizer = pipeline(
            "summarization", model=sum_model, device=0 if device=="cuda" else -1
        )
        self.sum_tokenizer = AutoTokenizer.from_pretrained(sum_model)

        # QA pipeline
        self.qa_pipe = pipeline(
            "question-answering", model=qa_model, device=0 if device=="cuda" else -1
        )
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model)

    def truncate_text(self, text: str, max_tokens: int, tokenizer) -> str:
        """Truncate text to max_tokens using tokenizer."""
        tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return tokenizer.decode(tokens)

    def search(self, query: str, top_k=5) -> List[Tuple[int, float]]:
        """Search FAISS index for top_k most similar chunks to the query."""
        q_emb = self.embedder.encode(query, convert_to_numpy=True).astype("float32")
        if q_emb.ndim == 1:
            q_emb = np.expand_dims(q_emb, axis=0)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        return [(int(idx), float(score)) for idx, score in zip(I[0], D[0])]

    def retrieve_chunks(self, query: str, top_k=5) -> List[dict]:
        """Retrieve top-k chunks with text and metadata."""
        hits = self.search(query, top_k=top_k)
        return [
            {"index": idx, "score": score, "text": self.texts[idx], "meta": self.metas[idx]}
            for idx, score in hits
        ]

    def summarize_brief(self, text: str, max_length=120) -> str:
        """Generate a brief summary of the text."""
        safe_text = self.truncate_text(text, max_tokens=512, tokenizer=self.sum_tokenizer)
        out = self.summarizer(safe_text, max_length=max_length, min_length=20, do_sample=False)
        return out[0].get("summary_text", "") if isinstance(out, list) else str(out)

    def summarize_detailed(self, texts: List[str], max_length=300) -> str:
        """
        Summarize multiple chunks by first summarizing each chunk individually,
        then combining the summaries for a final detailed summary.
        """
        # Step 1: Summarize each chunk briefly
        brief_summaries = [self.summarize_brief(t, max_length=80) for t in texts]
        joined_summary = "\n\n".join(brief_summaries)
        # Step 2: Final detailed summary
        out = self.summarizer(joined_summary, max_length=max_length, min_length=80, do_sample=False)
        return out[0].get("summary_text", "") if isinstance(out, list) else str(out)

    def answer_question(self, question: str, context: str) -> str:
        """Answer a question given a context."""
        safe_context = self.truncate_text(context, max_tokens=512, tokenizer=self.qa_tokenizer)
        try:
            out = self.qa_pipe(question=question, context=safe_context)
            return out.get("answer", str(out)) if isinstance(out, dict) else str(out)
        except Exception as e:
            return f"QA failed: {e}"

    def iterative_rag(self, query: str, iterations=2, top_k=5) -> str:
        """
        Iterative Retrieval-Augmented Generation loop.
        Summarizes chunks first to avoid token overflow.
        """
        current_query = query
        last_answer = ""
        for _ in range(iterations):
            retrieved = self.retrieve_chunks(current_query, top_k=top_k)
            top_texts = [r["text"] for r in retrieved]

            # Summarize each chunk first
            brief_summaries = [self.summarize_brief(t, max_length=80) for t in top_texts]
            context = "\n\n".join(brief_summaries)
            prompt = f"Context:\n{context}\n\nQuestion: {current_query}\nAnswer:"
            
            ans = self.summarizer(prompt, max_length=256, min_length=40, do_sample=False)
            answer_text = ans[0].get("summary_text", "") if isinstance(ans, list) else str(ans)
            last_answer = answer_text
            # Refine query using first 200 characters
            current_query = answer_text[:200]
        return last_answer

    def get_similarity_percentage(self, raw_score: float) -> float:
        """Convert raw FAISS score to a percentage."""
        return max(0.0, min(100.0, raw_score * 100.0))
