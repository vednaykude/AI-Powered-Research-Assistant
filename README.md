# Retrieval-Augmented Generation (RAG) on CNN/DailyMail Dataset  
### CS Course Final Project – December 8

## **Abstract**
This project implements a Retrieval-Augmented Generation (RAG) system using the CNN/DailyMail dataset. The objective is to explore whether retrieval improves LLM summarization quality by grounding outputs in retrieved article content. I build a reproducible pipeline including dataset preprocessing, document chunking, embedding generation, vector search, retrieval, and final answer generation using an open-source LLM. Experiments show that retrieval improves factual accuracy compared to zero-context generation.

---

# **Overview**

## **Problem Motivation**
Large language models often struggle with hallucinations and factual consistency, especially in summarization or question-answering tasks. The CNN/DailyMail dataset consists of news articles and summaries, making it ideal for testing document-grounded generation.

The problem addressed:
> *Can retrieval improve the accuracy and quality of summarization on news articles?*

This is relevant to real-world applications:
- News summarization
- Search engines
- Question answering over long documents
- Enterprise knowledge retrieval (legal, medical, financial)

---

## **Why the Problem Is Interesting**
RAG is becoming one of the most important modern AI techniques. It allows small models to perform well on tasks requiring factual recall. Evaluating RAG on a real-world dataset like CNN/DailyMail allows us to:
- test retrieval effectiveness
- measure quality improvements
- compare to baseline LLM summarization
- analyze limitations of dense retrieval

---

## **Approach Summary**
The project implements a clean RAG pipeline:

1. **Load dataset from HuggingFace (subset of CNN/DailyMail).**
2. **Preprocess and chunk text** into ~500-token segments.
3. **Generate embeddings** using sentence-transformers (`all-MiniLM-L6-v2`).
4. **Store embeddings in Chroma** for semantic search.
5. **Build retrieval pipeline** (top-k similarity search).
6. **Use a local LLM** (e.g., Llama-3 8B or Mistral 7B via HuggingFace Transformer or ctransformers) to generate summaries.
7. **Evaluate performance** using ROUGE scores.
8. **Compare:**
   - baseline summarizer  
   - RAG-enhanced summarizer  
   - ablation tests with different k

---

## **Key Components**
- **Data preprocessing**: cleaning, chunking, normalization  
- **Embedding generation**  
- **Vector search** (ChromaDB)  
- **LLM generation** (open-source models)  
- **Evaluation metrics**  
- **Limitations:**
  - limited context window  
  - small subset used due to compute limits  
  - retrieval quality depends heavily on chunking strategy  

---

# **Approach**

## **Methodology**
1. Extract article text.  
2. Normalize + chunk into manageable pieces.  
3. Embed each chunk.  
4. Store embeddings in vector DB.  
5. Query vector DB during generation.  
6. RAG model produces final summary grounded in retrieved chunks.

## **Models Used**
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** Open-source model via ctransformers (e.g., Mistral 7B or LLama-3)
- **Vector DB:** Chroma

## **Design Choices**
- Chunk size: **500 tokens**
- Overlap: **50 tokens**  
- Embeddings: 384-dimensional MiniLM  
- Top-k retrieval: 3–5  

## **Limitations**
- Summaries are short → requires careful evaluation  
- Chroma DB loads fully into RAM  
- Small model cannot match performance of GPT-4-class models  

---

# **Experiments**

## **Dataset**
Source:  
https://huggingface.co/datasets/ccdv/cnn_dailymail

**Subset used for reproducibility:**
- 200 training articles
- Avg article length: ~770 words
- Summary length: ~55 words

## **Implementation**
All implementation scripts are in `src/`:
- `data_processing.py` — load, clean, embed, store  
- `train.py` — retrieval + summarization  
- `evaluate.py` — ROUGE scoring  

## **Environment**
- Python 3.10  
- CPU or GPU (optional)  
- Libraries: transformers, chromadb, sentence-transformers, rouge-score  

---

# **Results**

## **Main Findings**
- Retrieval improves factual grounding and reduces hallucinations  
- ROUGE-L improvements are **modest** and below initial expectations  
- Qualitative gains are stronger than what ROUGE alone suggests

> **Takeaway:** RAG summaries include more article-specific details, even if ROUGE gains are limited.

### Example Observation
Baseline summaries sometimes omit core facts → RAG summaries recover those key details from retrieved chunks.

---

## **Supplementary Results**
- **Top-k > 5** introduces noise and drops performance  
- **Chunk size < 300 tokens** fragments context and hurts retrieval  
- MiniLM embeddings **outperform TF-IDF** for semantic search  
- ROUGE underestimates improvements in truthfulness and specificity  

---

# **Discussion**
Even with a small LLM and dataset subset, RAG consistently increases factual accuracy.  
However, performance strongly depends on:
- Chunking strategy  
- Retrieval precision  
- Model size and context window  

Future work:
- Stronger LLMs (Llama-3, Mixtral, Qwen)  
- **Hybrid BM25 + dense retrieval**  
- Cross-encoder reranking  
- Semantic-based chunk splitting  
- Scaling dataset size beyond 200 articles  

---

# **Conclusion**
This project successfully demonstrates an end-to-end RAG system for news summarization:

- 🧹 Preprocessing + chunking  
- 🔍 Embedding + vector search  
- 🧠 LLM summarization  
- 📊 ROUGE evaluation  

While ROUGE gains are small, RAG **clearly improves factual grounding**.  
The system is modular, reproducible, and easily extensible to other domains.

---

# **References**
- CNN/DailyMail Dataset — HuggingFace  
- ChromaDB Vector Store  
- SentenceTransformers  
- Mistral / LLaMA model documentation  
- ROUGE Metric Documentation
  
