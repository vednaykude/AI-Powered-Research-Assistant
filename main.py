# test_agent.py
from src.main_agent import Agent

# Initialize agent
agent = Agent()

# Queries to test
queries = [
    "Who designed the Burj Khalifa?",
    "Explain the architecture style of CCTV headquarters in Beijing",
    "What is the impact of climate change on urban planning?"
]

# Number of top chunks to retrieve
top_k = 3
iterations = 2  # for iterative RAG

for query in queries:
    print("="*80)
    print(f"Query: {query}\n")
    
    # Retrieve top chunks
    hits = agent.retrieve_chunks(query, top_k=top_k)
    
    for i, h in enumerate(hits, start=1):
        print(f"Top chunk {i}:")
        print(f"  Title: {h['meta']['title']}")
        print(f"  Similarity: {agent.get_similarity_percentage(h['score']):.2f}%")
        print(f"  Chunk preview: {h['text'][:200]}...\n")
    
    # Iterative RAG answer
    rag_answer = agent.iterative_rag(query, iterations=iterations, top_k=top_k)
    print(f"RAG Answer:\n{rag_answer}\n")
    
    # QA over top chunk
    top_context = hits[0]['text']
    qa_answer = agent.answer_question(query, top_context)
    print(f"QA Answer (top chunk): {qa_answer}\n")
