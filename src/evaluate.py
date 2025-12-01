# src/evaluate.py
from datasets import load_metric
from main_agent import Agent
import argparse
import json
from pathlib import Path

def evaluate_sample(agent: Agent, test_pairs, top_k=3):
    """
    Evaluate summaries using iterative chunk summarization.
    """
    rouge = load_metric("rouge")
    hyps = []
    refs = []

    for item in test_pairs:
        # Retrieve top_k chunks
        retrieved = agent.retrieve_chunks(item["article"], top_k=top_k)
        chunks = [r["text"] for r in retrieved]

        # Summarize each chunk briefly
        brief_summaries = [agent.summarize_brief(c, max_length=80) for c in chunks]

        # Combine brief summaries and generate a detailed summary
        final_summary = agent.summarize_detailed(brief_summaries, max_length=200)

        hyps.append(final_summary)
        refs.append(item["summary"])

    # Compute ROUGE
    results = rouge.compute(predictions=hyps, references=refs)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of test articles")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    # Load sample from meta.json
    meta_path = Path("data/meta.json")
    if not meta_path.exists():
        raise FileNotFoundError("meta.json not found. Run data processing first.")

    meta = json.load(open(meta_path))
    test_pairs = []
    for i, m in enumerate(meta["metas"][:args.n]):
        test_pairs.append({
            "article": meta["texts"][i],
            "summary": m.get("summary", "")
        })

    # Initialize agent
    agent = Agent()

    print(f"Evaluating on {len(test_pairs)} articles using top_k={args.top_k} chunks...")
    rouge_scores = evaluate_sample(agent, test_pairs, top_k=args.top_k)
    print("ROUGE Results:", rouge_scores)
