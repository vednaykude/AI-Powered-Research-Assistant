# download_arxiv.py
import arxiv
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "papers_example.csv"

def download_arxiv(query="LLM safety", max_results=10, download_pdf=False):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    records = []
    for result in search.results():
        try:
            pdf_path = result.download_pdf(dirpath=DATA_DIR) if download_pdf else None
        except Exception:
            pdf_path = None
        authors = [a.name for a in result.authors]
        records.append({
            "paper_id": result.entry_id,
            "title": result.title.strip(),
            "authors": authors,
            "abstract": result.summary.strip(),
            "url": result.pdf_url,
            "pdf_path": str(pdf_path) if pdf_path else None
        })
    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False)
    print(f"Downloaded {len(df)} papers → {CSV_PATH}")
    return df

if __name__ == "__main__":
    download_arxiv("LLM safety", max_results=10, download_pdf=False)
