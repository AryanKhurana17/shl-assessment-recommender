"""
Build the FAISS index from the SHL catalog.
Run this once before starting the server:
    python -m scripts.build_index
"""

from dotenv import load_dotenv
load_dotenv()

from app.catalog import CatalogManager
from app.retrieval import AssessmentRetriever


def main():
    print("Loading catalog...")
    catalog = CatalogManager()
    print(f"Loaded {len(catalog)} items")

    print("Building FAISS index (this may take a minute on first run)...")
    retriever = AssessmentRetriever(catalog)

    print("Saving index to data/faiss_index/...")
    retriever.save_index()

    print("Done! Index saved.")

    # Quick sanity check
    print("\nSanity check — searching for 'Java developer':")
    results = retriever.search("Java developer mid-level", k=5)
    for r in results:
        print(f"  - {r['name']} | {', '.join(r.get('keys', []))}")


if __name__ == "__main__":
    main()
