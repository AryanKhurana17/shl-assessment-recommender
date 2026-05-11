"""
Tests for app.retrieval — FAISS semantic retrieval.

Run: .venv/bin/python -m tests.test_retrieval
"""

from app.catalog import CatalogManager
from app.retrieval import AssessmentRetriever


def test_initialize_retriever():
    catalog = CatalogManager()
    retriever = AssessmentRetriever(catalog)
    print(f"[PASS] Retriever initialized with {len(catalog)} items")
    return retriever


def test_search_java(retriever=None):
    if retriever is None:
        catalog = CatalogManager()
        retriever = AssessmentRetriever(catalog)
    results = retriever.search("Java developer mid-level", k=5)
    print(f"\n[PASS] Search 'Java developer mid-level': {len(results)} results")
    for r in results:
        print(f"   - {r['name']} | {', '.join(r.get('keys', []))}")
    assert len(results) > 0, "Should find Java assessments"


def test_search_personality(retriever=None):
    if retriever is None:
        catalog = CatalogManager()
        retriever = AssessmentRetriever(catalog)
    results = retriever.search("personality assessment for senior leadership", k=5)
    print(f"\n[PASS] Search 'personality senior leadership': {len(results)} results")
    for r in results:
        print(f"   - {r['name']} | {', '.join(r.get('keys', []))}")
    assert len(results) > 0


def test_search_contact_centre(retriever=None):
    if retriever is None:
        catalog = CatalogManager()
        retriever = AssessmentRetriever(catalog)
    results = retriever.search("entry-level contact centre agents English", k=5)
    print(f"\n[PASS] Search 'contact centre agents': {len(results)} results")
    for r in results:
        print(f"   - {r['name']} | {', '.join(r.get('keys', []))}")
    assert len(results) > 0


def test_search_with_scores(retriever=None):
    if retriever is None:
        catalog = CatalogManager()
        retriever = AssessmentRetriever(catalog)
    results = retriever.search_with_scores("Excel Word admin assistant", k=5)
    print(f"\n[PASS] Search with scores 'Excel Word admin':")
    for item, score in results:
        print(f"   - {item['name']} (score: {score:.4f})")
    assert len(results) > 0


def test_search_safety(retriever=None):
    if retriever is None:
        catalog = CatalogManager()
        retriever = AssessmentRetriever(catalog)
    results = retriever.search("safety dependability plant operator chemical", k=5)
    print(f"\n[PASS] Search 'safety plant operator': {len(results)} results")
    for r in results:
        print(f"   - {r['name']} | {', '.join(r.get('keys', []))}")
    assert len(results) > 0


if __name__ == "__main__":
    retriever = test_initialize_retriever()
    test_search_java(retriever)
    test_search_personality(retriever)
    test_search_contact_centre(retriever)
    test_search_with_scores(retriever)
    test_search_safety(retriever)
    print("\nAll retrieval tests complete.")
