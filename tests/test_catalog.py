"""
Tests for app.catalog — SHL product catalog manager.

Run: .venv/bin/python -m tests.test_catalog
"""

from app.catalog import CatalogManager


def test_load_catalog():
    catalog = CatalogManager()
    print(f"[PASS] Loaded {len(catalog)} items")
    assert len(catalog) > 0, "Catalog should not be empty"


def test_type_code_mapping():
    catalog = CatalogManager()
    code = catalog.get_test_type_code(["Knowledge & Skills", "Simulations"])
    print(f"[PASS] K&S + Sim -> '{code}' (expected: 'K,S')")
    assert code == "K,S", f"Expected 'K,S', got '{code}'"

    code2 = catalog.get_test_type_code(["Personality & Behavior"])
    print(f"[PASS] P&B -> '{code2}' (expected: 'P')")
    assert code2 == "P", f"Expected 'P', got '{code2}'"


def test_url_validation():
    catalog = CatalogManager()
    valid = catalog.validate_url(
        "https://www.shl.com/products/product-catalog/view/net-framework-4-5/"
    )
    print(f"[PASS] Valid URL: {valid} (expected: True)")
    assert valid is True

    invalid = catalog.validate_url("https://www.shl.com/fake-url/")
    print(f"[PASS] Invalid URL: {invalid} (expected: False)")
    assert invalid is False


def test_find_by_name():
    catalog = CatalogManager()
    item = catalog.find_by_name(".NET Framework 4.5")
    print(f"[PASS] Found: {item['name'] if item else 'NOT FOUND'}")
    assert item is not None, "Should find .NET Framework 4.5"


def test_fuzzy_search():
    catalog = CatalogManager()
    results = catalog.search_by_name_fuzzy("java")
    print(f"[PASS] Fuzzy 'java': {len(results)} results")
    for r in results:
        print(f"   - {r['name']}")
    assert len(results) > 0, "Should find Java assessments"


def test_to_recommendation():
    catalog = CatalogManager()
    item = catalog.find_by_name(".NET Framework 4.5")
    rec = catalog.to_recommendation(item)
    print(f"[PASS] Recommendation: {rec}")
    assert "name" in rec
    assert "url" in rec
    assert "test_type" in rec


def test_format_for_llm():
    catalog = CatalogManager()
    items = catalog.search_by_name_fuzzy("excel")
    formatted = catalog.format_for_llm(items[:3])
    print(f"[PASS] LLM format:\n{formatted}")
    assert "[1]" in formatted, "Should have numbered items"


if __name__ == "__main__":
    test_load_catalog()
    test_type_code_mapping()
    test_url_validation()
    test_find_by_name()
    test_fuzzy_search()
    test_to_recommendation()
    test_format_for_llm()
    print("\nAll catalog tests complete.")
