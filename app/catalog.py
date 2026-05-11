"""
SHL Product Catalog Manager.

Loads catalog JSON and provides lookup, validation, 
and test_type code mapping utilities.
"""

import json
from pathlib import Path


# Mapping from catalog 'keys' field (full names) to single-letter test_type codes.
KEY_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Competencies": "C",
    "Biodata & Situational Judgment": "B",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

CODE_TO_KEY = {v: k for k, v in KEY_TO_CODE.items()}


class CatalogManager:
    """Manages the SHL product catalog for lookups and validation."""

    def __init__(self, catalog_path: str = None):
        if catalog_path is None:
            catalog_path = str(
                Path(__file__).parent.parent / "data" / "shl_product_catalog.json"
            )
        with open(catalog_path, "r") as f:
            self.items = json.loads(f.read(), strict=False)

        # Build lookup indexes
        self.url_set = {item["link"] for item in self.items}
        self.name_to_item = {item["name"]: item for item in self.items}
        self.url_to_item = {item["link"]: item for item in self.items}

    def get_test_type_code(self, keys: list[str]) -> str:
        """Convert catalog 'keys' list to comma-separated test_type codes.
        
        Example: ["Knowledge & Skills", "Simulations"] -> "K,S"
        
        Source: Sample conversation C4 shows test_type "A,S" for items with
        multiple assessment types.
        """
        codes = []
        for k in keys:
            if k in KEY_TO_CODE:
                codes.append(KEY_TO_CODE[k])
        # Remove duplicates, sort for consistency
        return ",".join(sorted(set(codes))) if codes else "K"

    def validate_url(self, url: str) -> bool:
        """Check if a URL exists in the catalog. Every URL it returns must come from the catalog."""
        return url in self.url_set

    def find_by_name(self, name: str) -> dict | None:
        """Find a catalog item by exact name match."""
        return self.name_to_item.get(name)

    def find_by_url(self, url: str) -> dict | None:
        """Find a catalog item by URL."""
        return self.url_to_item.get(url)

    def search_by_name_fuzzy(self, query: str) -> list[dict]:
        """Simple fuzzy search by name substring (case-insensitive).
        Used as fallback when vector search doesn't find specific items.
        """
        query_lower = query.lower()
        return [
            item for item in self.items
            if query_lower in item["name"].lower()
        ]

    def to_recommendation(self, item: dict) -> dict:
        """Convert a catalog item to the API recommendation format.
        
        Source: Assignment PDF response schema:
        {"name": "...", "url": "https://www.shl.com/...", "test_type": "K"}
        """
        return {
            "name": item["name"],
            "url": item["link"],
            "test_type": self.get_test_type_code(item.get("keys", [])),
        }

    def format_for_llm(self, items: list[dict], numbered: bool = True) -> str:
        """Format catalog items as a numbered list for LLM consumption."""
        lines = []
        for i, item in enumerate(items):
            idx = f"[{i+1}] " if numbered else ""
            keys_str = ", ".join(item.get("keys", []))
            langs = item.get("languages", [])
            lang_str = ", ".join(langs[:3])
            if len(langs) > 3:
                lang_str += f" (+{len(langs)-3} more)"
            lines.append(
                f"{idx}{item['name']} | "
                f"Type: {self.get_test_type_code(item.get('keys', []))} ({keys_str}) | "
                f"Duration: {item.get('duration', 'N/A')} | "
                f"Levels: {', '.join(item.get('job_levels', []))} | "
                f"Languages: {lang_str} | "
                f"Description: {item.get('description', 'N/A')[:150]}"
            )
        return "\n".join(lines)

    def get_all(self) -> list[dict]:
        """Return all catalog items."""
        return self.items

    def __len__(self) -> int:
        return len(self.items)
