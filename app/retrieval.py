"""
FAISS-based semantic retrieval

Design choice: FAISS + sentence-transformers (all-MiniLM-L6-v2)
- Free, local, no API key needed for embeddings
- ~6ms per query at runtime (vs ~200ms for API-based embeddings)
- 384-dim vectors, sufficient for 377 short-text catalog items

Embedding model: all-MiniLM-L6-v2 (sentence-transformers)
Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
"""

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.catalog import CatalogManager


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INDEX_PATH = str(Path(__file__).parent.parent / "data" / "faiss_index")


class AssessmentRetriever:
    """Semantic search over the SHL assessment catalog using FAISS."""

    def __init__(self, catalog: CatalogManager, index_path: str = None):
        self.catalog = catalog
        self.index_path = index_path or DEFAULT_INDEX_PATH

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

        # Load pre-built index or build new one
        if Path(self.index_path).exists():
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            self._build_index()

    def _make_document_text(self, item: dict) -> str:
        """Create a rich text representation of an assessment for embedding.
        
        Design choice: Include all searchable fields (name, description, 
        types, levels, languages) so semantic search can match on any dimension.
        For example, "Java developer" should match items with "Java" in the name,
        and "entry-level" should match items with that job level.
        """
        parts = [
            f"Assessment Name: {item['name']}",
            f"Description: {item.get('description', '')}",
            f"Assessment Types: {', '.join(item.get('keys', []))}",
            f"Job Levels: {', '.join(item.get('job_levels', []))}",
            f"Languages: {', '.join(item.get('languages', []))}",
            f"Duration: {item.get('duration', 'Not specified')}",
        ]
        return "\n".join(parts)

    def _build_index(self):
        """Build FAISS index from all catalog items.
        
        Design choice: Embed ALL 377 items upfront. The index is <2MB 
        and loads instantly. This avoids runtime embedding of catalog items.
        """
        documents = []
        for item in self.catalog.get_all():
            doc = Document(
                page_content=self._make_document_text(item),
                metadata=item,
            )
            documents.append(doc)

        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

    def save_index(self, path: str = None):
        """Save FAISS index to disk for fast startup."""
        save_path = path or self.index_path
        self.vectorstore.save_local(save_path)

    def search(self, query: str, k: int = 20) -> list[dict]:
        """Retrieve top-k assessments semantically similar to the query.
        
        Design choice: Retrieve k=20 (more than the max 10 recommendations)
        to give the LLM a larger pool to select from. This improves recall
        because the LLM can reason about relevance better than pure embedding
        similarity.
        """
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.metadata for doc in results]

    def search_with_scores(self, query: str, k: int = 20) -> list[tuple[dict, float]]:
        """Retrieve with similarity scores for debugging/tuning."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(doc.metadata, score) for doc, score in results]
