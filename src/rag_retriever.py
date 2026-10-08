"""
TruthLens - RAG Semantic Retriever
Uses SentenceTransformer ('all-MiniLM-L6-v2') and FAISS vector database to index and retrieve verified evidence.
"""

import os
import sys
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer
from src.knowledge_base import initialize_knowledge_base, DEFAULT_KB_PATH

FAISS_INDEX_PATH = "knowledge_base/faiss_index.bin"
FAISS_METADATA_PATH = "knowledge_base/faiss_metadata.json"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class RAGRetriever:
    """
    Retrieves evidence documents using dense vector representations and FAISS similarity search.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        index_path: str = FAISS_INDEX_PATH,
        metadata_path: str = FAISS_METADATA_PATH,
        kb_path: str = DEFAULT_KB_PATH
    ):
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.kb_path = kb_path

        # Load embedding model
        self.embedder = SentenceTransformer(self.model_name)
        if hasattr(self.embedder, "get_embedding_dimension"):
            self.embedding_dim = self.embedder.get_embedding_dimension()
        else:
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        self.index = None
        self.documents = []

        # Load or build index
        self._load_or_build_index()

    def _prepare_document_text(self, doc: Dict[str, Any]) -> str:
        """Constructs an information-dense text string for embedding representation."""
        title = doc.get("title", "")
        claim = doc.get("claim_reviewed", "")
        summary = doc.get("fact_check_summary", "")
        verdict = doc.get("verdict", "")
        return f"{title} | Claim: {claim} | Verdict: {verdict} | Evidence: {summary}"

    def build_index(self, force_rebuild: bool = False):
        """Builds and serializes FAISS index from knowledge base records."""
        raw_docs = initialize_knowledge_base(self.kb_path)
        self.documents = raw_docs

        texts_to_embed = [self._prepare_document_text(doc) for doc in self.documents]

        print(f"Embedding {len(texts_to_embed)} knowledge base documents using {self.model_name}...")
        embeddings = self.embedder.encode(
            texts_to_embed,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        ).astype(np.float32)

        # Flat Inner Product index with normalized vectors computes Cosine Similarity
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)
        self.index = index

        # Persist index and metadata
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

        print(f"FAISS index built and saved to: {self.index_path}")

    def _load_or_build_index(self):
        """Loads index from disk or builds from scratch if absent."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                return
            except Exception as e:
                print(f"Failed to load cached FAISS index: {e}. Rebuilding...")

        self.build_index()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.25,
        entity_boost: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic search against fact-checking knowledge base.
        Returns top-k documents matching above similarity threshold.
        """
        if not query or not query.strip() or self.index is None or len(self.documents) == 0:
            return []

        search_query = query.strip()
        if entity_boost:
            # Append salient entities to enhance retrieval relevance
            search_query += " " + " ".join(entity_boost[:3])

        query_vector = self.embedder.encode(
            [search_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_vector, k)

        raw_results = []
        for score, idx in zip(scores[0], indices[0]):
            sim_score = float(score)
            if idx < 0 or idx >= len(self.documents):
                continue
            if sim_score < threshold:
                continue

            doc = dict(self.documents[idx])
            doc["similarity_score"] = round(sim_score, 4)
            doc["relevance_percent"] = round(sim_score * 100, 1)
            raw_results.append(doc)

        # Dynamic Relevance Filtering:
        # If the top match is high (>50%), only include secondary matches if they are
        # genuinely relevant (>= 45%) and not distant noise.
        filtered_results = []
        if raw_results:
            top_score = raw_results[0]["similarity_score"]
            for i, doc in enumerate(raw_results):
                score = doc["similarity_score"]
                if i == 0:
                    filtered_results.append(doc)
                else:
                    # Only keep secondary matches if they meet a meaningful relevance threshold
                    # (at least 45% absolute similarity, and within 25% of the top match)
                    if score >= 0.45 and (top_score - score) <= 0.25:
                        filtered_results.append(doc)

        return filtered_results


if __name__ == "__main__":
    retriever = RAGRetriever()
    test_query = "Did Jennifer Aniston and Brad Pitt get married secretly in Paris?"
    print(f"Testing RAG Retrieval for Query: '{test_query}'")
    matches = retriever.retrieve(test_query, top_k=2)
    for i, m in enumerate(matches, 1):
        print(f"\nMatch #{i}: (Relevance: {m['relevance_percent']}%)")
        print(f"  Title:   {m['title']}")
        print(f"  Verdict: {m['verdict']}")
        print(f"  Source:  {m['source']}")
        print(f"  Snippet: {m['fact_check_summary'][:120]}...")
