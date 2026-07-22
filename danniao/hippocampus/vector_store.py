"""ChromaDB 向量存储：节点 embedding 物理底座。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from danniao.hippocampus.embeddings import cosine_similarity


class HippocampusVectorStore:
    COLLECTION = "hippocampus_nodes"

    def __init__(self, persist_dir: str | Path = ".chroma_hippocampus") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_node(
        self,
        node_id: str,
        embedding: list[float],
        *,
        concept: str,
        kind: str,
        dimension: str | None = None,
        value: str | None = None,
        parent_trunk: str | None = None,
    ) -> None:
        metadata: dict[str, Any] = {
            "concept": concept,
            "kind": kind,
            "dimension": dimension or "",
            "value": value or "",
            "parent_trunk": parent_trunk or "",
        }
        self._collection.upsert(
            ids=[node_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[concept],
        )

    def get_embedding(self, node_id: str) -> list[float] | None:
        got = self._collection.get(ids=[node_id], include=["embeddings"])
        if not got["ids"]:
            return None
        emb = got["embeddings"][0]
        return emb if emb is not None else None

    def query_trunks(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """返回 [(node_id, similarity, metadata), ...] 仅 trunk。"""
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["metadatas", "distances"],
        )
        hits: list[tuple[str, float, dict[str, Any]]] = []
        for node_id, dist, meta in zip(
            result["ids"][0],
            result["distances"][0],
            result["metadatas"][0],
        ):
            if meta.get("kind") != "trunk":
                continue
            # Chroma cosine distance: 0 = identical; similarity ≈ 1 - distance
            sim = 1.0 - float(dist)
            hits.append((node_id, sim, meta))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits

    def best_trunk_match(
        self,
        query_embedding: list[float],
        *,
        threshold: float = 0.85,
    ) -> tuple[str | None, float]:
        hits = self.query_trunks(query_embedding, top_k=1)
        if not hits:
            return None, 0.0
        node_id, sim, _ = hits[0]
        if sim >= threshold:
            return node_id, sim
        return None, sim

    def count(self) -> int:
        return self._collection.count()
