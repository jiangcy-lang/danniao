"""向量轨接口（总规范 §2 / Step 4 预留）。

向量即节点重构后，VectorStore 是向量视图的抽象接口。
实现见 ``chroma_store.py``（ChromaDB）和 ``NullVectorStore``（空占位）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class VectorStore(ABC):
    """向量数据库抽象：余弦相似度检索节点。"""

    @abstractmethod
    def upsert(
        self,
        node_id: str,
        vector: Sequence[float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """插入或更新向量节点。

        Args:
            node_id: 节点唯一标识
            vector: 嵌入向量
            metadata: 可选元数据（label / kind / modality 等）
        """
        ...

    @abstractmethod
    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> list[tuple[str, float]]:
        """按向量相似度检索。

        Args:
            vector: 查询向量
            top_k: 返回条数上限
            threshold: 相似度阈值（0~1，仅返回 >= threshold 的结果）

        Returns:
            ``[(node_id, similarity), ...]`` 按相似度降序
        """
        ...

    def get(self, node_id: str) -> dict[str, Any] | None:
        """按 ID 获取节点（含向量、元数据）。默认实现返回 None。"""
        return None

    def count(self) -> int:
        """返回存储中的向量总数。默认实现返回 0。"""
        return 0



