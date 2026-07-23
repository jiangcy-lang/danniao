"""嵌入管道：文本 → 向量（丹鸟的感官器官）。

使用 sentence-transformers 的多语言模型，将文本映射到归一化的语义向量。
懒加载：首次调用 embed_text 时才加载模型。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

import numpy as np


class EmbeddingPipeline:
    """文本嵌入管道（sentence-transformers 包装）。

    默认模型 ``paraphrase-multilingual-MiniLM-L12-v2``：
    - 384 维输出
    - 支持中英等 50+ 语言
    - 模型体积约 120MB
    """

    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._device = device
        self._model: SentenceTransformer | None = None  # 懒加载

    def _ensure_model(self) -> "SentenceTransformer":
        """懒加载模型实例。"""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers 未安装。请运行: pip install sentence-transformers"
            ) from exc

        kwargs: dict = {}
        if self._cache_dir:
            kwargs["cache_folder"] = self._cache_dir
        if self._device:
            kwargs["device"] = self._device

        self._model = SentenceTransformer(self._model_name, **kwargs)
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """将文本编码为归一化向量（L2 norm = 1）。

        Args:
            text: 输入文本（中英文均可）

        Returns:
            归一化的 numpy 向量，shape=(dim,)，dtype=float32
        """
        model = self._ensure_model()
        vec = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """批量编码文本为归一化向量。

        Args:
            texts: 输入文本列表

        Returns:
            归一化的 numpy 矩阵，shape=(len(texts), dim)
        """
        model = self._ensure_model()
        vecs = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """返回嵌入维度（需要加载模型）。"""
        model = self._ensure_model()
        # 兼容新旧版本方法名
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()
