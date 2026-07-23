"""Step 4B 验收测试：扩散激活。

测试两种传播路径：
1. 沿图边传播（强）—— 不需要外部依赖
2. 沿向量邻近传播（弱）—— 需要真实向量存储
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from danniao.hippocampus import (
    ChromaVectorStore,
    EmbeddingPipeline,
    VectorCognitiveSpace,
    VectorStore,
)
from danniao.hippocampus.spreading import SpreadingActivation, SpreadConfig


# ---------- 测试工具 ----------


class _MockEmbedding:
    """确定性 mock 嵌入管道，无需 sentence-transformers。

    使用字符码 + 位置混合散列确保不同文本产生不同归一化向量。
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for i, ch in enumerate(text):
            idx = (ord(ch) + i * 31) % self._dim
            vec[idx] += float(ord(ch)) + 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_text(t) for t in texts])

    @property
    def dimension(self) -> int:
        return self._dim


class _StubStore(VectorStore):
    """最小 VectorStore 桩。"""

    def upsert(self, node_id, vector, *, metadata=None):
        pass

    def search(self, vector, *, top_k=5, threshold=0.7):
        return []

    def get(self, node_id):
        return None

    def count(self) -> int:
        return 0


def _make_mock_space() -> VectorCognitiveSpace:
    """创建带 mock 嵌入的认知空间（无外部依赖）。"""
    store = _StubStore()
    emb = _MockEmbedding()
    return VectorCognitiveSpace(vector_store=store, embedding=emb)


def _has_deps() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import chromadb  # noqa: F401

        return True
    except ImportError:
        return False


_skip_no_deps = pytest.mark.skipif(
    not _has_deps(), reason="sentence-transformers 或 chromadb 未安装"
)


def _cleanup(*paths: str):
    for p in paths:
        shutil.rmtree(p, ignore_errors=True)


# ==================== 图边传播（无外部依赖） ====================


def test_spread_empty_seeds():
    """空种子列表返回空结果。"""
    space = _make_mock_space()
    spreader = SpreadingActivation(space)
    assert spreader.spread([]) == {}


def test_spread_seed_not_in_graph():
    """种子节点不在图中时返回空结果。"""
    space = _make_mock_space()
    spreader = SpreadingActivation(space)
    assert spreader.spread(["nonexistent"]) == {}


def test_spread_single_seed_no_neighbors():
    """孤立种子节点：只有种子被激活。"""
    space = _make_mock_space()
    vec = space.embedding.embed_text("苹果")
    node_id = space.add_node(vec, label="苹果", kind="trunk")

    spreader = SpreadingActivation(space)
    activation = spreader.spread([node_id])

    assert node_id in activation
    assert activation[node_id] == pytest.approx(1.0)
    assert len(activation) == 1


def test_spread_along_graph_edges():
    """沿图边传播：种子 → 直接邻居被激活。"""
    space = _make_mock_space()

    # 创建 苹果 → 颜色-红 → 深红色
    vec1 = space.embedding.embed_text("苹果")
    vec2 = space.embedding.embed_text("颜色-红")
    vec3 = space.embedding.embed_text("深红色")

    n1 = space.add_node(vec1, label="苹果", kind="trunk")
    n2 = space.add_node(vec2, label="颜色-红", kind="feature")
    n3 = space.add_node(vec3, label="深红色", kind="feature")

    space.add_edge(n1, n2, weight=0.8)
    space.add_edge(n2, n3, weight=0.6)

    spreader = SpreadingActivation(space)
    activation = spreader.spread([n1])

    # 种子被激活
    assert n1 in activation
    assert activation[n1] == pytest.approx(1.0)

    # 直接邻居被激活
    assert n2 in activation
    assert activation[n2] > 0.0

    # 二跳邻居被激活，但激活值低于直接邻居
    assert n3 in activation
    assert activation[n3] < activation[n2]


def test_spread_activation_decreases_with_distance():
    """激活值随传播距离递减。"""
    space = _make_mock_space()

    # 创建链式：A → B → C → D
    nodes = []
    for i, label in enumerate(["A", "B", "C", "D"]):
        vec = space.embedding.embed_text(label)
        nid = space.add_node(vec, label=label, kind="trunk")
        nodes.append(nid)

    for i in range(len(nodes) - 1):
        space.add_edge(nodes[i], nodes[i + 1], weight=1.0)

    spreader = SpreadingActivation(space)
    activation = spreader.spread([nodes[0]])

    # 激活值递减
    values = [activation.get(nid, 0.0) for nid in nodes]
    for i in range(len(values) - 1):
        assert values[i] > values[i + 1], f"节点 {i} 的激活应高于节点 {i + 1}"


def test_spread_respects_edge_weight():
    """边权重高的邻居获得更多激活。"""
    space = _make_mock_space()

    vec_seed = space.embedding.embed_text("种子")
    vec_weak = space.embedding.embed_text("弱关联")
    vec_strong = space.embedding.embed_text("强关联")

    seed = space.add_node(vec_seed, label="种子", kind="trunk")
    weak_n = space.add_node(vec_weak, label="弱关联", kind="feature")
    strong_n = space.add_node(vec_strong, label="强关联", kind="feature")

    space.add_edge(seed, weak_n, weight=0.2)
    space.add_edge(seed, strong_n, weight=0.9)

    spreader = SpreadingActivation(space)
    activation = spreader.spread([seed])

    assert activation[strong_n] > activation[weak_n]


def test_spread_max_hops_limit():
    """max_hops 限制传播深度。"""
    space = _make_mock_space()

    # 创建 5 节点链
    nodes = []
    for i in range(5):
        vec = space.embedding.embed_text(f"node_{i}")
        nid = space.add_node(vec, label=f"node_{i}", kind="trunk")
        nodes.append(nid)

    for i in range(4):
        space.add_edge(nodes[i], nodes[i + 1], weight=1.0)

    # max_hops=1：只传播到直接邻居
    config = SpreadConfig(max_hops=1, activation_threshold=0.001)
    spreader = SpreadingActivation(space, config=config)
    activation = spreader.spread([nodes[0]])

    assert nodes[0] in activation  # 种子
    assert nodes[1] in activation  # 1 跳
    assert nodes[2] not in activation  # 2 跳不应到达


def test_spread_threshold_stops_propagation():
    """低于阈值的激活不再传播。"""
    space = _make_mock_space()

    vec1 = space.embedding.embed_text("A")
    vec2 = space.embedding.embed_text("B")

    n1 = space.add_node(vec1, label="A", kind="trunk")
    n2 = space.add_node(vec2, label="B", kind="feature")
    space.add_edge(n1, n2, weight=0.01)  # 极低权重

    # 高阈值：弱边不应传播
    config = SpreadConfig(activation_threshold=0.5)
    spreader = SpreadingActivation(space, config=config)
    activation = spreader.spread([n1])

    assert n1 in activation
    # 边权重 0.01，传播量 = 1.0 * 0.5 * 0.7 * 0.01 = 0.0035 < 0.5
    assert n2 not in activation or activation[n2] < 0.5


def test_get_activated_nodes_sorted():
    """get_activated_nodes 返回按激活水平降序的列表。"""
    space = _make_mock_space()

    vec_seed = space.embedding.embed_text("种子")
    vec_a = space.embedding.embed_text("A")
    vec_b = space.embedding.embed_text("B")

    seed = space.add_node(vec_seed, label="种子", kind="trunk")
    na = space.add_node(vec_a, label="A", kind="feature")
    nb = space.add_node(vec_b, label="B", kind="feature")

    space.add_edge(seed, na, weight=0.3)
    space.add_edge(seed, nb, weight=0.9)

    spreader = SpreadingActivation(space)
    activated = spreader.get_activated_nodes([seed], threshold=0.01)

    assert len(activated) >= 2
    # 种子应在最前
    assert activated[0][0] == seed
    # 按激活水平降序
    for i in range(len(activated) - 1):
        assert activated[i][1] >= activated[i + 1][1]


# ==================== 多种子传播 ====================


def test_spread_multiple_seeds():
    """多个种子节点同时传播。"""
    space = _make_mock_space()

    vec_a = space.embedding.embed_text("A")
    vec_b = space.embedding.embed_text("B")
    vec_c = space.embedding.embed_text("C")

    na = space.add_node(vec_a, label="A", kind="trunk")
    nb = space.add_node(vec_b, label="B", kind="trunk")
    nc = space.add_node(vec_c, label="C", kind="feature")

    # A → C, B → C（C 被两条路径激活）
    space.add_edge(na, nc, weight=0.5)
    space.add_edge(nb, nc, weight=0.5)

    spreader = SpreadingActivation(space)
    activation = spreader.spread([na, nb])

    assert activation[na] == pytest.approx(1.0)
    assert activation[nb] == pytest.approx(1.0)
    assert nc in activation
    # C 从两个种子获得激活，应高于单种子
    single_activation = spreader.spread([na])
    assert activation[nc] > single_activation.get(nc, 0.0)


# ==================== 真实依赖测试 ====================


@_skip_no_deps
def test_spread_with_real_vectors():
    """真实向量存储：扩散激活沿向量邻近传播。"""
    try:
        store = ChromaVectorStore(path=".test_chroma_spread")
        emb = EmbeddingPipeline()
        space = VectorCognitiveSpace(vector_store=store, embedding=emb)

        # 创建几个语义相关的节点
        texts = ["苹果", "水果", "红色", "甜味", "计算机"]
        node_ids = []
        for t in texts:
            vec = emb.embed_text(t)
            nid = space.add_node(vec, label=t, kind="trunk")
            node_ids.append(nid)

        # 建图边：苹果 → 红色
        space.add_edge(node_ids[0], node_ids[2], weight=0.8)

        spreader = SpreadingActivation(space)
        activation = spreader.spread([node_ids[0]])

        # 种子被激活
        assert node_ids[0] in activation
        assert activation[node_ids[0]] == pytest.approx(1.0)

        # 图边邻居被激活
        assert node_ids[2] in activation  # 红色

        # 向量邻近节点也可能被激活（水果 与 苹果 语义相近）
        assert node_ids[1] in activation  # 水果

        # 不相关节点激活值低
        computer_activation = activation.get(node_ids[4], 0.0)
        apple_activation = activation.get(node_ids[0], 0.0)
        assert computer_activation < apple_activation
    finally:
        _cleanup(".test_chroma_spread")
