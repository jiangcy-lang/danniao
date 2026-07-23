"""Step 4A 验收测试：向量节点 + 嵌入管道 + 原始日志 + 门控 + 动力学。

所有测试基于向量即节点架构，无纯字符串回退，无伪实现。
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from danniao.hippocampus import (
    ChromaVectorStore,
    DynamicCognitiveTree,
    EmbeddingPipeline,
    EpisodicLog,
    InformationTriggerGate,
    NeuroDynamicsEngine,
    VectorCognitiveSpace,
    VectorStore,
    hash_vector,
)


# ---------- 测试工具 ----------


def _make_tree(chroma_path: str):
    """创建带向量空间的认知树。"""
    store = ChromaVectorStore(path=chroma_path)
    emb = EmbeddingPipeline()
    space = VectorCognitiveSpace(vector_store=store, embedding=emb)
    tree = DynamicCognitiveTree(space)
    return tree, space


def _cleanup(*paths: str):
    for p in paths:
        shutil.rmtree(p, ignore_errors=True)


def _has_deps():
    try:
        import sentence_transformers  # noqa: F401
        import chromadb  # noqa: F401

        return True
    except ImportError:
        return False


_skip_no_deps = pytest.mark.skipif(not _has_deps(), reason="sentence-transformers 或 chromadb 未安装")


class _StubStore(VectorStore):
    """最小 VectorStore 桩，用于不需要真实向量数据库的测试。"""

    def upsert(self, node_id, vector, *, metadata=None):
        pass

    def search(self, vector, *, top_k=5, threshold=0.7):
        return []

    def get(self, node_id):
        return None

    def count(self) -> int:
        return 0


# ==================== 向量哈希（无外部依赖） ====================


def test_vector_hash_stability():
    """同一向量多次哈希结果相同，跨 dtype 一致。"""
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert hash_vector(vec) == hash_vector(vec.copy())
    assert hash_vector(vec) == hash_vector(vec.astype(np.float64))
    assert hash_vector(vec).startswith("v_")


def test_vector_hash_different_vectors():
    """不同向量应生成不同 ID。"""
    v1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    v2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    assert hash_vector(v1) != hash_vector(v2)


# ==================== 原始日志（SQLite，无外部依赖） ====================


def test_episodic_log_append_and_replay():
    """日志可追加且可回放，幂等不重复。"""
    log = EpisodicLog(path=":memory:")
    log.append("int_001", "苹果", source="user")
    log.append("int_001", "苹果", source="user")  # 幂等
    assert log.count() == 1

    log.append("int_002", "青苹果", source="user")
    assert log.count() == 2

    entries = log.replay()
    assert len(entries) == 2
    assert entries[0]["text"] == "苹果"
    assert entries[1]["text"] == "青苹果"
    log.close()


def test_episodic_log_metadata():
    """日志支持元数据。"""
    log = EpisodicLog(path=":memory:")
    log.append("int_003", "测试", source="search", metadata={"url": "http://example.com"})
    entries = log.replay()
    assert entries[0]["metadata"]["url"] == "http://example.com"
    log.close()


# ==================== 强制约束 ====================


def test_tree_rejects_none_space():
    """DynamicCognitiveTree 不接受 None space。"""
    with pytest.raises(ValueError):
        DynamicCognitiveTree(space=None)  # type: ignore


def test_space_rejects_none_store():
    """VectorCognitiveSpace 不接受 None vector_store。"""
    emb = EmbeddingPipeline()
    with pytest.raises(ValueError):
        VectorCognitiveSpace(vector_store=None, embedding=emb)  # type: ignore


def test_space_rejects_none_embedding():
    """VectorCognitiveSpace 不接受 None embedding。"""
    store = _StubStore()
    with pytest.raises(ValueError):
        VectorCognitiveSpace(vector_store=store, embedding=None)  # type: ignore


# ==================== 需要依赖的测试 ====================


@_skip_no_deps
def test_embedding_cross_language():
    """「苹果」和「apple」在向量空间中相近。"""
    emb = EmbeddingPipeline()
    vec_zh = emb.embed_text("苹果")
    vec_en = emb.embed_text("apple")

    similarity = float(np.dot(vec_zh, vec_en))
    assert similarity > 0.5, f"中英同义词相似度应 > 0.5，实际: {similarity}"
    assert emb.dimension == 384
    assert abs(float(np.linalg.norm(vec_zh)) - 1.0) < 1e-5


@_skip_no_deps
def test_vector_node_create_and_search():
    """创建向量节点后，可通过向量相似度检索到。"""
    try:
        store = ChromaVectorStore(path=".test_chroma_search")
        emb = EmbeddingPipeline()
        space = VectorCognitiveSpace(vector_store=store, embedding=emb)

        vec = emb.embed_text("苹果")
        node_id = space.add_node(vec, label="苹果", kind="trunk")

        results = space.find_semantic_neighbors(vec, top_k=1, threshold=0.9)
        assert len(results) == 1
        assert results[0][0] == node_id
        assert results[0][1] > 0.9
    finally:
        _cleanup(".test_chroma_search")


@_skip_no_deps
def test_tree_basic_operations():
    """认知树基本操作：主干 + 特征 + 描述 + 打印。"""
    try:
        tree, space = _make_tree(".test_chroma_tree_ops")

        # 添加主干（自动嵌入）
        tree.add_trunk("苹果")
        assert tree.trunk_count() == 1

        # 手动添加特征
        tree.add_feature_child("苹果", "颜色", "红")
        assert tree.has_feature("苹果", "颜色", "红")
        assert "颜色-红" in tree.children_ids("苹果")
        assert tree.feature_count("苹果") == 1

        # 描述
        desc = tree.describe("苹果")
        assert "苹果" in desc
        assert "红" in desc

        # 打印树
        tree_str = tree.print_tree("苹果")
        assert "苹果" in tree_str
        assert "颜色-红" in tree_str

        # 向量存储中有记录
        assert space.vector_store.count() >= 2

        # 再添加一个特征
        tree.add_feature_child("苹果", "味道", "甜")
        assert tree.feature_count("苹果") == 2
        assert space.vector_store.count() >= 3
    finally:
        _cleanup(".test_chroma_tree_ops")


@_skip_no_deps
def test_tree_activate():
    """激活操作提升节点权重。"""
    try:
        tree, _ = _make_tree(".test_chroma_activate")
        tree.add_trunk("苹果")

        node = tree.get_node("苹果")
        initial_w = float(node.get("activation_weight", 0.0))

        tree.activate("苹果")

        node = tree.get_node("苹果")
        assert float(node.get("activation_weight", 0.0)) > initial_w
    finally:
        _cleanup(".test_chroma_activate")


@_skip_no_deps
def test_gate_match_existing_trunk():
    """门控：输入「苹果」匹配到已有主干「苹果」。"""
    try:
        tree, _ = _make_tree(".test_chroma_gate_match")
        tree.add_trunk("苹果")

        gate = InformationTriggerGate(tree)
        result = gate.process("苹果")

        assert result.matched_trunk == "苹果"
        assert result.matched_trunk_similarity > 0.8
        assert not result.is_new_trunk
        assert result.prediction_error < 0.2
    finally:
        _cleanup(".test_chroma_gate_match")


@_skip_no_deps
def test_gate_cross_language_match():
    """门控：输入「apple」跨语言匹配到已有主干「苹果」。"""
    try:
        tree, _ = _make_tree(".test_chroma_gate_xlang")
        tree.add_trunk("苹果")

        gate = InformationTriggerGate(tree)
        result = gate.process("apple")

        assert result.matched_trunk == "苹果"
        assert result.matched_trunk_similarity > 0.5
        assert not result.is_new_trunk
    finally:
        _cleanup(".test_chroma_gate_xlang")


@_skip_no_deps
def test_gate_create_new_trunk():
    """门控：输入完全不同的词时创建新主干。"""
    try:
        tree, _ = _make_tree(".test_chroma_gate_new")
        tree.add_trunk("苹果")

        gate = InformationTriggerGate(tree)
        result = gate.process("计算机")

        assert result.is_new_trunk
        assert result.prediction_error == 1.0
        assert tree.trunk_count() == 2
    finally:
        _cleanup(".test_chroma_gate_new")


@_skip_no_deps
def test_dynamics_hebbian_and_decay():
    """动力学引擎：Hebbian 增强 + 全局衰减。"""
    try:
        tree, _ = _make_tree(".test_chroma_dynamics")
        tree.add_trunk("苹果")
        tree.add_feature_child("苹果", "颜色", "红")

        trunk_id = tree._find_by_label("苹果")
        child_id = tree._find_feature_by_dim_val(trunk_id, "颜色", "红")

        engine = NeuroDynamicsEngine(tree.graph)

        # Hebbian 增强
        initial_w = engine.get_edge_weight(trunk_id, child_id)
        engine.hebbian_reinforce(trunk_id, child_id)
        assert engine.get_edge_weight(trunk_id, child_id) > initial_w

        # 全局衰减
        before_w = engine.get_edge_weight(trunk_id, child_id)
        engine.apply_decay()
        assert engine.get_edge_weight(trunk_id, child_id) < before_w

        # 衰减后权重不为 0（数据不删除）
        assert engine.get_edge_weight(trunk_id, child_id) > 0
    finally:
        _cleanup(".test_chroma_dynamics")


@_skip_no_deps
def test_acceptance_full_scenario():
    """完整验收场景：苹果 → 添加特征 → 再提苹果 → describe。

    Step 4A 诚实范围：
    - 门控匹配主干（向量）
    - 特征通过 add_feature_child 手动添加（特征繁衍为 Step 4B）
    - 动力学强化已有路径
    """
    try:
        tree, _ = _make_tree(".test_chroma_acceptance")
        gate = InformationTriggerGate(tree)

        # 第一次输入「苹果」→ 创建主干
        result = gate.process("苹果")
        assert result.is_new_trunk
        assert tree.trunk_count() == 1

        # 手动添加特征（Step 4B 将通过扩散激活自动繁衍）
        tree.add_feature_child("苹果", "颜色", "红")
        tree.add_feature_child("苹果", "味道", "甜")

        # 再提苹果 → 匹配到已有主干，激活，不创建新节点
        before_count = tree.feature_count("苹果")
        result2 = gate.process("苹果")
        assert result2.matched_trunk == "苹果"
        assert not result2.is_new_trunk
        assert result2.prediction_error < 0.2
        assert tree.feature_count("苹果") == before_count

        # describe 输出包含已知特征
        desc = tree.describe("苹果")
        assert "苹果" in desc
        assert "红" in desc
        assert "甜" in desc

        # 动力学强化路径
        engine = NeuroDynamicsEngine(tree.graph)
        trunk_id = tree._find_by_label("苹果")
        for child_id in tree.graph.successors(trunk_id):
            engine.hebbian_reinforce(trunk_id, child_id)

        # 验证强化后权重 > 初始 0.1
        for child_id in tree.graph.successors(trunk_id):
            assert engine.get_edge_weight(trunk_id, child_id) > 0.1
    finally:
        _cleanup(".test_chroma_acceptance")
