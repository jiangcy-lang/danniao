"""Step 5b 验收测试：持续心智。

测试同步模式 process() 的完整链路：
门控 → 日志 → 内稳态 → 激活 → 扩散 → 强化 → 状态更新

以及状态查询和多次交互。
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
)
from danniao.hippocampus.spreading import SpreadingActivation
from danniao.motivation import Homeostasis
from danniao.mind import ContinuousMind, ProcessResult


# ---------- 测试工具 ----------


class _MockEmbedding:
    """确定性 mock 嵌入：每个唯一文本分配一个正交 one-hot 向量。

    确保不同文本的余弦相似度为 0，同文本为 1.0。
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim
        self._cache: dict[str, np.ndarray] = {}
        self._counter = 0

    def embed_text(self, text: str) -> np.ndarray:
        if text not in self._cache:
            vec = np.zeros(self._dim, dtype=np.float32)
            vec[self._counter % self._dim] = 1.0
            self._counter += 1
            self._cache[text] = vec
        return self._cache[text]

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


def _make_mind_mock() -> ContinuousMind:
    """创建带 mock 依赖的持续心智（无外部依赖）。"""
    store = _StubStore()
    emb = _MockEmbedding()
    space = VectorCognitiveSpace(vector_store=store, embedding=emb)
    tree = DynamicCognitiveTree(space)
    gate = InformationTriggerGate(tree)
    dynamics = NeuroDynamicsEngine(tree.graph)
    spreading = SpreadingActivation(space)
    homeostasis = Homeostasis()
    log = EpisodicLog(path=":memory:")

    return ContinuousMind(
        tree=tree,
        gate=gate,
        dynamics=dynamics,
        spreading=spreading,
        homeostasis=homeostasis,
        episodic_log=log,
    )


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


# ==================== 同步模式 ====================


def test_process_returns_result():
    """process() 返回 ProcessResult。"""
    mind = _make_mind_mock()
    result = mind.process("苹果")

    assert isinstance(result, ProcessResult)
    assert result.text == "苹果"
    assert result.is_new_trunk  # 第一次输入 → 新主干
    assert result.trunk_count == 1
    assert result.node_count >= 1


def test_process_logs_to_episodic():
    """process() 将输入记录到原始日志。"""
    mind = _make_mind_mock()
    mind.process("苹果")

    entries = mind.episodic_log.replay()
    assert len(entries) == 1
    assert entries[0]["text"] == "苹果"


def test_process_updates_homeostasis():
    """process() 更新内稳态状态。"""
    mind = _make_mind_mock()
    before = mind.homeostasis.state.curiosity

    # 新主干 → 高预测误差 → 好奇心上升
    mind.process("完全不同的东西")

    after = mind.homeostasis.state.curiosity
    assert after > before


def test_process_match_lowers_prediction_error():
    """匹配已有主干时预测误差低。"""
    mind = _make_mind_mock()

    # 第一次：创建主干
    r1 = mind.process("苹果")
    assert r1.is_new_trunk
    assert r1.prediction_error == 1.0

    # 第二次：匹配到已有主干
    r2 = mind.process("苹果")
    assert not r2.is_new_trunk
    assert r2.prediction_error < 0.2


def test_multiple_inputs_grow_tree():
    """多次输入使认知树生长。"""
    mind = _make_mind_mock()

    mind.process("apple")
    assert mind.tree.trunk_count() == 1

    mind.process("zebra")
    assert mind.tree.trunk_count() == 2

    mind.process("quantum")
    assert mind.tree.trunk_count() == 3


def test_internal_state_changes_over_interactions():
    """内稳态状态随交互变化。"""
    mind = _make_mind_mock()

    # 第一次交互：高预测误差 → 好奇心上升
    r1 = mind.process("苹果")
    curiosity_1 = r1.internal_state.curiosity

    # 第二次匹配：置信度上升
    r2 = mind.process("苹果")
    curiosity_2 = r2.internal_state.curiosity
    confidence_2 = r2.internal_state.confidence

    # 匹配后置信度应高于初始
    assert confidence_2 > 0.3  # 初始 0.3


def test_status_returns_snapshot():
    """status() 返回完整状态快照。"""
    mind = _make_mind_mock()
    mind.process("苹果")

    status = mind.status()

    assert status.node_count >= 1
    assert status.trunk_count == 1
    assert status.is_running is False  # 同步模式未启动 live()
    assert status.dominant_drive is not None
    assert len(status.recent_nodes) >= 1


def test_last_result_tracked():
    """last_result 记录最近一次处理。"""
    mind = _make_mind_mock()

    assert mind.last_result is None

    mind.process("苹果")
    assert mind.last_result is not None
    assert mind.last_result.text == "苹果"

    mind.process("香蕉")
    assert mind.last_result.text == "香蕉"


def test_activated_nodes_in_result():
    """处理结果包含扩散激活的节点。"""
    mind = _make_mind_mock()

    # 创建主干 + 手动添加特征
    mind.process("苹果")
    mind.tree.add_feature_child("苹果", "颜色", "红")

    # 再次处理苹果 → 匹配 → 扩散激活
    result = mind.process("苹果")

    # 扩散激活结果应非空（至少包含种子节点）
    assert len(result.activated_nodes) >= 1


# ==================== 真实依赖集成测试 ====================


@_skip_no_deps
def test_full_pipeline_with_real_deps():
    """真实依赖：完整管道验证。"""
    _cleanup(".test_chroma_mind")  # 清理上一轮残留（Windows 文件锁可能阻止删除）
    try:
        store = ChromaVectorStore(path=".test_chroma_mind")
        emb = EmbeddingPipeline()
        space = VectorCognitiveSpace(vector_store=store, embedding=emb)
        tree = DynamicCognitiveTree(space)
        gate = InformationTriggerGate(tree)
        dynamics = NeuroDynamicsEngine(tree.graph)
        spreading = SpreadingActivation(space)
        homeostasis = Homeostasis()
        log = EpisodicLog(path=":memory:")

        mind = ContinuousMind(
            tree=tree,
            gate=gate,
            dynamics=dynamics,
            spreading=spreading,
            homeostasis=homeostasis,
            episodic_log=log,
        )

        # 1. 第一次输入「苹果」→ 新主干
        r1 = mind.process("苹果")
        assert r1.is_new_trunk
        assert r1.trunk_count == 1

        # 2. 添加特征
        tree.add_feature_child("苹果", "颜色", "红")
        tree.add_feature_child("苹果", "味道", "甜")

        # 3. 再次输入「苹果」→ 匹配
        r2 = mind.process("苹果")
        assert not r2.is_new_trunk
        assert r2.matched_trunk == "苹果"
        assert r2.prediction_error < 0.2

        # 4. 输入「apple」→ 跨语言匹配
        r3 = mind.process("apple")
        assert not r3.is_new_trunk
        assert r3.matched_trunk == "苹果"

        # 5. 输入「计算机」→ 新主干
        r4 = mind.process("计算机")
        assert r4.is_new_trunk
        assert r4.trunk_count == 2

        # 6. 状态检查
        status = mind.status()
        assert status.trunk_count == 2
        assert status.node_count >= 4  # 2 主干 + 2 特征
        assert len(status.recent_nodes) >= 4

        # 7. 日志检查
        entries = log.replay()
        assert len(entries) == 4
    finally:
        _cleanup(".test_chroma_mind")
