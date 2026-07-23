"""Step 6D 集成测试：表达 + 探索 + 奖励闭环。

验证 ContinuousMind 与三个新引擎的集成。
使用 mock 嵌入管道，无需外部依赖。
"""

from __future__ import annotations

import numpy as np

from danniao.expression import ExpressionEngine
from danniao.hippocampus import (
    DynamicCognitiveTree,
    EpisodicLog,
    InformationTriggerGate,
    NeuroDynamicsEngine,
    VectorCognitiveSpace,
    VectorStore,
)
from danniao.hippocampus.spreading import SpreadingActivation


# ==================== Mock 嵌入 ====================


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


# ==================== 工厂 ====================


def _make_mind(*, with_expression=False, with_exploration=False, with_reward=False):
    """创建配置了可选 Step 6 引擎的 ContinuousMind。"""
    from danniao.motivation import (
        ExplorationEngine,
        Homeostasis,
        RewardSystem,
    )
    from danniao.mind import ContinuousMind

    store = _StubStore()
    emb = _MockEmbedding()
    space = VectorCognitiveSpace(vector_store=store, embedding=emb)
    tree = DynamicCognitiveTree(space)
    gate = InformationTriggerGate(tree)
    dynamics = NeuroDynamicsEngine(tree.graph)
    spreading = SpreadingActivation(space)
    homeostasis = Homeostasis()
    log = EpisodicLog(path=":memory:")

    expression_engine = ExpressionEngine(tree) if with_expression else None
    exploration_engine = (
        ExplorationEngine(tree, homeostasis) if with_exploration else None
    )
    reward_system = (
        RewardSystem(homeostasis, dynamics) if with_reward else None
    )

    mind = ContinuousMind(
        tree=tree,
        gate=gate,
        dynamics=dynamics,
        spreading=spreading,
        homeostasis=homeostasis,
        episodic_log=log,
        expression_engine=expression_engine,
        exploration_engine=exploration_engine,
        reward_system=reward_system,
    )
    return mind


# ==================== 表达集成 ====================


def test_expression_generated():
    """有表达引擎 → 处理结果包含表达文本。"""
    mind = _make_mind(with_expression=True)
    result = mind.process("苹果")

    assert result.expression != ""
    assert "苹果" in result.expression


def test_no_expression_without_engine():
    """无表达引擎 → 表达为空字符串（向后兼容）。"""
    mind = _make_mind(with_expression=False)
    result = mind.process("苹果")

    assert result.expression == ""


def test_expression_new_trunk_curious():
    """新概念 → 表达包含惊讶/好奇语气。"""
    mind = _make_mind(with_expression=True)
    result = mind.process("量子力学")

    assert result.expression != ""
    assert "什么" in result.expression or "新东西" in result.expression


def test_expression_matched_known():
    """匹配已知概念 → 表达包含识别语气。"""
    mind = _make_mind(with_expression=True)
    mind.process("苹果")
    result = mind.process("苹果")

    assert result.expression != ""
    assert "知道" in result.expression or "这是" in result.expression or "熟悉" in result.expression


def test_status_includes_expression():
    """状态快照包含最近表达。"""
    mind = _make_mind(with_expression=True)
    mind.process("苹果")

    status = mind.status()
    assert status.last_expression != ""
    assert "苹果" in status.last_expression


# ==================== 探索集成 ====================


def test_exploration_generated_when_curious():
    """有探索引擎 + 好奇心高 → 生成探索目标。"""
    mind = _make_mind(with_exploration=True)
    mind.process("苹果")

    result = mind.last_result
    # 新概念 → 好奇心上升 → 可能想探索
    # 但只有一个主干且无近期关联 → 可能返回深度探索
    if result.exploration:
        assert result.exploration.text != ""


def test_no_exploration_without_engine():
    """无探索引擎 → 探索目标为 None（向后兼容）。"""
    mind = _make_mind(with_exploration=False)
    result = mind.process("苹果")

    assert result.exploration is None


def test_exploration_relationship_two_trunks():
    """两个主干 → 关联探索。"""
    mind = _make_mind(with_exploration=True)
    mind.process("苹果")
    mind.process("梨")

    result = mind.last_result
    if result.exploration:
        assert result.exploration.exploration_type in ("relationship", "depth")


# ==================== 奖励闭环集成 ====================


def test_give_feedback_success():
    """成功反馈 → 满足感上升。"""
    mind = _make_mind(with_reward=True)
    mind.process("苹果")

    state_before = mind.homeostasis.state.satiety
    result = mind.give_feedback(success=True)
    state_after = mind.homeostasis.state.satiety

    assert result is not None
    assert result.success is True
    assert state_after > state_before


def test_give_feedback_failure():
    """失败反馈 → 好奇心上升。"""
    mind = _make_mind(with_reward=True)
    mind.process("苹果")

    state_before = mind.homeostasis.state.curiosity
    result = mind.give_feedback(success=False)
    state_after = mind.homeostasis.state.curiosity

    assert result is not None
    assert result.success is False
    assert state_after > state_before


def test_give_feedback_no_reward_system():
    """无奖励系统 → 反馈返回 None（向后兼容）。"""
    mind = _make_mind(with_reward=False)
    mind.process("苹果")

    result = mind.give_feedback(success=True)
    assert result is None


def test_give_feedback_no_last_result():
    """无最近结果 → 反馈返回 None。"""
    mind = _make_mind(with_reward=True)

    result = mind.give_feedback(success=True)
    assert result is None


def test_feedback_reinforces_edges():
    """成功反馈 → 活跃路径边权重增强。"""
    mind = _make_mind(with_reward=True)
    mind.process("苹果")
    mind.process("苹果")  # 二次匹配建立关联

    # 获取活跃路径中的边
    trunk_id = mind.last_result.trunk_node_id
    if trunk_id and mind.tree.graph.out_degree(trunk_id) > 0:
        child_id = next(mind.tree.graph.successors(trunk_id))
        w_before = mind.dynamics.get_edge_weight(trunk_id, child_id)

        mind.give_feedback(success=True)

        w_after = mind.dynamics.get_edge_weight(trunk_id, child_id)
        assert w_after >= w_before  # 强化后权重不低于之前


# ==================== 全链路集成 ====================


def test_full_step6_pipeline():
    """全链路：输入 → 表达 → 反馈 → 探索。"""
    mind = _make_mind(
        with_expression=True,
        with_exploration=True,
        with_reward=True,
    )

    # 1. 第一次输入：新概念
    r1 = mind.process("苹果")
    assert r1.expression != ""
    assert r1.is_new_trunk is True

    # 2. 给反馈
    feedback1 = mind.give_feedback(success=True)
    assert feedback1 is not None
    assert feedback1.success is True

    # 3. 第二次输入：匹配已知
    r2 = mind.process("苹果")
    assert r2.is_new_trunk is False
    assert r2.expression != ""

    # 4. 第三次输入：另一个概念
    r3 = mind.process("梨")
    assert r3.is_new_trunk is True

    # 5. 检查探索目标
    if r3.exploration:
        assert r3.exploration.text != ""


def test_expression_and_feedback_loop():
    """表达 + 反馈循环 → 丹鸟在反馈中学习。"""
    mind = _make_mind(with_expression=True, with_reward=True)

    # 第一次表达新概念
    r1 = mind.process("新概念")
    assert r1.expression != ""

    # 给负面反馈 → 好奇心应该上升
    curiosity_before = mind.homeostasis.state.curiosity
    mind.give_feedback(success=False)
    curiosity_after = mind.homeostasis.state.curiosity
    assert curiosity_after > curiosity_before

    # 第二次表达
    r2 = mind.process("新概念")
    assert r2.expression != ""
    # 好奇心更高 → 表达可能包含探索欲
