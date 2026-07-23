"""Step 6A 验收测试：奖励系统。

验证交付反馈 → 内稳态更新 + 路径强化/弱化。
所有测试无需外部依赖。
"""

from __future__ import annotations

import networkx as nx

from danniao.hippocampus.dynamics import NeuroDynamicsEngine
from danniao.motivation import Homeostasis, RewardResult, RewardSystem


# ==================== 工具 ====================


def _make_dynamics_with_path(node_ids: list[str]) -> tuple[NeuroDynamicsEngine, list[str]]:
    """创建带预设边的动力学引擎。

    返回 (engine, node_ids)。
    """
    graph = nx.DiGraph()
    for nid in node_ids:
        graph.add_node(nid, label=nid, kind="trunk", activation_weight=0.5)
    for i in range(len(node_ids) - 1):
        graph.add_edge(node_ids[i], node_ids[i + 1], weight=0.3, last_activated_time="")
    engine = NeuroDynamicsEngine(graph)
    return engine, node_ids


# ==================== 成功交付 ====================


def test_success_raises_satiety():
    """成功交付 → 满足感上升。"""
    homeo = Homeostasis()
    reward = RewardSystem(homeo, NeuroDynamicsEngine(nx.DiGraph()))
    before = homeo.state.satiety

    result = reward.observe_delivery(success=True)

    assert result.satiety_after > result.satiety_before
    assert result.satiety_after > before


def test_success_reinforces_edges():
    """成功交付 + 活跃路径 → 边权重增强。"""
    engine, ids = _make_dynamics_with_path(["A", "B", "C"])
    original_w = engine.get_edge_weight("A", "B")
    reward = RewardSystem(Homeostasis(), engine, reinforce_delta=0.15)

    result = reward.observe_delivery(success=True, active_path=ids)

    assert len(result.reinforced_edges) == 2
    assert ("A", "B") in result.reinforced_edges
    assert ("B", "C") in result.reinforced_edges
    assert engine.get_edge_weight("A", "B") > original_w
    assert engine.get_edge_weight("A", "B") >= original_w + 0.1


def test_success_no_curiosity_spike():
    """成功交付 → 好奇心不应大幅上升（满足时略微下降）。"""
    homeo = Homeostasis()
    reward = RewardSystem(homeo, NeuroDynamicsEngine(nx.DiGraph()))
    before = homeo.state.curiosity

    reward.observe_delivery(success=True)

    assert homeo.state.curiosity <= before


# ==================== 失败交付 ====================


def test_failure_lowers_satiety():
    """失败交付 → 满足感下降。"""
    homeo = Homeostasis()
    reward = RewardSystem(homeo, NeuroDynamicsEngine(nx.DiGraph()))
    before = homeo.state.satiety

    result = reward.observe_delivery(success=False)

    assert result.satiety_after < before


def test_failure_raises_curiosity():
    """失败交付 → 好奇心上升（想知道为什么失败）。"""
    homeo = Homeostasis()
    reward = RewardSystem(homeo, NeuroDynamicsEngine(nx.DiGraph()))
    before = homeo.state.curiosity

    reward.observe_delivery(success=False)

    assert homeo.state.curiosity > before


def test_failure_weakens_edges():
    """失败交付 + 活跃路径 → 边权重弱化。"""
    engine, ids = _make_dynamics_with_path(["A", "B", "C"])
    original_w = engine.get_edge_weight("A", "B")
    reward = RewardSystem(Homeostasis(), engine, weaken_delta=0.1)

    result = reward.observe_delivery(success=False, active_path=ids)

    assert len(result.weakened_edges) == 2
    assert ("A", "B") in result.weakened_edges
    assert engine.get_edge_weight("A", "B") < original_w


# ==================== 无路径 / 边界 ====================


def test_no_active_path_only_homeostasis():
    """无活跃路径 → 只更新内稳态，不操作边。"""
    engine, _ = _make_dynamics_with_path(["A", "B"])
    original_w = engine.get_edge_weight("A", "B")
    reward = RewardSystem(Homeostasis(), engine)

    result = reward.observe_delivery(success=True, active_path=None)

    assert result.reinforced_edges == []
    assert result.weakened_edges == []
    assert engine.get_edge_weight("A", "B") == original_w


def test_single_node_path_no_edges():
    """单节点路径 → 不操作边（需 >= 2 个节点才有边）。"""
    engine, _ = _make_dynamics_with_path(["A", "B"])
    original_w = engine.get_edge_weight("A", "B")
    reward = RewardSystem(Homeostasis(), engine)

    result = reward.observe_delivery(success=True, active_path=["A"])

    assert result.reinforced_edges == []
    assert engine.get_edge_weight("A", "B") == original_w


def test_empty_path_no_edges():
    """空路径 → 不操作边。"""
    engine, _ = _make_dynamics_with_path(["A", "B"])
    original_w = engine.get_edge_weight("A", "B")
    reward = RewardSystem(Homeostasis(), engine)

    result = reward.observe_delivery(success=False, active_path=[])

    assert result.weakened_edges == []
    assert engine.get_edge_weight("A", "B") == original_w


# ==================== 累积效果 ====================


def test_multiple_successes_accumulate_satiety():
    """多次成功 → 满足感累积上升。"""
    reward = RewardSystem(Homeostasis(), NeuroDynamicsEngine(nx.DiGraph()))

    r1 = reward.observe_delivery(success=True)
    r2 = reward.observe_delivery(success=True)
    r3 = reward.observe_delivery(success=True)

    assert r3.satiety_after > r2.satiety_after > r1.satiety_after


def test_multiple_failures_accumulate_curiosity():
    """多次失败 → 好奇心累积上升。"""
    reward = RewardSystem(Homeostasis(), NeuroDynamicsEngine(nx.DiGraph()))

    r1 = reward.observe_delivery(success=False)
    r2 = reward.observe_delivery(success=False)

    assert r2.curiosity_after > r1.curiosity_after


def test_alternating_feedback():
    """交替反馈 → 满足感和好奇心交替变化。"""
    reward = RewardSystem(Homeostasis(), NeuroDynamicsEngine(nx.DiGraph()))

    r_success = reward.observe_delivery(success=True)
    assert r_success.satiety_after > r_success.satiety_before

    r_fail = reward.observe_delivery(success=False)
    assert r_fail.satiety_after < r_success.satiety_after
    assert r_fail.curiosity_after > r_success.curiosity_after


# ==================== 结果完整性 ====================


def test_result_records_before_after():
    """结果记录处理前后的状态。"""
    reward = RewardSystem(Homeostasis(), NeuroDynamicsEngine(nx.DiGraph()))

    result = reward.observe_delivery(success=True, active_path=["A", "B"])

    assert result.satiety_before != result.satiety_after
    assert result.success is True


def test_edge_weight_not_below_zero():
    """弱化不会使边权重低于 0。"""
    graph = nx.DiGraph()
    graph.add_node("A", label="A")
    graph.add_node("B", label="B")
    graph.add_edge("A", "B", weight=0.02, last_activated_time="")
    engine = NeuroDynamicsEngine(graph)
    reward = RewardSystem(Homeostasis(), engine, weaken_delta=0.1)

    reward.observe_delivery(success=False, active_path=["A", "B"])

    assert engine.get_edge_weight("A", "B") == 0.0


def test_reinforce_creates_edge_if_missing():
    """强化时边不存在 → 创建新边（Hebbian 行为）。"""
    graph = nx.DiGraph()
    graph.add_node("A", label="A")
    graph.add_node("B", label="B")
    engine = NeuroDynamicsEngine(graph)
    reward = RewardSystem(Homeostasis(), engine, reinforce_delta=0.1)

    assert not graph.has_edge("A", "B")

    reward.observe_delivery(success=True, active_path=["A", "B"])

    assert graph.has_edge("A", "B")
    assert engine.get_edge_weight("A", "B") > 0
