"""奖励系统：交付反馈 → 内稳态更新 + 路径强化/弱化（Step 6A）。

丹鸟产生表达后，造物主（或环境）给出反馈。
奖励系统处理反馈，更新内稳态，并强化或弱化导致该结果的认识路径。

成功 → 满足感上升 + Hebbian 强化活跃路径（LTP）
失败 → 满足感下降 + 好奇心上升 + 活跃路径弱化（LTD）

这是丹鸟"学习"的关键闭环：表达 → 反馈 → 强化/弱化 → 下次更好。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from danniao.hippocampus.dynamics import NeuroDynamicsEngine
    from danniao.motivation import Homeostasis


@dataclass
class RewardResult:
    """奖励处理结果。"""

    success: bool
    """交付是否成功。"""

    satiety_before: float = 0.0
    """处理前满足感。"""

    satiety_after: float = 0.0
    """处理后满足感。"""

    curiosity_before: float = 0.0
    """处理前好奇心。"""

    curiosity_after: float = 0.0
    """处理后好奇心。"""

    reinforced_edges: list[tuple[str, str]] = field(default_factory=list)
    """被强化的边列表（成功时）。"""

    weakened_edges: list[tuple[str, str]] = field(default_factory=list)
    """被弱化的边列表（失败时）。"""


class RewardSystem:
    """奖励系统：处理交付反馈，驱动学习闭环。

    用法::

        reward = RewardSystem(homeostasis, dynamics)
        result = reward.observe_delivery(
            success=True,
            active_path=[trunk_id, neighbor_id1, neighbor_id2],
        )
        # 成功 → homeostasis 满足感上升 + 路径边权重增强
        # 失败 → homeostasis 好奇心上升 + 路径边权重弱化
    """

    def __init__(
        self,
        homeostasis: Homeostasis,
        dynamics: NeuroDynamicsEngine,
        *,
        reinforce_delta: float = 0.1,
        weaken_delta: float = 0.05,
    ) -> None:
        """初始化奖励系统。

        Args:
            homeostasis: 内稳态引擎
            dynamics: 神经动力学引擎
            reinforce_delta: 成功时边权重增量（Hebbian LTP）
            weaken_delta: 失败时边权重减量（LTD）
        """
        self.homeostasis = homeostasis
        self.dynamics = dynamics
        self.reinforce_delta = reinforce_delta
        self.weaken_delta = weaken_delta

    def observe_delivery(
        self,
        success: bool,
        active_path: list[str] | None = None,
    ) -> RewardResult:
        """处理交付反馈。

        1. 更新内稳态（满足感/好奇心）
        2. 强化或弱化活跃路径上的边

        Args:
            success: 交付是否成功
            active_path: 活跃路径节点 ID 列表 [trunk_id, neighbor1, ...]
                         成功时强化路径上相邻节点间的边；
                         失败时弱化路径上相邻节点间的边。

        Returns:
            RewardResult 处理结果
        """
        # 记录处理前状态
        state_before = self.homeostasis.snapshot()
        result = RewardResult(
            success=success,
            satiety_before=state_before.satiety,
            curiosity_before=state_before.curiosity,
        )

        # 1. 更新内稳态
        self.homeostasis.observe_delivery(success)

        # 2. 强化或弱化活跃路径
        if active_path and len(active_path) >= 2:
            if success:
                result.reinforced_edges = self._reinforce_path(active_path)
            else:
                result.weakened_edges = self._weaken_path(active_path)

        # 记录处理后状态
        state_after = self.homeostasis.snapshot()
        result.satiety_after = state_after.satiety
        result.curiosity_after = state_after.curiosity

        return result

    def _reinforce_path(self, path: list[str]) -> list[tuple[str, str]]:
        """强化路径上相邻节点间的边（Hebbian LTP）。

        Args:
            path: 节点 ID 列表，按顺序

        Returns:
            被强化的边列表
        """
        reinforced: list[tuple[str, str]] = []
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            self.dynamics.hebbian_reinforce(
                source, target, delta=self.reinforce_delta
            )
            reinforced.append((source, target))
        return reinforced

    def _weaken_path(self, path: list[str]) -> list[tuple[str, str]]:
        """弱化路径上相邻节点间的边（LTD）。

        Args:
            path: 节点 ID 列表，按顺序

        Returns:
            被弱化的边列表
        """
        weakened: list[tuple[str, str]] = []
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            self.dynamics.weaken_edge(
                source, target, delta=self.weaken_delta
            )
            weakened.append((source, target))
        return weakened
