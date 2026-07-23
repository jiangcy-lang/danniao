"""探索引擎：好奇心驱动的主动探索（Step 6C）。

丹鸟不是被动等指令的问答机。当好奇心高、能量充足、不太满足时，
它会主动产生探索目标——提出问题、寻找关联、追求新知。

探索目标类型：
- 深度探索：已知概念但理解浅（主干特征少）→ "我想了解更多关于[X]"
- 关联探索：两个近期概念但未建立关联 → "[X]和[Y]之间有什么关系？"
- 求新探索：认知树几乎空 → "有什么新东西可以学？"

探索引擎只读认知状态并生成目标，不直接执行探索。
执行由 ContinuousMind 决定（可能输出给用户、或触发搜索）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from danniao.motivation import Drive

if TYPE_CHECKING:
    from danniao.hippocampus.tree import DynamicCognitiveTree
    from danniao.motivation import Homeostasis


@dataclass
class ExplorationTarget:
    """探索目标。"""

    text: str
    """探索问题/探针文本。"""

    target_node_ids: list[str] = field(default_factory=list)
    """触发探索的节点 ID 列表。"""

    drive: Drive = Drive.CURIOSITY
    """触发探索的驱动力。"""

    urgency: float = 0.5
    """探索紧迫度（0~1）。好奇心越高越紧迫。"""

    exploration_type: str = "depth"
    """探索类型：depth / relationship / novelty / observe / study / ask_external。"""

    needs_external: bool = False
    """是否需要联网获取外部信息。"""


class ExplorationEngine:
    """探索引擎：好奇心驱动的主动探索目标生成。

    用法::

        engine = ExplorationEngine(tree, homeostasis)
        target = engine.propose(recent_node_ids=["n1", "n2"])
        if target:
            print(target.text)  # "「苹果」和「梨」之间有什么关系？"
    """

    def __init__(
        self,
        tree: DynamicCognitiveTree,
        homeostasis: Homeostasis,
    ) -> None:
        """初始化探索引擎。

        Args:
            tree: 认知树（用于查询节点信息和知识缺口）
            homeostasis: 内稳态引擎（用于判断是否想探索）
        """
        self.tree = tree
        self.homeostasis = homeostasis

    def propose(
        self,
        recent_node_ids: list[str] | None = None,
    ) -> ExplorationTarget | None:
        """生成探索目标。

        如果丹鸟当前不想探索（好奇心低/能量低/太满足），返回 None。

        Args:
            recent_node_ids: 最近处理的节点 ID 列表

        Returns:
            ExplorationTarget 或 None（不想探索时）
        """
        # 1. 检查是否想探索
        if not self.homeostasis.wants_to_explore():
            return None

        state = self.homeostasis.snapshot()
        urgency = min(1.0, state.curiosity)
        drive = self.homeostasis.dominant_drive()

        recent_node_ids = recent_node_ids or []

        # 2. 尝试关联探索：两个近期主干
        target = self._try_relationship_exploration(
            recent_node_ids, drive, urgency
        )
        if target:
            return target

        # 3. 尝试深度探索：近期主干特征少
        target = self._try_depth_exploration(
            recent_node_ids, drive, urgency
        )
        if target:
            return target

        # 4. 求新探索：认知树空或近期无主干
        target = self._try_novelty_exploration(drive, urgency)
        if target:
            return target

        # 5. 外部探索：有主干但深度/关联未命中
        target = self._try_external_exploration(recent_node_ids, drive, urgency)
        if target:
            return target

        return None

    # ==================== 探索策略 ====================

    def _try_relationship_exploration(
        self,
        recent_node_ids: list[str],
        drive: Drive,
        urgency: float,
    ) -> ExplorationTarget | None:
        """关联探索：两个近期主干之间是否有关联？

        条件：近期有两个以上的主干节点，且它们之间没有图边。
        """
        recent_trunks = self._get_recent_trunks(recent_node_ids, limit=5)

        if len(recent_trunks) < 2:
            return None

        # 找两个没有直接边连接的主干
        for i in range(len(recent_trunks)):
            for j in range(i + 1, len(recent_trunks)):
                id_a, label_a = recent_trunks[i]
                id_b, label_b = recent_trunks[j]

                # 检查是否已有边（任一方向）
                if self.tree.graph.has_edge(id_a, id_b):
                    continue
                if self.tree.graph.has_edge(id_b, id_a):
                    continue

                return ExplorationTarget(
                    text=f"「{label_a}」和「{label_b}」之间有什么关系？",
                    target_node_ids=[id_a, id_b],
                    drive=drive,
                    urgency=urgency,
                    exploration_type="relationship",
                )

        return None

    def _try_depth_exploration(
        self,
        recent_node_ids: list[str],
        drive: Drive,
        urgency: float,
    ) -> ExplorationTarget | None:
        """深度探索：已知概念但理解浅。

        条件：近期有主干节点，且该主干子节点（特征）少于阈值。
        """
        recent_trunks = self._get_recent_trunks(recent_node_ids, limit=3)

        for node_id, label in recent_trunks:
            feature_count = self._count_features(node_id)
            if feature_count < 2:
                return ExplorationTarget(
                    text=f"我想了解更多关于「{label}」的事情。",
                    target_node_ids=[node_id],
                    drive=drive,
                    urgency=urgency,
                    exploration_type="depth",
                )

        return None

    def _try_novelty_exploration(
        self,
        drive: Drive,
        urgency: float,
    ) -> ExplorationTarget | None:
        """求新探索：认知树空或所有主干都已被充分探索。

        条件：认知树没有主干，或所有主干都有足够特征。
        """
        trunk_count = self.tree.trunk_count()

        if trunk_count == 0:
            return ExplorationTarget(
                text="有什么新东西可以学吗？",
                target_node_ids=[],
                drive=drive,
                urgency=urgency,
                exploration_type="novelty",
            )

        return None

    def _try_external_exploration(
        self,
        recent_node_ids: list[str],
        drive: Drive,
        urgency: float,
    ) -> ExplorationTarget | None:
        """外部探索：有主干但深度/关联未命中时，想从外部世界了解。

        条件：认知树有主干，且近期有主干节点，但深度和关联探索都未命中。
        优先级最低，作为探索意愿的 fallback。
        """
        if self.tree.trunk_count() == 0:
            return None

        recent_trunks = self._get_recent_trunks(recent_node_ids, limit=1)
        if not recent_trunks:
            return None

        node_id, label = recent_trunks[0]
        return ExplorationTarget(
            text=f"「{label}」在外部世界里是怎样的？",
            target_node_ids=[node_id],
            drive=drive,
            urgency=urgency,
            exploration_type="ask_external",
            needs_external=True,
        )

    # ==================== 工具方法 ====================

    def _get_recent_trunks(
        self,
        recent_node_ids: list[str],
        *,
        limit: int = 5,
    ) -> list[tuple[str, str]]:
        """从近期节点中提取主干节点 (node_id, label)。

        按最近优先排列，去重。
        """
        result: list[tuple[str, str]] = []
        seen: set[str] = set()

        for nid in reversed(recent_node_ids):
            if nid in seen:
                continue
            if nid not in self.tree.graph:
                continue

            nd = self.tree.graph.nodes[nid]
            if nd.get("kind") != "trunk":
                continue

            label = str(nd.get("label", nid))
            result.append((nid, label))
            seen.add(nid)

            if len(result) >= limit:
                break

        return result

    def _count_features(self, trunk_id: str) -> int:
        """统计主干下的特征子节点数。"""
        count = 0
        for child_id in self.tree.graph.successors(trunk_id):
            child = self.tree.graph.nodes[child_id]
            if child.get("kind") == "feature":
                count += 1
        return count
