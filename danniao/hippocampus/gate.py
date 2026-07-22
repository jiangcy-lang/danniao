"""信息触发沉淀门控：常规激活 vs 新维度繁衍。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from danniao.hippocampus.features import ParsedInput, parse_input
from danniao.hippocampus.tree import DynamicCognitiveTree

Action = Literal[
    "no_op",
    "spawned_trunk",
    "routine_activate",
    "spawned_children",
    "reinforced_children",
]


@dataclass
class GateResult:
    action: Action
    trunk: str | None
    prediction_error: bool = False
    spawned: list[str] = field(default_factory=list)
    message: str = ""
    parsed: ParsedInput | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "trunk": self.trunk,
            "prediction_error": self.prediction_error,
            "spawned": list(self.spawned),
            "message": self.message,
        }


class InformationTriggerGate:
    """对外部输入做认知门控，决定激活或繁衍。"""

    def __init__(self, tree: DynamicCognitiveTree) -> None:
        self.tree = tree

    def process(self, text: str) -> GateResult:
        parsed = parse_input(text)
        if parsed.trunk is None:
            return GateResult(
                action="no_op",
                trunk=None,
                message="未识别到已知主干概念",
                parsed=parsed,
            )

        trunk = parsed.trunk
        features = list(parsed.features)
        trunk_exists = self.tree.get_node(trunk) is not None

        if not trunk_exists:
            self.tree.add_trunk(trunk)
            spawned: list[str] = []
            if features:
                # 首句即高信息量：允许同次繁衍
                spawned = self._spawn_new(trunk, features)
                return GateResult(
                    action="spawned_children" if spawned else "spawned_trunk",
                    trunk=trunk,
                    prediction_error=bool(spawned),
                    spawned=spawned,
                    message=f"新建主干「{trunk}」" + (f"并繁衍 {spawned}" if spawned else ""),
                    parsed=parsed,
                )
            return GateResult(
                action="spawned_trunk",
                trunk=trunk,
                prediction_error=False,
                spawned=[],
                message=f"新建孤立主干「{trunk}」",
                parsed=parsed,
            )

        # 主干已存在
        self.tree.activate(trunk)

        if not features:
            return GateResult(
                action="routine_activate",
                trunk=trunk,
                prediction_error=False,
                message=f"常规输入：仅激活主干「{trunk}」",
                parsed=parsed,
            )

        new_feats = [
            (dim, val)
            for dim, val in features
            if not self.tree.has_feature(trunk, dim, val)
        ]
        if not new_feats:
            for dim, val in features:
                child_id = f"{dim}-{val}"
                if child_id in self.tree.graph:
                    self.tree.activate(child_id)
            return GateResult(
                action="reinforced_children",
                trunk=trunk,
                prediction_error=False,
                message=f"特征已存在：强化「{trunk}」下匹配子节点",
                parsed=parsed,
            )

        spawned = self._spawn_new(trunk, new_feats)
        return GateResult(
            action="spawned_children",
            trunk=trunk,
            prediction_error=True,
            spawned=spawned,
            message=f"预测误差：繁衍子节点 {spawned}",
            parsed=parsed,
        )

    def _spawn_new(self, trunk: str, features: list[tuple[str, str]]) -> list[str]:
        spawned: list[str] = []
        for dim, val in features:
            if not self.tree.has_feature(trunk, dim, val):
                child_id = self.tree.add_feature_child(trunk, dim, val)
                spawned.append(child_id)
        return spawned
