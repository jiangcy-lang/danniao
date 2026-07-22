"""信息触发沉淀门控：向量相似度匹配 + 常规激活 / 新维度繁衍。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from danniao.hippocampus.embeddings import MultimodalEncoder
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
    cosine_similarity: float | None = None
    input_modality: str = "text"

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "trunk": self.trunk,
            "prediction_error": self.prediction_error,
            "spawned": list(self.spawned),
            "message": self.message,
            "cosine_similarity": self.cosine_similarity,
            "input_modality": self.input_modality,
        }


class InformationTriggerGate:
    """对外部输入做认知门控：余弦匹配主干 + 预测误差繁衍。"""

    def __init__(
        self,
        tree: DynamicCognitiveTree,
        *,
        similarity_threshold: float = 0.85,
        encoder: MultimodalEncoder | None = None,
    ) -> None:
        self.tree = tree
        self.similarity_threshold = similarity_threshold
        self.encoder = encoder or MultimodalEncoder(text_embedder=tree.embedder)

    def process(self, text: str) -> GateResult:
        parsed = parse_input(text)
        query_label = parsed.trunk or text.strip()
        if not query_label:
            return GateResult(
                action="no_op",
                trunk=None,
                message="空输入",
                parsed=parsed,
                input_modality="text",
            )

        query_vec = self.encoder.encode_text(query_label)
        matched_trunk, sim = self.tree.match_trunk_by_vector(
            query_vec, threshold=self.similarity_threshold
        )

        # 向量未命中时，若词典解析出主干名则作为新/待建概念
        trunk = matched_trunk or parsed.trunk
        if trunk is None:
            return GateResult(
                action="no_op",
                trunk=None,
                message="未识别到已知主干概念（向量与词典均未命中）",
                parsed=parsed,
                cosine_similarity=sim,
                input_modality="text",
            )

        features = list(parsed.features)
        trunk_exists = self.tree.get_node(trunk) is not None

        if not trunk_exists:
            self.tree.add_trunk(trunk)
            spawned: list[str] = []
            if features:
                spawned = self._spawn_new(trunk, features)
                return GateResult(
                    action="spawned_children" if spawned else "spawned_trunk",
                    trunk=trunk,
                    prediction_error=bool(spawned),
                    spawned=spawned,
                    message=f"新建主干「{trunk}」" + (f"并繁衍 {spawned}" if spawned else ""),
                    parsed=parsed,
                    cosine_similarity=sim,
                    input_modality="text",
                )
            return GateResult(
                action="spawned_trunk",
                trunk=trunk,
                prediction_error=False,
                spawned=[],
                message=f"新建孤立主干「{trunk}」（embedding 已入库）",
                parsed=parsed,
                cosine_similarity=sim,
                input_modality="text",
            )

        # 主干已存在：相似度 ≥ τ → 可激活；新维度 → 繁衍
        self.tree.activate(trunk)

        if not features:
            return GateResult(
                action="routine_activate",
                trunk=trunk,
                prediction_error=False,
                message=f"常规输入：相似度 {sim:.3f}，仅激活主干「{trunk}」",
                parsed=parsed,
                cosine_similarity=sim,
                input_modality="text",
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
                message=f"特征已存在：相似度 {sim:.3f}，强化子节点",
                parsed=parsed,
                cosine_similarity=sim,
                input_modality="text",
            )

        spawned = self._spawn_new(trunk, new_feats)
        return GateResult(
            action="spawned_children",
            trunk=trunk,
            prediction_error=True,
            spawned=spawned,
            message=f"预测误差：相似度 {sim:.3f}，繁衍 {spawned}",
            parsed=parsed,
            cosine_similarity=sim,
            input_modality="text",
        )

    def process_image(self, image_path: str, *, label_hint: str | None = None) -> GateResult:
        """多模态：图像经 CLIP 编码后向量匹配主干（无新维度则仅激活）。"""
        try:
            query_vec = self.encoder.encode_image(image_path)
        except RuntimeError as exc:
            return GateResult(
                action="no_op",
                trunk=None,
                message=str(exc),
                input_modality="image",
            )

        matched_trunk, sim = self.tree.match_trunk_by_vector(
            query_vec, threshold=self.similarity_threshold
        )
        if matched_trunk is None:
            hint = label_hint or "未知概念"
            self.tree.add_trunk(hint)
            self.tree.vector_store.upsert_node(
                hint,
                query_vec,
                concept=hint,
                kind="trunk",
            )
            return GateResult(
                action="spawned_trunk",
                trunk=hint,
                message=f"图像未匹配已有主干（sim={sim:.3f}），新建「{hint}」",
                cosine_similarity=sim,
                input_modality="image",
            )

        self.tree.activate(matched_trunk)
        return GateResult(
            action="routine_activate",
            trunk=matched_trunk,
            message=f"图像匹配主干「{matched_trunk}」（sim={sim:.3f}），仅激活",
            cosine_similarity=sim,
            input_modality="image",
        )

    def _spawn_new(self, trunk: str, features: list[tuple[str, str]]) -> list[str]:
        spawned: list[str] = []
        for dim, val in features:
            if not self.tree.has_feature(trunk, dim, val):
                child_id = self.tree.add_feature_child(trunk, dim, val)
                spawned.append(child_id)
        return spawned
