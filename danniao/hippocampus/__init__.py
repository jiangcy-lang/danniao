"""海马体：动态认知树与信息门控（记忆底座，非整脑）。"""

from danniao.hippocampus.embeddings import MultimodalEncoder
from danniao.hippocampus.gate import GateResult, InformationTriggerGate
from danniao.hippocampus.tree import DynamicCognitiveTree

__all__ = [
    "DynamicCognitiveTree",
    "InformationTriggerGate",
    "GateResult",
    "MultimodalEncoder",
]
