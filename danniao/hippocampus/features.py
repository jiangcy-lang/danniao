"""文本 → 主干 + 维度特征（MVP 规则词典，可替换）。"""

from __future__ import annotations

from dataclasses import dataclass

# 特征值 → 维度（较长词优先匹配）
FEATURE_LEXICON: list[tuple[str, str, str]] = [
    # (surface, dimension, value)
    ("红色", "颜色", "红"),
    ("绿色", "颜色", "绿"),
    ("青色", "颜色", "青"),
    ("红", "颜色", "红"),
    ("绿", "颜色", "绿"),
    ("青", "颜色", "青"),
    ("甜", "味道", "甜"),
    ("酸", "味道", "酸"),
    ("脆", "口感", "脆"),
    ("大", "大小", "大"),
    ("小", "大小", "小"),
]

# 已知主干概念（可扩展）
TRUNK_LEXICON: list[str] = ["苹果", "梨"]


@dataclass(frozen=True)
class ParsedInput:
    trunk: str | None
    features: tuple[tuple[str, str], ...]  # (dimension, value)


def parse_input(text: str) -> ParsedInput:
    """从自然语言中抽取主干与特征；无法识别主干则 trunk=None。"""
    trunk: str | None = None
    for name in sorted(TRUNK_LEXICON, key=len, reverse=True):
        if name in text:
            trunk = name
            break

    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    # 较长 surface 优先，避免「红」吃掉「红色」的一部分后重复——按序扫描并用占位消除
    remaining = text
    for surface, dimension, value in sorted(FEATURE_LEXICON, key=lambda x: len(x[0]), reverse=True):
        if surface in remaining:
            key = (dimension, value)
            if key not in seen:
                seen.add(key)
                found.append(key)
            remaining = remaining.replace(surface, " " * len(surface), 1)

    return ParsedInput(trunk=trunk, features=tuple(found))
