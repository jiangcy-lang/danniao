"""
海马体动态认知树验收脚本（造物主标准）

1. 输入「苹果」→ 仅孤立主干
2. 输入「红色的甜苹果」→ 繁衍 颜色-红、味道-甜
3. 再输入「苹果」→ 不新增子节点
4. describe → 最小表达输出
"""

from __future__ import annotations

from danniao.hippocampus import DynamicCognitiveTree, InformationTriggerGate


def run_acceptance() -> None:
    tree = DynamicCognitiveTree()
    gate = InformationTriggerGate(tree)

    # --- Step A: 仅告知苹果 ---
    r1 = gate.process("苹果")
    assert r1.action == "spawned_trunk", r1
    assert tree.trunk_count() == 1
    assert tree.get_node("苹果") is not None
    assert tree.get_children("苹果") == []
    assert tree.feature_count("苹果") == 0
    print("[OK] 输入「苹果」→ 仅孤立主干节点")

    # --- Step B: 高信息量 ---
    r2 = gate.process("红色的甜苹果")
    assert r2.prediction_error is True, r2
    assert r2.action == "spawned_children", r2
    children = {c["concept"] for c in tree.get_children("苹果")}
    assert children == {"颜色-红", "味道-甜"}, children
    assert tree.has_feature("苹果", "颜色", "红")
    assert tree.has_feature("苹果", "味道", "甜")
    assert tree.trunk_count() == 1
    print("[OK] 输入「红色的甜苹果」→ 繁衍 颜色-红、味道-甜")

    # --- Step C: 常规再提及 ---
    before = tree.feature_count("苹果")
    r3 = gate.process("苹果")
    assert r3.action == "routine_activate", r3
    assert tree.feature_count("苹果") == before
    print("[OK] 再次输入「苹果」→ 仅激活，不繁衍")

    # --- Step D: 最小表达 ---
    desc = tree.describe("苹果")
    assert "苹果" in desc
    assert "红" in desc and "甜" in desc
    print(f"[OK] describe 输出: {desc}")

    print("\n✅ 验收全部通过：动态认知树 + 信息门控 + 最小表达")


if __name__ == "__main__":
    run_acceptance()


def test_cognitive_tree_acceptance() -> None:
    run_acceptance()
