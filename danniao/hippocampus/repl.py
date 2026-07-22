"""交互式海马体：输入一句话，看门控结果与 describe 输出。"""

from __future__ import annotations

import sys

from danniao.hippocampus import DynamicCognitiveTree, InformationTriggerGate


def format_result(gate_result) -> str:
    lines = [
        f"  动作: {gate_result.action}",
        f"  预测误差: {gate_result.prediction_error}",
    ]
    if gate_result.spawned:
        lines.append(f"  新节点: {', '.join(gate_result.spawned)}")
    if gate_result.message:
        lines.append(f"  说明: {gate_result.message}")
    return "\n".join(lines)


def run_interactive(*, persist_path: str | None = None) -> None:
    tree = DynamicCognitiveTree()
    if persist_path:
        try:
            tree = DynamicCognitiveTree.load_json(persist_path)
            print(f"[已加载] {persist_path}")
        except FileNotFoundError:
            print(f"[新建] 将保存到 {persist_path}")

    gate = InformationTriggerGate(tree)

    print("=" * 50)
    print("丹鸟 DanNiao — 海马体交互")
    print("输入自然语言；空行退出。命令: tree | save")
    print("=" * 50)

    while True:
        try:
            text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not text:
            print("再见。")
            break

        if text.lower() == "tree":
            for node, data in tree.graph.nodes(data=True):
                kind = data.get("kind", "?")
                print(f"  [{kind}] {node}")
            for u, v, d in tree.graph.edges(data=True):
                w = d.get("weight", 0)
                print(f"  {u} --({w})--> {v}")
            continue

        if text.lower() == "save" and persist_path:
            tree.save_json(persist_path)
            print(f"[已保存] {persist_path}")
            continue

        result = gate.process(text)
        print("\n[门控]")
        print(format_result(result))

        if result.trunk:
            print("\n[丹鸟说]")
            print(tree.describe(result.trunk))

        if persist_path:
            tree.save_json(persist_path)


def run_demo() -> None:
    """固定演示：苹果 → 红色的甜苹果 → 苹果（可见输入输出）。"""
    tree = DynamicCognitiveTree()
    gate = InformationTriggerGate(tree)
    steps = ["苹果", "红色的甜苹果", "苹果"]

    print("=" * 50)
    print("丹鸟 DanNiao — 海马体演示（固定用例）")
    print("=" * 50)

    for text in steps:
        print(f"\n你> {text}")
        result = gate.process(text)
        print("\n[门控]")
        print(format_result(result))
        if result.trunk:
            print("\n[丹鸟说]")
            print(tree.describe(result.trunk))

    print("\n" + "=" * 50)
    print("演示结束。交互模式: python -m danniao.hippocampus.repl")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        path = sys.argv[1] if len(sys.argv) > 1 else None
        run_interactive(persist_path=path)
