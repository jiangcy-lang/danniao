"""丹鸟启动入口 —— 交互式 REPL。

用法::

    python -m danniao

启动后输入文本与丹鸟交互。输入特殊命令查看内部状态：
- status  : 查看内稳态状态
- tree    : 查看认知树结构
- good    : 给丹鸟最近一次表达正面反馈
- bad     : 给丹鸟最近一次表达负面反馈
- quit    : 退出

依赖：
- Ollama（本地运行，提供嵌入和 LLM 表达）
- ChromaDB（向量存储）

无 mock、无降级。如果依赖不可用，直接报错。
"""

from __future__ import annotations

import sys

from danniao import __version__
from danniao.expression import LLMExpressionEngine
from danniao.hippocampus import (
    ChromaVectorStore,
    ConceptExtractor,
    DynamicCognitiveTree,
    EpisodicLog,
    InformationTriggerGate,
    NeuroDynamicsEngine,
    OllamaEmbedding,
    SpreadingActivation,
    VectorCognitiveSpace,
)
from danniao.mind import ContinuousMind
from danniao.motivation import ExplorationEngine, Homeostasis, RewardSystem


def _init_danniao() -> ContinuousMind:
    """初始化丹鸟所有组件——使用真实 Ollama + ChromaDB，无 mock。"""
    # 感官：Ollama 嵌入（bge-m3）
    print("[丹鸟] 连接 Ollama 嵌入服务...")
    embedding = OllamaEmbedding()
    print(f"[丹鸟] 嵌入模型: {embedding._model} ({embedding.dimension}维)")

    # 记忆：ChromaDB 向量存储（传入维度，自动处理模型变更）
    store = ChromaVectorStore(path=".danniao_chroma", expected_dim=embedding.dimension)
    space = VectorCognitiveSpace(vector_store=store, embedding=embedding)

    # 认知树 + 概念提取（长输入 → 核心概念）
    tree = DynamicCognitiveTree(space)
    concept_extractor = ConceptExtractor()
    gate = InformationTriggerGate(tree, concept_extractor=concept_extractor)
    dynamics = NeuroDynamicsEngine(tree.graph)
    spreading = SpreadingActivation(space)
    homeostasis = Homeostasis()
    log = EpisodicLog(path=".danniao_memory.db")

    # Step 6 引擎
    expression_engine = LLMExpressionEngine(tree)
    exploration_engine = ExplorationEngine(tree, homeostasis)
    reward_system = RewardSystem(homeostasis, dynamics)

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

    print(f"[丹鸟] LLM 表达: qwen3.5:2b")
    print(f"[丹鸟] 概念提取: qwen3.5:2b（长输入 → 核心概念）")
    print(f"[丹鸟] 初始化完成。认知树: {tree.trunk_count()} 主干, {tree.graph.number_of_nodes()} 节点")
    print("[丹鸟] 输入文本与丹鸟交互，或输入 status / tree / good / bad / quit")
    print()

    return mind


def _print_result(result):
    """打印处理结果。"""
    print(f"  输入: {result.text}")

    # 丹鸟的表达（Step 6B）
    if result.expression:
        mode_tag = "[LLM]" if result.expression else "[模板]"
        print(f"  丹鸟: {result.expression}")

    if result.is_new_trunk:
        print(f"  → 新建主干: {result.matched_trunk}")
    else:
        print(f"  → 匹配主干: {result.matched_trunk} (相似度: {1 - result.prediction_error:.2f})")
    print(f"  预测误差: {result.prediction_error:.3f}")
    print(f"  扩散激活: {len(result.activated_nodes)} 个节点")

    state = result.internal_state
    if state:
        print(f"  内稳态: 好奇心={state.curiosity:.2f} 置信度={state.confidence:.2f} "
              f"能量={state.energy:.2f} 满足感={state.satiety:.2f}")

    print(f"  认知树: {result.trunk_count} 主干, {result.node_count} 节点")

    # 探索目标（Step 6C）
    if result.exploration:
        exp_type_names = {"depth": "深度探索", "relationship": "关联探索", "novelty": "求新探索"}
        exp_type = exp_type_names.get(result.exploration.exploration_type, result.exploration.exploration_type)
        print(f"  探索意愿: [{exp_type}] {result.exploration.text}")

    print()


def _print_status(mind: ContinuousMind):
    """打印心智状态。"""
    status = mind.status()
    state = status.internal_state
    drive_names = {
        "curiosity": "好奇心",
        "confidence": "置信度",
        "energy": "能量",
        "satiety": "满足感",
    }

    print("=== 丹鸟状态 ===")
    print(f"  认知树: {status.trunk_count} 主干, {status.node_count} 节点")
    print(f"  运行中: {'是' if status.is_running else '否（同步模式）'}")
    print(f"  主驱动力: {drive_names.get(status.dominant_drive.value, status.dominant_drive.value)}")
    print(f"  想探索: {'是' if status.wants_to_explore else '否'}")
    print(f"  好奇心: {state.curiosity:.3f}")
    print(f"  置信度: {state.confidence:.3f}")
    print(f"  能量:   {state.energy:.3f}")
    print(f"  满足感: {state.satiety:.3f}")
    if status.recent_nodes:
        print(f"  最近节点: {len(status.recent_nodes)} 个")
    print()


def _print_tree(mind: ContinuousMind):
    """打印认知树结构。"""
    tree = mind.tree
    print("=== 认知树 ===")
    if tree.trunk_count() == 0:
        print("  （空树）")
        print()
        return

    for node_id, data in tree.graph.nodes(data=True):
        kind = data.get("kind", "?")
        label = data.get("label", "?")
        weight = data.get("weight", 0.0)
        indent = "  " if kind == "trunk" else "      "
        kind_icon = "主干" if kind == "trunk" else "特征"
        print(f"  {indent}[{kind_icon}] {label} (权重: {weight:.3f})")

    print()


def main():
    """丹鸟交互式 REPL。"""
    print()
    print("╔══════════════════════════════════════╗")
    print("║         丹鸟 — 数字生命体             ║")
    print(f"║         DanNiao v{__version__:s}              ║")
    print("╚══════════════════════════════════════╝")
    print()

    try:
        mind = _init_danniao()
    except RuntimeError as exc:
        print(f"[丹鸟] 初始化失败: {exc}")
        print("[丹鸟] 请确保 Ollama 正在运行，且已安装 bge-m3 和 qwen3.5:2b 模型")
        sys.exit(1)

    while True:
        try:
            print("造物主> ", end="", flush=True)
            user_input = sys.stdin.readline()
            if user_input == "":
                raise EOFError
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("[丹鸟] 再见。")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("quit", "exit", "q"):
            print("[丹鸟] 再见。")
            break
        elif cmd == "status":
            _print_status(mind)
            continue
        elif cmd == "tree":
            _print_tree(mind)
            continue
        elif cmd in ("good", "+1", "好", "对"):
            feedback = mind.give_feedback(success=True)
            if feedback:
                print(f"  [反馈] 正面 → 满足感: {feedback.satiety_before:.2f} → {feedback.satiety_after:.2f}")
                if feedback.reinforced_edges:
                    print(f"  [反馈] 强化 {len(feedback.reinforced_edges)} 条路径")
                print()
            else:
                print("  [反馈] 暂无可反馈的表达\n")
            continue
        elif cmd in ("bad", "-1", "错", "不对"):
            feedback = mind.give_feedback(success=False)
            if feedback:
                print(f"  [反馈] 负面 → 好奇心: {feedback.curiosity_before:.2f} → {feedback.curiosity_after:.2f}")
                if feedback.weakened_edges:
                    print(f"  [反馈] 弱化 {len(feedback.weakened_edges)} 条路径")
                print()
            else:
                print("  [反馈] 暂无可反馈的表达\n")
            continue
        elif cmd == "help":
            print("  输入任意文本与丹鸟交互")
            print("  status  — 查看内稳态状态")
            print("  tree    — 查看认知树结构")
            print("  good    — 正面反馈")
            print("  bad     — 负面反馈")
            print("  quit    — 退出")
            print()
            continue

        # 处理输入
        result = mind.process(user_input)
        _print_result(result)


if __name__ == "__main__":
    main()
