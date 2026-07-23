"""丹鸟持续心智：一直睁眼看世界（Step 5b）。

人脑从出生就没有停止过工作。丹鸟也一样。

没有 tick，没有 sleep。四条流并行持续运行：
- 感知流：输入到达即处理，无输入时做内部活动
- 动力学流：权重衰减持续运行
- 扩散流：激活扩散持续运行
- 好奇心流：内稳态持续评估

每条流有自己的自然节律，系统整体永不休眠。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from danniao.expression import ExpressionContext, ExpressionEngine
from danniao.motivation import (
    Drive,
    ExplorationEngine,
    ExplorationTarget,
    Homeostasis,
    InternalState,
    RewardResult,
    RewardSystem,
)

if TYPE_CHECKING:
    from danniao.actions.knowledge_ingestion import IngestionReport, KnowledgeIngestion
    from danniao.actions.world_interface import WorldInterface
    from danniao.hippocampus.dynamics import NeuroDynamicsEngine
    from danniao.hippocampus.episodic_log import EpisodicLog
    from danniao.hippocampus.gate import GateResult, InformationTriggerGate
    from danniao.hippocampus.spreading import SpreadingActivation
    from danniao.hippocampus.tree import DynamicCognitiveTree

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """单次感知-处理结果。"""

    text: str
    """输入文本。"""

    matched_trunk: str | None = None
    """匹配到的主干标签。"""

    is_new_trunk: bool = False
    """是否是新概念。"""

    prediction_error: float = 0.0
    """预测误差（0~1）。"""

    trunk_node_id: str | None = None
    """主干节点 ID。"""

    activated_nodes: dict[str, float] = field(default_factory=dict)
    """扩散激活的节点 {node_id: activation_level}。"""

    internal_state: InternalState | None = None
    """处理后的内稳态快照。"""

    node_count: int = 0
    """认知树总节点数。"""

    trunk_count: int = 0
    """主干数量。"""

    expression: str = ""
    """丹鸟的语言表达（Step 6B）。"""

    exploration: ExplorationTarget | None = None
    """探索目标（Step 6C），None 表示当前不想探索。"""


@dataclass
class MindStatus:
    """心智状态快照。"""

    internal_state: InternalState
    """内稳态状态。"""

    node_count: int
    """认知树总节点数。"""

    trunk_count: int
    """主干数量。"""

    dominant_drive: Drive
    """当前主导驱动力。"""

    wants_to_explore: bool
    """是否想探索。"""

    recent_nodes: list[str]
    """近期节点列表。"""

    is_running: bool
    """是否在持续运行。"""

    last_expression: str = ""
    """丹鸟最近一次的语言表达（Step 6B）。"""


class ContinuousMind:
    """持续心智核心 —— 一直睁眼看世界。

    同步模式（测试 / 简单使用）::

        mind = ContinuousMind(tree, gate, dynamics, spreading, homeostasis, log)
        result = mind.process("苹果")

    异步模式（持续运行）::

        asyncio.run(mind.live())
        # 在另一个协程中：
        await mind.perceive("苹果")

    四条流并行运行，系统整体永不休眠。
    """

    def __init__(
        self,
        tree: DynamicCognitiveTree,
        gate: InformationTriggerGate,
        dynamics: NeuroDynamicsEngine,
        spreading: SpreadingActivation,
        homeostasis: Homeostasis,
        episodic_log: EpisodicLog,
        *,
        expression_engine: ExpressionEngine | None = None,
        exploration_engine: ExplorationEngine | None = None,
        reward_system: RewardSystem | None = None,
    ) -> None:
        """初始化持续心智。

        Args:
            tree: 认知树
            gate: 信息触发门控
            dynamics: 神经动力学引擎
            spreading: 扩散激活引擎
            homeostasis: 内稳态引擎
            episodic_log: 情景记忆日志
            expression_engine: 表达引擎（可选，Step 6B）
            exploration_engine: 探索引擎（可选，Step 6C）
            reward_system: 奖励系统（可选，Step 6A）
        """
        self.tree = tree
        self.gate = gate
        self.dynamics = dynamics
        self.spreading = spreading
        self.homeostasis = homeostasis
        self.episodic_log = episodic_log

        # Step 6 引擎（可选）
        self.expression_engine = expression_engine
        self.exploration_engine = exploration_engine
        self.reward_system = reward_system

        # 运行状态
        self._running = False
        self._input_queue: asyncio.Queue[tuple[str, str]] | None = None

        # 激活种子（感知流产生，扩散流消费）
        self._activation_seeds: list[str] = []

        # 最近处理的节点（用于空闲时回放）
        self._recent_nodes: list[str] = []
        self._max_recent = 20

        # 最近一次处理结果
        self._last_result: ProcessResult | None = None

    # ==================== 同步模式 ====================

    def process(self, text: str, source: str = "user") -> ProcessResult:
        """同步处理一次输入。

        完整链路：门控 → 日志 → 内稳态 → 激活 → 扩散 → 强化 → 表达 → 探索。

        Args:
            text: 输入文本
            source: 输入来源（user / search / exploration）

        Returns:
            ProcessResult 处理结果
        """
        # 1. 门控处理
        gate_result: GateResult = self.gate.process(text)

        # 2. 记录到原始日志
        self.episodic_log.append(
            interaction_id=gate_result.timestamp,
            text=text,
            source=source,
            metadata={
                "matched_trunk": gate_result.matched_trunk,
                "prediction_error": gate_result.prediction_error,
                "is_new_trunk": gate_result.is_new_trunk,
            },
        )

        # 3. 更新内稳态
        self.homeostasis.observe_prediction_error(gate_result.prediction_error)
        if gate_result.matched_trunk:
            self.homeostasis.observe_match(gate_result.matched_trunk_similarity)

        # 4. 记录激活种子
        trunk_id = gate_result.trunk_node_id
        if trunk_id:
            self._activation_seeds.append(trunk_id)
            self._recent_nodes.append(trunk_id)
            if len(self._recent_nodes) > self._max_recent:
                self._recent_nodes.pop(0)

        # 5. 扩散激活
        activated: dict[str, float] = {}
        if self._activation_seeds:
            activated = self.spreading.spread(self._activation_seeds)
            self._activation_seeds.clear()

        # 6. 动力学强化活跃路径
        if trunk_id:
            self._reinforce_active_paths(trunk_id)

        # 7. 内稳态快照（用于表达和结果）
        state = self.homeostasis.snapshot()

        # 8. 表达（Step 6B）
        expression_text = ""
        if self.expression_engine:
            expr_ctx = ExpressionContext(
                input_text=text,
                matched_trunk=gate_result.matched_trunk,
                is_new_trunk=gate_result.is_new_trunk,
                prediction_error=gate_result.prediction_error,
                activated_nodes=activated,
                curiosity=state.curiosity,
                confidence=state.confidence,
                energy=state.energy,
                satiety=state.satiety,
            )
            expr_result = self.expression_engine.express(expr_ctx)
            expression_text = expr_result.text

        # 9. 探索（Step 6C）
        exploration_target: ExplorationTarget | None = None
        if self.exploration_engine:
            exploration_target = self.exploration_engine.propose(
                self._recent_nodes
            )

        # 10. 构建结果
        result = ProcessResult(
            text=text,
            matched_trunk=gate_result.matched_trunk,
            is_new_trunk=gate_result.is_new_trunk,
            prediction_error=gate_result.prediction_error,
            trunk_node_id=trunk_id,
            activated_nodes=activated,
            internal_state=state,
            node_count=self.tree.graph.number_of_nodes(),
            trunk_count=self.tree.trunk_count(),
            expression=expression_text,
            exploration=exploration_target,
        )
        self._last_result = result
        return result

    def _reinforce_active_paths(self, trunk_id: str) -> None:
        """Hebbian 强化活跃路径。"""
        for child_id in self.tree.graph.successors(trunk_id):
            self.dynamics.hebbian_reinforce(trunk_id, child_id)

    # ==================== 异步模式 ====================

    async def perceive(self, text: str, source: str = "user") -> None:
        """异步感知：立即入队，不阻塞。

        输入会被感知流在下一个循环中处理。

        Args:
            text: 输入文本
            source: 输入来源
        """
        if self._input_queue is None:
            self._input_queue = asyncio.Queue()
        await self._input_queue.put((text, source))

    async def live(self) -> None:
        """启动持续心智。永不停止，直到 stop()。

        四条流并行运行：
        - 感知流：处理输入，无输入时做内部活动
        - 动力学流：权重衰减
        - 扩散流：激活扩散
        - 好奇心流：内稳态评估
        """
        self._running = True
        self._input_queue = asyncio.Queue()

        await asyncio.gather(
            self._perception_stream(),
            self._dynamics_stream(),
            self._spreading_stream(),
            self._curiosity_stream(),
        )

    def stop(self) -> None:
        """停止持续心智。"""
        self._running = False

    async def _perception_stream(self) -> None:
        """感知流：有输入就处理，没输入就做内部活动。

        内部活动 = 回放最近节点（记忆排练的雏形）。
        """
        while self._running:
            assert self._input_queue is not None
            try:
                text, source = await asyncio.wait_for(
                    self._input_queue.get(), timeout=0.5
                )
                self.process(text, source)
            except asyncio.TimeoutError:
                # 无输入 —— 系统仍在运行
                # 如果好奇心高，回放最近节点（记忆排练雏形）
                if self._recent_nodes and self.homeostasis.wants_to_explore():
                    self._idle_rehearsal()

    def _idle_rehearsal(self) -> None:
        """空闲排练：回放最近节点，保持激活。

        这是记忆巩固（Step 7）的雏形。
        空闲时重新激活最近的节点，防止它们过快衰减。
        """
        if not self._recent_nodes:
            return

        # 取最近的几个节点重新激活
        recent = self._recent_nodes[-5:]
        for nid in recent:
            if nid in self.tree.graph:
                self.tree.activate(nid)
                self._activation_seeds.append(nid)

    async def _dynamics_stream(self) -> None:
        """动力学流：突触衰减持续运行。

        节律：每 1 秒一次。这不是 tick —— 是突触衰减的自然节律。
        """
        while self._running:
            try:
                self.dynamics.apply_decay()
            except Exception:
                pass
            await asyncio.sleep(1.0)

    async def _spreading_stream(self) -> None:
        """扩散流：激活扩散持续运行。

        节律：每 0.5 秒一次。只要有激活种子就扩散。
        """
        while self._running:
            if self._activation_seeds:
                seeds = self._activation_seeds.copy()
                self._activation_seeds.clear()
                try:
                    self.spreading.spread(seeds)
                except Exception:
                    pass
            await asyncio.sleep(0.5)

    async def _curiosity_stream(self) -> None:
        """好奇心流：内稳态持续评估。

        节律：每 2 秒一次。评估驱动力，判断是否想探索。
        """
        while self._running:
            self.homeostasis.decay()
            await asyncio.sleep(2.0)

    # ==================== 状态查询 ====================

    def status(self) -> MindStatus:
        """返回当前心智状态快照。"""
        state = self.homeostasis.snapshot()
        last_expr = ""
        if self._last_result:
            last_expr = self._last_result.expression
        return MindStatus(
            internal_state=state,
            node_count=self.tree.graph.number_of_nodes(),
            trunk_count=self.tree.trunk_count(),
            dominant_drive=self.homeostasis.dominant_drive(),
            wants_to_explore=self.homeostasis.wants_to_explore(),
            recent_nodes=list(self._recent_nodes),
            is_running=self._running,
            last_expression=last_expr,
        )

    def give_feedback(self, success: bool) -> RewardResult | None:
        """给丹鸟的最近一次表达提供反馈（Step 6A 奖励闭环）。

        成功 → 满足感上升 + 活跃路径强化（Hebbian LTP）
        失败 → 好奇心上升 + 活跃路径弱化（LTD）

        Args:
            success: 表达是否被认可

        Returns:
            RewardResult 或 None（无奖励系统或无最近结果）
        """
        if not self.reward_system or not self._last_result:
            return None

        # 构建活跃路径：trunk + top activated nodes
        active_path: list[str] = []
        if self._last_result.trunk_node_id:
            active_path.append(self._last_result.trunk_node_id)

        sorted_activated = sorted(
            self._last_result.activated_nodes.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for nid, _ in sorted_activated[:4]:
            if nid not in active_path:
                active_path.append(nid)

        return self.reward_system.observe_delivery(success, active_path)

    @property
    def last_result(self) -> ProcessResult | None:
        """最近一次处理结果。"""
        return self._last_result
