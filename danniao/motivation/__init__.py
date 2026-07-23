"""丹鸟内稳态：驱动力系统（Step 5a）。

人脑的驱动力来自内稳态——饥饿、好奇、疲倦、满足。
丹鸟同理：curiosity 驱动探索，satiety 标记满足，
confidence 反映理解程度，energy 约束活动强度。

驱动力不是 tick 驱动的——它们由事件触发更新（观测到预测误差、交付结果），
并自然衰减（像生物驱动力一样随时间淡化）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time


class Drive(Enum):
    """驱动力类型。"""

    CURIOSITY = "curiosity"
    CONFIDENCE = "confidence"
    ENERGY = "energy"
    SATIETY = "satiety"


@dataclass
class InternalState:
    """内稳态状态快照。所有值 0~1。"""

    curiosity: float = 0.5
    """好奇心：高 = 想探索新事物。由预测误差驱动。"""

    confidence: float = 0.3
    """置信度：对当前认知的理解程度。由匹配相似度驱动。"""

    energy: float = 1.0
    """能量：活动预算。随活动消耗，随休息恢复。"""

    satiety: float = 0.2
    """满足感：交付成功带来的满足。高 = 暂时不想探索。"""

    last_update: float = field(default_factory=time.time)
    """上次更新的时间戳。用于计算衰减间隔。"""

    def copy(self) -> "InternalState":
        return InternalState(
            curiosity=self.curiosity,
            confidence=self.confidence,
            energy=self.energy,
            satiety=self.satiety,
            last_update=self.last_update,
        )


@dataclass
class HomeostasisConfig:
    """内稳态配置参数。"""

    # 衰减速率（每秒）
    curiosity_decay: float = 0.02
    """好奇心衰减：每秒下降 2%。"""

    confidence_decay: float = 0.01
    """置信度衰减：每秒下降 1%。"""

    energy_recovery: float = 0.05
    """能量恢复：每秒恢复 5%（无活动时）。"""

    satiety_decay: float = 0.03
    """满足感衰减：每秒下降 3%。"""

    # 基线
    curiosity_baseline: float = 0.3
    """好奇心衰减到的基线。"""

    confidence_baseline: float = 0.2
    """置信度衰减到的基线。"""

    satiety_baseline: float = 0.1
    """满足感衰减到的基线。"""

    # 观测影响
    curiosity_boost_per_error: float = 0.15
    """每次高预测误差提升的好奇心。"""

    confidence_boost_per_match: float = 0.1
    """每次良好匹配提升的置信度。"""

    energy_cost_per_activity: float = 0.02
    """每次活动消耗的能量。"""

    satiety_boost_per_success: float = 0.2
    """每次成功交付提升的满足感。"""


class Homeostasis:
    """内稳态引擎：管理丹鸟的内部驱动力。

    不是 tick 驱动 —— 驱动力由事件观测更新，随时间自然衰减。
    像生物的内稳态一样：饿了想吃，饱了不想，好奇了想探索。

    用法::

        homeo = Homeostasis()
        homeo.observe_prediction_error(0.8)  # 高误差 → 好奇心上升
        homeo.observe_match(0.95)            # 良好匹配 → 置信度上升
        homeo.observe_delivery(True)         # 交付成功 → 满足感上升
        homeo.decay()                        # 自然衰减

        drive = homeo.dominant_drive()       # 当前最强驱动力
        state = homeo.snapshot()             # 状态快照
    """

    def __init__(
        self,
        *,
        config: HomeostasisConfig | None = None,
    ) -> None:
        self.config = config or HomeostasisConfig()
        self.state = InternalState()

    def observe_prediction_error(self, error: float) -> None:
        """观测到预测误差 → 更新好奇心和能量。

        高误差 = 遇到未知 → 好奇心上升。
        任何处理都消耗少量能量。

        Args:
            error: 预测误差值（0~1）
        """
        cfg = self.config
        self.state.curiosity = min(
            1.0,
            self.state.curiosity + error * cfg.curiosity_boost_per_error,
        )
        self.state.energy = max(
            0.0,
            self.state.energy - cfg.energy_cost_per_activity,
        )
        self.state.last_update = time.time()

    def observe_match(self, similarity: float) -> None:
        """观测到匹配结果 → 更新置信度。

        高相似度 = 理解正确 → 置信度上升。

        Args:
            similarity: 匹配相似度（0~1）
        """
        cfg = self.config
        boost = similarity * cfg.confidence_boost_per_match
        self.state.confidence = min(1.0, self.state.confidence + boost)
        self.state.last_update = time.time()

    def observe_delivery(self, success: bool) -> None:
        """观测到交付结果 → 更新满足感。

        成功交付 = 任务完成 → 满足感上升，好奇心暂时下降。
        失败交付 = 需要改进 → 满足感下降，好奇心上升。

        Args:
            success: 是否成功交付
        """
        cfg = self.config
        if success:
            self.state.satiety = min(
                1.0,
                self.state.satiety + cfg.satiety_boost_per_success,
            )
            # 满足时好奇心略微下降
            self.state.curiosity = max(
                0.0,
                self.state.curiosity - 0.05,
            )
        else:
            self.state.satiety = max(0.0, self.state.satiety - 0.1)
            # 失败时好奇心上升（想知道为什么失败）
            self.state.curiosity = min(
                1.0,
                self.state.curiosity + 0.1,
            )
        self.state.last_update = time.time()

    def decay(self) -> None:
        """自然衰减：所有驱动力随时间向基线回归。

        好奇心、置信度、满足感向基线衰减；
        能量向 1.0 恢复。
        """
        cfg = self.config
        now = time.time()
        elapsed = now - self.state.last_update

        if elapsed <= 0:
            return

        # 好奇心衰减
        self.state.curiosity = max(
            cfg.curiosity_baseline,
            self.state.curiosity - cfg.curiosity_decay * elapsed,
        )

        # 置信度衰减
        self.state.confidence = max(
            cfg.confidence_baseline,
            self.state.confidence - cfg.confidence_decay * elapsed,
        )

        # 满足感衰减
        self.state.satiety = max(
            cfg.satiety_baseline,
            self.state.satiety - cfg.satiety_decay * elapsed,
        )

        # 能量恢复
        self.state.energy = min(
            1.0,
            self.state.energy + cfg.energy_recovery * elapsed,
        )

        self.state.last_update = now

    def dominant_drive(self) -> Drive:
        """返回当前最强的驱动力。

        满足感高时优先返回 SATIETY（不想动）；
        否则返回其他中最强的。

        Returns:
            Drive 枚举值
        """
        s = self.state

        # 满足感很高时，系统倾向"休息"
        if s.satiety > 0.7:
            return Drive.SATIETY

        # 能量很低时，系统需要"恢复"
        if s.energy < 0.2:
            return Drive.ENERGY

        # 否则取最强的驱动力
        drives = {
            Drive.CURIOSITY: s.curiosity,
            Drive.CONFIDENCE: s.confidence,
        }
        return max(drives, key=drives.get)

    def wants_to_explore(self) -> bool:
        """系统当前是否想探索？

        好奇心高且能量充足且不太满足时 = 想探索。

        Returns:
            bool
        """
        s = self.state
        return (
            s.curiosity > 0.5
            and s.energy > 0.3
            and s.satiety < 0.6
        )

    def snapshot(self) -> InternalState:
        """返回当前状态的快照副本。"""
        self.decay()
        return self.state.copy()


# 奖励系统（Step 6A）
from danniao.motivation.reward import RewardResult, RewardSystem  # noqa: E402

# 探索引擎（Step 6C）
from danniao.motivation.exploration import ExplorationEngine, ExplorationTarget  # noqa: E402

__all__ = [
    "Drive",
    "Homeostasis",
    "HomeostasisConfig",
    "InternalState",
    "RewardResult",
    "RewardSystem",
    "ExplorationEngine",
    "ExplorationTarget",
]
