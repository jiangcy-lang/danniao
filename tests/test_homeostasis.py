"""Step 5a 验收测试：内稳态驱动力系统。

验证驱动力的事件驱动更新和自然衰减。
所有测试无需外部依赖。
"""

from __future__ import annotations

import time

from danniao.motivation import (
    Drive,
    Homeostasis,
    HomeostasisConfig,
    InternalState,
)


# ==================== 初始状态 ====================


def test_initial_state():
    """初始状态：好奇心中等，置信度低，能量满，满足感低。"""
    homeo = Homeostasis()
    s = homeo.snapshot()

    assert 0.4 <= s.curiosity <= 0.6   # 0.5 基线
    assert 0.2 <= s.confidence <= 0.4  # 0.3 基线
    assert s.energy == pytest_approx(1.0)
    assert s.satiety <= 0.3            # 0.2 基线


def pytest_approx(val, rel=0.05):
    """简单近似匹配，避免 pytest.approx 的时间漂移。"""
    class _Approx:
        def __init__(self, v, r):
            self.v = v
            self.r = r
        def __eq__(self, other):
            return abs(other - self.v) <= self.v * self.r
        def __repr__(self):
            return f"~{self.v}"
    return _Approx(val, rel)


# ==================== 预测误差 → 好奇心 ====================


def test_prediction_error_raises_curiosity():
    """高预测误差 → 好奇心上升。"""
    homeo = Homeostasis()
    before = homeo.state.curiosity

    homeo.observe_prediction_error(0.9)

    after = homeo.state.curiosity
    assert after > before
    assert after > 0.5  # 显著上升


def test_low_prediction_error_small_boost():
    """低预测误差 → 好奇心小幅上升。"""
    homeo = Homeostasis()
    before = homeo.state.curiosity

    homeo.observe_prediction_error(0.1)

    after = homeo.state.curiosity
    assert after > before  # 仍上升
    boost = after - before
    assert boost < 0.05    # 但幅度小


def test_activity_consumes_energy():
    """每次处理都消耗能量。"""
    homeo = Homeostasis()
    before = homeo.state.energy

    homeo.observe_prediction_error(0.5)

    after = homeo.state.energy
    assert after < before


# ==================== 匹配 → 置信度 ====================


def test_good_match_raises_confidence():
    """高相似度匹配 → 置信度上升。"""
    homeo = Homeostasis()
    before = homeo.state.confidence

    homeo.observe_match(0.95)

    after = homeo.state.confidence
    assert after > before


def test_poor_match_small_confidence_boost():
    """低相似度匹配 → 置信度小幅上升。"""
    homeo = Homeostasis()
    before = homeo.state.confidence

    homeo.observe_match(0.3)

    after = homeo.state.confidence
    assert after > before
    assert (after - before) < 0.05


# ==================== 交付 → 满足感 ====================


def test_successful_delivery_raises_satiety():
    """成功交付 → 满足感上升，好奇心下降。"""
    homeo = Homeostasis()
    before_satiety = homeo.state.satiety
    before_curiosity = homeo.state.curiosity

    homeo.observe_delivery(True)

    assert homeo.state.satiety > before_satiety
    assert homeo.state.curiosity <= before_curiosity


def test_failed_delivery_lowers_satiety():
    """失败交付 → 满足感下降，好奇心上升。"""
    homeo = Homeostasis()
    before_satiety = homeo.state.satiety
    before_curiosity = homeo.state.curiosity

    homeo.observe_delivery(False)

    assert homeo.state.satiety < before_satiety
    assert homeo.state.curiosity > before_curiosity


# ==================== 衰减 ====================


def test_decay_reduces_drives():
    """衰减使驱动力向基线回归。"""
    config = HomeostasisConfig(
        curiosity_decay=10.0,   # 极快衰减（测试用）
        confidence_decay=10.0,
        satiety_decay=10.0,
        energy_recovery=10.0,
    )
    homeo = Homeostasis(config=config)

    # 先拉高驱动力
    homeo.observe_prediction_error(1.0)
    homeo.observe_match(1.0)
    homeo.observe_delivery(True)

    high_curiosity = homeo.state.curiosity
    high_satiety = homeo.state.satiety

    # 等一小段时间让衰减生效
    time.sleep(0.1)
    homeo.decay()

    assert homeo.state.curiosity < high_curiosity
    assert homeo.state.satiety < high_satiety


def test_decay_recovers_energy():
    """衰减使能量恢复。"""
    config = HomeostasisConfig(energy_recovery=10.0)
    homeo = Homeostasis(config=config)

    # 消耗能量
    for _ in range(20):
        homeo.observe_prediction_error(0.5)

    low_energy = homeo.state.energy
    assert low_energy < 1.0

    # 等待恢复
    time.sleep(0.1)
    homeo.decay()

    assert homeo.state.energy > low_energy


def test_decay_to_baseline():
    """驱动力衰减到基线后不再继续下降。"""
    config = HomeostasisConfig(
        curiosity_decay=100.0,
        curiosity_baseline=0.3,
    )
    homeo = Homeostasis(config=config)
    homeo.observe_prediction_error(1.0)  # 拉高好奇心

    time.sleep(0.05)
    homeo.decay()

    assert homeo.state.curiosity >= 0.3 - 0.01  # 不低于基线


# ==================== 主驱动力 ====================


def test_dominant_drive_curiosity():
    """好奇心高时，主驱动力是 CURIOSITY。"""
    homeo = Homeostasis()
    # 拉高好奇心
    for _ in range(5):
        homeo.observe_prediction_error(1.0)

    assert homeo.dominant_drive() == Drive.CURIOSITY


def test_dominant_drive_satiety():
    """满足感高时，主驱动力是 SATIETY。"""
    homeo = Homeostasis()
    for _ in range(5):
        homeo.observe_delivery(True)

    assert homeo.state.satiety > 0.7
    assert homeo.dominant_drive() == Drive.SATIETY


def test_dominant_drive_energy():
    """能量低时，主驱动力是 ENERGY。"""
    config = HomeostasisConfig(energy_cost_per_activity=0.1)
    homeo = Homeostasis(config=config)

    # 大量消耗能量
    for _ in range(20):
        homeo.observe_prediction_error(0.1)

    assert homeo.state.energy < 0.2
    assert homeo.dominant_drive() == Drive.ENERGY


# ==================== 探索意愿 ====================


def test_wants_to_explore_after_high_error():
    """高预测误差后，系统想探索。"""
    homeo = Homeostasis()
    homeo.observe_prediction_error(1.0)

    assert homeo.wants_to_explore() is True


def test_does_not_explore_when_satiated():
    """满足感高时，系统不想探索。"""
    homeo = Homeostasis()
    for _ in range(5):
        homeo.observe_delivery(True)

    assert homeo.state.satiety > 0.6
    assert homeo.wants_to_explore() is False


def test_does_not_explore_when_low_energy():
    """能量低时，系统不想探索。"""
    config = HomeostasisConfig(energy_cost_per_activity=0.1)
    homeo = Homeostasis(config=config)

    for _ in range(20):
        homeo.observe_prediction_error(0.1)

    assert homeo.state.energy < 0.3
    assert homeo.wants_to_explore() is False


# ==================== 快照 ====================


def test_snapshot_is_copy():
    """快照是副本，修改不影响原状态。"""
    homeo = Homeostasis()
    snap = homeo.snapshot()

    snap.curiosity = 0.99
    assert homeo.state.curiosity != 0.99
