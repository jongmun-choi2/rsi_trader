"""설정 로딩 모듈.

config.yaml 을 읽어 불변(frozen) dataclass 로 변환한다.
알 수 없는 키는 오타 방지를 위해 즉시 오류로 처리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ALLOWED_MODES = ("backtest", "paper")  # 실거래(live)는 의도적으로 미지원


@dataclass(frozen=True)
class IndicatorConfig:
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_ma_period: int = 10

    def __post_init__(self) -> None:
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast 는 macd_slow 보다 작아야 합니다")
        for name in ("rsi_period", "bb_period", "macd_signal", "volume_ma_period"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 는 1 이상이어야 합니다")


@dataclass(frozen=True)
class StrategyConfig:
    oversold: float = 30.0
    overbought: float = 70.0
    volume_surge_ratio: float = 1.5
    require_volume_surge: bool = False
    engulf_max_wait: int = 5
    pivot_left: int = 3
    pivot_right: int = 3
    pivot_min_gap: int = 5
    pivot_max_gap: int = 40
    divergence_rsi_ceiling: float = 40.0
    macd_max_wait: int = 10
    double_bottom_tolerance: float = 0.015
    double_bottom_max_wait: int = 10
    stop_buffer_pct: float = 0.002
    reward_risk: float = 2.0
    use_signal_exit: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.oversold < self.overbought < 100:
            raise ValueError("0 < oversold < overbought < 100 이어야 합니다")
        if self.reward_risk <= 0:
            raise ValueError("reward_risk 는 0보다 커야 합니다")
        if self.pivot_min_gap > self.pivot_max_gap:
            raise ValueError("pivot_min_gap 은 pivot_max_gap 이하여야 합니다")


@dataclass(frozen=True)
class RiskConfig:
    initial_equity: float = 10_000_000
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.3
    daily_max_loss_pct: float = 0.03
    max_positions: int = 3

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade <= 0.05:
            raise ValueError("risk_per_trade 는 0 초과 0.05(5%) 이하여야 합니다")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct 는 0 초과 1 이하여야 합니다")
        if self.max_positions < 1:
            raise ValueError("max_positions 는 1 이상이어야 합니다")


@dataclass(frozen=True)
class CostConfig:
    commission_rate: float = 0.00015
    sell_tax_rate: float = 0.0020
    slippage_pct: float = 0.0005


@dataclass(frozen=True)
class AppConfig:
    market: str = "KR"
    timeframe_minutes: int = 15
    symbols: tuple[str, ...] = ("005930",)
    mode: str = "backtest"
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    def __post_init__(self) -> None:
        if self.mode not in ALLOWED_MODES:
            raise ValueError(f"mode 는 {ALLOWED_MODES} 중 하나여야 합니다 (실거래 미지원)")
        if self.market not in ("KR", "US"):
            raise ValueError("market 은 KR 또는 US 여야 합니다")
        if self.timeframe_minutes < 1:
            raise ValueError("timeframe_minutes 는 1 이상이어야 합니다")


_SECTIONS = {
    "indicators": IndicatorConfig,
    "strategy": StrategyConfig,
    "risk": RiskConfig,
    "costs": CostConfig,
}


def _build(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: 알 수 없는 설정 키 {sorted(unknown)}")
    return cls(**data)


def config_from_dict(raw: dict[str, Any]) -> AppConfig:
    raw = dict(raw)
    kwargs: dict[str, Any] = {key: _build(cls, raw.pop(key, None) or {}) for key, cls in _SECTIONS.items()}
    if "symbols" in raw:
        raw["symbols"] = tuple(str(s) for s in raw["symbols"])
    return _build(AppConfig, {**raw, **kwargs})


def load_config(path: str | Path) -> AppConfig:
    text = Path(path).read_text(encoding="utf-8")
    return config_from_dict(yaml.safe_load(text) or {})
