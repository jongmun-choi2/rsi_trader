"""RSI 복합 매매 전략 (매수 전용).

진입 셋업
- A_ENGULFING      : RSI 과매도 + BB 하단 이탈 이후 N봉 내 상승 장악형 양봉
- A_DOUBLE_BOTTOM  : 첫 저점 BB 이탈, 두 번째 저점 BB 안쪽인 쌍바닥의 넥라인 돌파
- B_DIVERGENCE_MACD: 강세 다이버전스 확인 후 MACD 골든크로스

청산 신호(보유 중일 때만)
- RSI 과매수 + 하락 장악형 음봉
- 약세 다이버전스 + MACD 데드크로스
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum

import numpy as np
import pandas as pd

from rsi_trader.config import IndicatorConfig, StrategyConfig
from rsi_trader.indicators import add_indicators
from rsi_trader.patterns import (
    bearish_engulfing,
    bullish_engulfing,
    find_bearish_divergence,
    find_bullish_divergence,
    find_double_bottom,
    swing_points,
)


class Setup(str, Enum):
    ENGULFING = "A_ENGULFING"
    DOUBLE_BOTTOM = "A_DOUBLE_BOTTOM"
    DIVERGENCE_MACD = "B_DIVERGENCE_MACD"


@dataclass(frozen=True)
class Signal:
    bar_index: int
    setup: Setup
    entry: float
    stop: float
    target: float
    reason: str


_FLAG_COLUMNS = ("bull_engulf", "bear_engulf", "swing_low", "swing_high")


@dataclass(frozen=True)
class MarketArrays:
    """전략 판정에 필요한 컬럼을 numpy 배열로 보관 (봉 단위 반복 시 pandas 오버헤드 제거)."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    rsi: np.ndarray
    bb_lower: np.ndarray
    bb_mid: np.ndarray
    bb_upper: np.ndarray
    macd: np.ndarray
    macd_signal: np.ndarray
    vol_ma: np.ndarray
    bull_engulf: np.ndarray
    bear_engulf: np.ndarray
    swing_low: np.ndarray
    swing_high: np.ndarray
    cloud_top: np.ndarray
    kijun: np.ndarray

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "MarketArrays":
        missing = [f.name for f in fields(cls) if f.name not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼 누락: {missing}")
        return cls(**{
            f.name: df[f.name].to_numpy(dtype=bool if f.name in _FLAG_COLUMNS else float)
            for f in fields(cls)
        })

    def __len__(self) -> int:
        return len(self.close)


def prepare(df: pd.DataFrame, indicators: IndicatorConfig, strategy: StrategyConfig) -> pd.DataFrame:
    """OHLCV → 지표 + 패턴 컬럼이 추가된 DataFrame."""
    out = add_indicators(df, indicators)
    out["bull_engulf"] = bullish_engulfing(out)
    out["bear_engulf"] = bearish_engulfing(out)
    out["swing_low"] = swing_points(out["low"], strategy.pivot_left, strategy.pivot_right, "low")
    out["swing_high"] = swing_points(out["high"], strategy.pivot_left, strategy.pivot_right, "high")
    return out


def _crossed_above(a: np.ndarray, b: np.ndarray, t: int) -> bool:
    return t >= 1 and a[t] > b[t] and a[t - 1] <= b[t - 1]


class RsiStrategy:
    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ 진입
    def entry_signal(self, a: MarketArrays, t: int) -> Signal | None:
        """시점 t 봉 종가 기준 진입 신호. 우선순위: 장악형 → 쌍바닥 → 다이버전스."""
        for check in (self._engulfing, self._double_bottom, self._divergence_macd):
            signal = check(a, t)
            if signal is not None:
                return signal
        return None

    def _build(self, setup: Setup, t: int, entry: float, stop_base: float, reason: str) -> Signal | None:
        stop = stop_base * (1 - self.cfg.stop_buffer_pct)
        if not np.isfinite(entry) or not np.isfinite(stop) or entry <= stop:
            return None  # 손절 기준을 정의할 수 없는 자리는 진입하지 않는다
        target = entry + self.cfg.reward_risk * (entry - stop)
        return Signal(t, setup, float(entry), float(stop), float(target), reason)

    def _engulfing(self, a: MarketArrays, t: int) -> Signal | None:
        cfg = self.cfg
        if not a.bull_engulf[t]:
            return None
        for k in range(t - 1, max(-1, t - 1 - cfg.engulf_max_wait), -1):
            if not (a.rsi[k] <= cfg.oversold and a.close[k] < a.bb_lower[k]):
                continue
            surge = a.vol_ma[k] > 0 and a.volume[k] >= a.vol_ma[k] * cfg.volume_surge_ratio
            if cfg.require_volume_surge and not surge:
                continue
            reason = f"RSI {a.rsi[k]:.1f} 과매도 + BB 하단 이탈 후 상승 장악형 양봉"
            if surge:
                reason += " (거래량 급증)"
            return self._build(Setup.ENGULFING, t, a.close[t], float(np.min(a.low[k:t + 1])), reason)
        return None

    def _double_bottom(self, a: MarketArrays, t: int) -> Signal | None:
        pattern = find_double_bottom(a.low, a.high, a.rsi, a.bb_lower, a.swing_low, t, self.cfg)
        if pattern is None or t < 1:
            return None
        if not (a.close[t] > pattern.neckline >= a.close[t - 1]):
            return None
        stop_base = min(a.low[pattern.first], a.low[pattern.second])
        reason = f"쌍바닥(두 번째 저점 BB 안쪽) 넥라인 {pattern.neckline:.2f} 돌파"
        return self._build(Setup.DOUBLE_BOTTOM, t, a.close[t], stop_base, reason)

    def _divergence_macd(self, a: MarketArrays, t: int) -> Signal | None:
        if not _crossed_above(a.macd, a.macd_signal, t):
            return None
        div = find_bullish_divergence(a.low, a.rsi, a.swing_low, t, self.cfg)
        if div is None or t - div.second > self.cfg.macd_max_wait:
            return None
        reason = (
            f"강세 다이버전스(RSI {a.rsi[div.first]:.1f}→{a.rsi[div.second]:.1f}) 후 MACD 골든크로스"
        )
        return self._build(Setup.DIVERGENCE_MACD, t, a.close[t], a.low[div.second], reason)

    # ------------------------------------------------------------------ 청산
    def exit_signal(self, a: MarketArrays, t: int) -> str | None:
        """보유 포지션의 지표 기반 청산 사유 (손절/익절 가격 청산과 별개)."""
        cfg = self.cfg
        if not cfg.use_signal_exit:
            return None
        if a.rsi[t] >= cfg.overbought and a.bear_engulf[t]:
            return "RSI 과매수 + 하락 장악형 음봉"
        if _crossed_above(a.macd_signal, a.macd, t):
            div = find_bearish_divergence(a.high, a.rsi, a.swing_high, t, cfg)
            if div is not None and t - div.second <= cfg.macd_max_wait:
                return "약세 다이버전스 + MACD 데드크로스"
        return None
