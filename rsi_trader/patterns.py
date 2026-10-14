"""캔들 패턴 · 스윙 피벗 · 다이버전스 · 쌍바닥 탐지.

스윙 피벗은 좌우 봉을 비교해야 확정되므로 `swing_points` 결과에는 미래 `right`개 봉이 반영된다.
따라서 시점 t 에서는 반드시 `confirmed_pivots` 로 `i + right <= t` 인 피벗만 사용해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rsi_trader.config import StrategyConfig


# ---------------------------------------------------------------------------
# 캔들 패턴
# ---------------------------------------------------------------------------
def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """상승 장악형: 직전 음봉의 몸통을 현재 양봉 몸통이 완전히(더 크게) 감싼다."""
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    result = (
        (prev_close < prev_open)
        & (df["close"] > df["open"])
        & (df["open"] <= prev_close)
        & (df["close"] >= prev_open)
        & ((df["close"] - df["open"]) > (prev_open - prev_close))
    )
    return result.fillna(False).astype(bool)


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """하락 장악형: 직전 양봉의 몸통을 현재 음봉 몸통이 완전히(더 크게) 감싼다."""
    prev_open, prev_close = df["open"].shift(1), df["close"].shift(1)
    result = (
        (prev_close > prev_open)
        & (df["close"] < df["open"])
        & (df["open"] >= prev_close)
        & (df["close"] <= prev_open)
        & ((df["open"] - df["close"]) > (prev_close - prev_open))
    )
    return result.fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# 스윙 피벗
# ---------------------------------------------------------------------------
def swing_points(values: pd.Series, left: int, right: int, kind: str) -> pd.Series:
    """스윙 저점(kind='low') / 고점(kind='high') 위치를 True 로 표시한다.

    동일 값이 연속되면 가장 먼저 나온 봉을 피벗으로 본다(왼쪽은 엄격 비교, 오른쪽은 비엄격 비교).
    """
    if kind not in ("low", "high"):
        raise ValueError("kind 는 'low' 또는 'high' 여야 합니다")
    arr = values.to_numpy(dtype=float)
    out = np.zeros(len(arr), dtype=bool)
    for i in range(left, len(arr) - right):
        left_win, right_win = arr[i - left:i], arr[i + 1:i + right + 1]
        if kind == "low":
            out[i] = arr[i] < left_win.min() and arr[i] <= right_win.min()
        else:
            out[i] = arr[i] > left_win.max() and arr[i] >= right_win.max()
    return pd.Series(out, index=values.index)


def confirmed_pivots(flags: np.ndarray, t: int, right: int, lookback: int) -> list[int]:
    """시점 t 기준으로 이미 확정된(i + right <= t) 피벗 인덱스를 오래된 순으로 반환한다."""
    end = t - right
    start = max(0, t - lookback)
    if end < start:
        return []
    return [i for i in range(start, end + 1) if flags[i]]


# ---------------------------------------------------------------------------
# 다이버전스 / 쌍바닥
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PivotPair:
    first: int
    second: int


@dataclass(frozen=True)
class DoubleBottom:
    first: int
    second: int
    neckline: float


def _candidate_pairs(pivots: list[int], cfg: StrategyConfig):
    """가장 최근 피벗(second)과 간격 조건을 만족하는 이전 피벗(first) 쌍을 최근 순으로 생성한다."""
    if len(pivots) < 2:
        return
    second = pivots[-1]
    for first in reversed(pivots[:-1]):
        gap = second - first
        if gap < cfg.pivot_min_gap:
            continue
        if gap > cfg.pivot_max_gap:
            break
        yield first, second


def find_bullish_divergence(
    low: np.ndarray, rsi: np.ndarray, swing_low: np.ndarray, t: int, cfg: StrategyConfig
) -> PivotPair | None:
    """강세 다이버전스: 가격 저점은 낮아지는데 RSI 저점은 높아짐 (첫 저점 RSI 는 과매도권 근처)."""
    lookback = cfg.pivot_max_gap + cfg.macd_max_wait + cfg.pivot_right
    pivots = confirmed_pivots(swing_low, t, cfg.pivot_right, lookback)
    for first, second in _candidate_pairs(pivots, cfg):
        if (
            low[second] < low[first]
            and rsi[second] > rsi[first]
            and rsi[first] <= cfg.divergence_rsi_ceiling
        ):
            return PivotPair(first, second)
    return None


def find_bearish_divergence(
    high: np.ndarray, rsi: np.ndarray, swing_high: np.ndarray, t: int, cfg: StrategyConfig
) -> PivotPair | None:
    """약세 다이버전스: 가격 고점은 높아지는데 RSI 고점은 낮아짐 (첫 고점 RSI 는 과매수권 근처)."""
    lookback = cfg.pivot_max_gap + cfg.macd_max_wait + cfg.pivot_right
    pivots = confirmed_pivots(swing_high, t, cfg.pivot_right, lookback)
    floor = 100.0 - cfg.divergence_rsi_ceiling
    for first, second in _candidate_pairs(pivots, cfg):
        if high[second] > high[first] and rsi[second] < rsi[first] and rsi[first] >= floor:
            return PivotPair(first, second)
    return None


def find_double_bottom(
    low: np.ndarray,
    high: np.ndarray,
    rsi: np.ndarray,
    bb_lower: np.ndarray,
    swing_low: np.ndarray,
    t: int,
    cfg: StrategyConfig,
) -> DoubleBottom | None:
    """쌍바닥: 첫 저점은 BB 하단 이탈 + RSI 과매도, 두 번째 저점은 비슷한 가격이면서 BB 안쪽."""
    lookback = cfg.pivot_max_gap + cfg.double_bottom_max_wait + cfg.pivot_right
    pivots = confirmed_pivots(swing_low, t, cfg.pivot_right, lookback)
    if not pivots or t - pivots[-1] > cfg.double_bottom_max_wait:
        return None
    for first, second in _candidate_pairs(pivots, cfg):
        similar = abs(low[second] - low[first]) / low[first] <= cfg.double_bottom_tolerance
        rsi_window = rsi[max(0, first - cfg.pivot_left): first + cfg.pivot_right + 1]
        oversold_near_first = np.nanmin(rsi_window) <= cfg.oversold if np.any(~np.isnan(rsi_window)) else False
        if (
            similar
            and low[first] < bb_lower[first]
            and oversold_near_first
            and low[second] >= bb_lower[second]
            and second - first > 1
        ):
            neckline = float(np.max(high[first + 1:second]))
            return DoubleBottom(first, second, neckline)
    return None
