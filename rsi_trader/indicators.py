"""기술적 지표 (순수 함수).

모든 지표는 과거 데이터만 사용(causal)하므로 백테스트에서 미래 참조가 발생하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsi_trader.config import IndicatorConfig


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 50.0 if avg_gain == 0 else 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI.

    첫 값은 처음 `period`개 변화량의 단순평균으로 시작하고, 이후 Wilder 평활을 적용한다.
    (TradingView 기본 RSI 와 동일한 방식)
    """
    if period < 1:
        raise ValueError("period 는 1 이상이어야 합니다")
    values = close.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) <= period:
        return pd.Series(out, index=close.index, name="rsi")

    delta = np.diff(values)
    gains = np.clip(delta, 0, None)
    losses = np.clip(-delta, 0, None)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = _rsi_value(avg_gain, avg_loss)
    return pd.Series(out, index=close.index, name="rsi")


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """볼린저 밴드 (모표준편차 ddof=0, TradingView 방식)."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": mid + num_std * std, "bb_lower": mid - num_std * std},
        index=close.index,
    )


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD. 초기 불안정 구간(워밍업)은 NaN 으로 마스킹해 가짜 크로스를 방지한다."""
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    position = np.arange(len(close))
    line = line.where(position >= slow - 1)
    sig = sig.where(position >= slow + signal - 2)
    return pd.DataFrame(
        {"macd": line, "macd_signal": sig, "macd_hist": line - sig},
        index=close.index,
    )


def ichimoku(high: pd.Series, low: pd.Series, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52) -> pd.DataFrame:
    """일목균형표 (전환선, 기준선, 선행스팬A, 선행스팬B, 구름대상단)."""
    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
    senkou_b = ((high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2).shift(kijun_period)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    return pd.DataFrame(
        {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b, "cloud_top": cloud_top},
        index=high.index,
    )


def add_indicators(df: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """OHLCV DataFrame 에 지표 컬럼을 추가한 복사본을 반환한다."""
    out = df.copy()
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out = out.join(bollinger(out["close"], cfg.bb_period, cfg.bb_std))
    out = out.join(macd(out["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal))
    out = out.join(ichimoku(out["high"], out["low"]))
    out["vol_ma"] = out["volume"].rolling(cfg.volume_ma_period).mean()
    return out

