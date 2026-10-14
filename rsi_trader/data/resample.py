"""1분봉을 N분봉(예: 15분봉)으로 변환(Resampling)하는 유틸리티."""

from __future__ import annotations

import pandas as pd


def resample_candles(df_1m: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """1분봉 데이터프레임을 주어진 분 단위로 리샘플링한다.

    입력 df_1m:
      - 인덱스: datetime 또는 timestamp 컬럼 포함 (OpenTime 기준)
      - 컬럼: open, high, low, close, volume
    """
    if df_1m.empty:
        return df_1m.copy()

    df = df_1m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        else:
            raise ValueError("DataFrame 에 DatetimeIndex 또는 timestamp 컬럼이 필요합니다.")

    rule = f"{timeframe_minutes}min"

    # 집계 룰
    resampled = df.resample(rule, label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    # 거래가 전혀 없는 빈 구간 제거
    resampled = resampled.dropna(subset=["close"])
    return resampled.reset_index()
