"""지표, 패턴, 전략, 리스크 관리 단위 테스트."""

from datetime import date
import numpy as np
import pandas as pd
import pytest

from rsi_trader.config import (
    AppConfig,
    IndicatorConfig,
    RiskConfig,
    StrategyConfig,
    config_from_dict,
)
from rsi_trader.indicators import add_indicators, bollinger, macd, rsi
from rsi_trader.patterns import (
    bearish_engulfing,
    bullish_engulfing,
    confirmed_pivots,
    find_bullish_divergence,
    swing_points,
)
from rsi_trader.risk import RiskManager
from rsi_trader.strategy import MarketArrays, RsiStrategy, Setup, prepare


# ---------------------------------------------------------------------------
# 1. Config Test
# ---------------------------------------------------------------------------
def test_config_validation():
    cfg = config_from_dict({
        "market": "KR",
        "symbols": ["005930"],
        "mode": "backtest",
        "strategy": {"oversold": 25.0, "reward_risk": 2.5},
    })
    assert cfg.market == "KR"
    assert cfg.strategy.oversold == 25.0
    assert cfg.strategy.reward_risk == 2.5

    # 실거래 모드(live) 방지 테스트
    with pytest.raises(ValueError):
        config_from_dict({"mode": "live"})


# ---------------------------------------------------------------------------
# 2. Indicators Test
# ---------------------------------------------------------------------------
def test_rsi_bounds():
    np.random.seed(42)
    # 100개 랜덤 워크 가격 데이터
    prices = 10000 + np.cumsum(np.random.randn(100) * 100)
    series = pd.Series(prices)
    res = rsi(series, period=14)
    valid_res = res.dropna()
    assert len(valid_res) == 100 - 14
    assert (valid_res >= 0).all()
    assert (valid_res <= 100).all()


def test_bollinger_bands():
    series = pd.Series([100.0] * 30)
    bb = bollinger(series, period=20, num_std=2.0)
    # 변동성이 0이면 상/중/하단이 동일해야 함
    assert (bb.loc[20:, "bb_upper"] == 100.0).all()
    assert (bb.loc[20:, "bb_lower"] == 100.0).all()


def test_macd_calculation():
    series = pd.Series(np.linspace(100, 200, 50))
    res = macd(series, fast=12, slow=26, signal=9)
    # 지속 상승 구간에서는 macd 가 양수여야 함
    assert (res.loc[35:, "macd"] > 0).all()


# ---------------------------------------------------------------------------
# 3. Patterns Test
# ---------------------------------------------------------------------------
def test_bullish_engulfing():
    df = pd.DataFrame({
        "open": [105.0, 95.0],
        "close": [100.0, 110.0],  # 0번: 음봉(105->100), 1번: 양봉(95->110) 장악형
    })
    res = bullish_engulfing(df)
    assert not res.iloc[0]
    assert res.iloc[1]


def test_bearish_engulfing():
    df = pd.DataFrame({
        "open": [100.0, 110.0],
        "close": [105.0, 95.0],   # 0번: 양봉(100->105), 1번: 음봉(110->95) 장악형
    })
    res = bearish_engulfing(df)
    assert not res.iloc[0]
    assert res.iloc[1]


def test_swing_pivot_no_lookahead():
    # V자 반등: 인덱스 3이 최저점
    lows = pd.Series([10, 9, 8, 5, 8, 9, 10])
    flags = swing_points(lows, left=2, right=2, kind="low").to_numpy()
    assert flags[3]  # 인덱스 3이 피벗 저점

    # 시점 t=4 에서는 right=2 조건을 만족하지 못하므로 확정 피벗이 아니어야 함 (3 + 2 <= 4 False)
    assert 3 not in confirmed_pivots(flags, t=4, right=2, lookback=10)
    # 시점 t=5 에서 비로소 확정 (3 + 2 <= 5 True)
    assert 3 in confirmed_pivots(flags, t=5, right=2, lookback=10)


# ---------------------------------------------------------------------------
# 4. Strategy Test (Setup A & B)
# ---------------------------------------------------------------------------
def test_strategy_setup_a_engulfing():
    cfg = StrategyConfig(oversold=30.0, engulf_max_wait=3, reward_risk=2.0)
    strategy = RsiStrategy(cfg)

    # 4개 봉 데이터: 0,1번 봉에서 RSI 과매도 및 볼린저 하단 이탈, 2번 봉에서 상승 장악형 양봉
    n = 10
    df = pd.DataFrame({
        "open": [100.0] * n,
        "high": [102.0] * n,
        "low": [98.0] * n,
        "close": [99.0] * n,
        "volume": [1000.0] * n,
        "rsi": [50.0] * n,
        "bb_lower": [95.0] * n,
        "bb_mid": [100.0] * n,
        "bb_upper": [105.0] * n,
        "macd": [0.0] * n,
        "macd_signal": [0.0] * n,
        "vol_ma": [1000.0] * n,
        "bull_engulf": [False] * n,
        "bear_engulf": [False] * n,
        "swing_low": [False] * n,
        "swing_high": [False] * n,
        "cloud_top": [100.0] * n,
        "kijun": [100.0] * n,
    })

    # 인덱스 5: RSI 25, 종가 90 (BB lower 95 이탈 음봉)
    df.loc[5, "rsi"] = 25.0
    df.loc[5, "close"] = 90.0
    df.loc[5, "low"] = 89.0
    # 인덱스 6: 상승 장악형 양봉 발생
    df.loc[6, "bull_engulf"] = True
    df.loc[6, "open"] = 89.0
    df.loc[6, "close"] = 96.0
    df.loc[6, "low"] = 88.0

    arrays = MarketArrays.from_frame(df)
    sig = strategy.entry_signal(arrays, t=6)

    assert sig is not None
    assert sig.setup == Setup.ENGULFING
    assert sig.entry == 96.0
    # 손절가는 최저가 88.0 기준 약간 버퍼 아래
    assert sig.stop < 88.0
    # 익절가는 1:2 손익비 반영
    assert sig.target > sig.entry + 2 * (sig.entry - sig.stop - 0.1)


# ---------------------------------------------------------------------------
# 5. Risk Manager Test
# ---------------------------------------------------------------------------
def test_risk_manager_position_sizing():
    cfg = RiskConfig(
        initial_equity=10_000_000,
        risk_per_trade=0.01,    # 1회 최대 위험액: 100,000원
        max_position_pct=0.3,   # 단일 종목 최대 투자금: 3,000,000원
        daily_max_loss_pct=0.03 # 일일 손실 한도: 300,000원
    )
    rm = RiskManager(cfg)
    rm.update_date(date(2026, 10, 8))

    # 진입가 10,000원, 손절가 9,500원 (주당 손실 500원)
    # 허용 손실 100,000원 / 500원 = 200주 (총 매수금 2,000,000원 <= 3,000,000원)
    pos = rm.calculate_position_size(entry=10000.0, stop=9500.0, active_positions_count=0)
    assert pos is not None
    assert pos.shares == 200
    assert pos.total_cost == 2_000_000.0

    # 당일 손실 -350,000원 기록 -> 일일 서킷브레이커 작동
    rm.record_pnl(-350_000.0)
    assert rm.check_daily_loss()
    # 서킷 작동 후 신규 진입 차단
    assert rm.calculate_position_size(entry=10000.0, stop=9500.0, active_positions_count=0) is None
