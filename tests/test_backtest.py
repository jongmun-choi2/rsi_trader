"""브로커 및 백테스터 통합 단위 테스트."""

import pandas as pd
import numpy as np
from datetime import datetime

import pytest
from rsi_trader.config import AppConfig, CostConfig
from rsi_trader.broker.paper_broker import PaperBroker
from rsi_trader.backtest import Backtester


def test_paper_broker_buy_sell():
    costs = CostConfig(commission_rate=0.00015, sell_tax_rate=0.002, slippage_pct=0.001)
    broker = PaperBroker(initial_cash=10_000_000, costs=costs)

    # 매수 실행: 10,000원, 100주
    bought = broker.buy(
        symbol="005930",
        shares=100,
        price=10_000.0,
        stop_price=9_500.0,
        target_price=11_000.0,
        time=datetime(2026, 10, 8, 9, 30),
        setup_name="TEST_SETUP",
    )
    assert bought
    assert "005930" in broker.positions
    pos = broker.positions["005930"]
    # 슬리피지(0.1%) 적용 매수가 확인
    assert pos.entry_price == pytest.approx(10_010.0)

    # 매도 실행: 10,500원 (이익 실현)
    rec = broker.sell(
        symbol="005930",
        price=10_500.0,
        time=datetime(2026, 10, 8, 10, 0),
        reason="TAKE_PROFIT",
    )
    assert rec is not None
    assert rec.pnl > 0
    assert len(broker.trade_history) == 1
    assert "005930" not in broker.positions


def test_backtester_execution():
    config = AppConfig()
    backtester = Backtester(config)

    # 100개 봉 합성 데이터 생성
    n = 100
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame({
        "open": prices,
        "high": prices + 50,
        "low": prices - 50,
        "close": prices,
        "volume": [10000] * n,
        "timestamp": pd.date_range("2026-10-01", periods=n, freq="15min"),
    })

    result = backtester.run(df, symbol="005930")
    assert result is not None
    assert result.max_drawdown_pct >= 0.0
    assert result.total_return_pct is not None
