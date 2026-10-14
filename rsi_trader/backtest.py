"""이벤트 기반 백테스터 모듈.

미래 참조 없이 봉 단위(Bar-by-Bar)로 시뮬레이션하여 거래 성과(총수익률, 승률, 손익비, MDD)를 산출한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np

from rsi_trader.config import AppConfig
from rsi_trader.broker.paper_broker import PaperBroker, TradeRecord
from rsi_trader.risk import RiskManager
from rsi_trader.strategy import MarketArrays, RsiStrategy, prepare


@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    total_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: list[TradeRecord]


class Backtester:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.strategy = RsiStrategy(config.strategy)
        self.risk = RiskManager(config.risk)
        self.broker = PaperBroker(config.risk.initial_equity, config.costs)

    def run(self, df_bars: pd.DataFrame, symbol: str) -> BacktestResult:
        """준비된 OHLCV 캔들 데이터프레임을 바 단위로 실행."""
        df_prepared = prepare(df_bars, self.config.indicators, self.config.strategy)
        arrays = MarketArrays.from_frame(df_prepared)

        has_timestamp = "timestamp" in df_prepared.columns
        timestamps = (
            pd.to_datetime(df_prepared["timestamp"])
            if has_timestamp
            else pd.date_range("2026-01-01", periods=len(df_prepared), freq="15min")
        )

        equity_curve: list[float] = [self.broker.total_equity]

        for t in range(len(arrays)):
            current_time = timestamps[t]
            self.risk.update_date(current_time.date())

            current_low = arrays.low[t]
            current_high = arrays.high[t]
            current_close = arrays.close[t]

            # 1. 기존 보유 포지션 익절 / 손절 / 시그널 청산 확인
            if symbol in self.broker.positions:
                pos = self.broker.positions[symbol]
                
                # A. 손절 체크 (저가가 stop 이하로 떨어졌을 때)
                if current_low <= pos.stop_price:
                    exit_price = min(pos.stop_price, current_close)
                    rec = self.broker.sell(symbol, exit_price, current_time, "STOP_LOSS")
                    if rec:
                        self.risk.record_pnl(rec.pnl)

                # B. 익절 체크 (고가가 target 이상으로 도달했을 때)
                elif current_high >= pos.target_price:
                    exit_price = max(pos.target_price, current_close)
                    rec = self.broker.sell(symbol, exit_price, current_time, "TAKE_PROFIT")
                    if rec:
                        self.risk.record_pnl(rec.pnl)

                # C. 전략상 지표 청산 시그널 체크
                else:
                    exit_reason = self.strategy.exit_signal(arrays, t)
                    if exit_reason:
                        rec = self.broker.sell(symbol, current_close, current_time, exit_reason)
                        if rec:
                            self.risk.record_pnl(rec.pnl)

            # 2. 신규 진입 시그널 확인 (포지션이 없을 때)
            if symbol not in self.broker.positions:
                sig = self.strategy.entry_signal(arrays, t)
                if sig is not None:
                    pos_size = self.risk.calculate_position_size(
                        entry=sig.entry,
                        stop=sig.stop,
                        active_positions_count=len(self.broker.positions),
                    )
                    if pos_size is not None and pos_size.shares > 0:
                        self.broker.buy(
                            symbol=symbol,
                            shares=pos_size.shares,
                            price=sig.entry,
                            stop_price=sig.stop,
                            target_price=sig.target,
                            time=current_time,
                            setup_name=sig.setup.value,
                        )

            equity_curve.append(self.broker.total_equity)

        # 결과 집계
        trades = self.broker.trade_history
        total_trades = len(trades)
        wins = [tr for tr in trades if tr.pnl > 0]
        losses = [tr for tr in trades if tr.pnl <= 0]
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(tr.pnl for tr in wins)
        gross_loss = abs(sum(tr.pnl for tr in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        total_pnl = sum(tr.pnl for tr in trades)
        total_return_pct = (total_pnl / self.config.risk.initial_equity) * 100.0

        # MDD 계산
        equity_series = pd.Series(equity_curve)
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_drawdown_pct = abs(float(drawdown.min())) * 100.0 if not drawdown.empty else 0.0

        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
        )
