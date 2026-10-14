"""페이퍼 트레이딩 실시간 실행 루프.

실시간으로 캔들을 동기화하고, 새로운 완성 봉이 생성될 때마다 신호를 판정하여 가상 체결을 진행한다.
"""

from __future__ import annotations

import time
import logging
from datetime import datetime
import pandas as pd

from rsi_trader.config import AppConfig
from rsi_trader.broker.toss_client import TossApiClient
from rsi_trader.broker.paper_broker import PaperBroker
from rsi_trader.data.resample import resample_candles
from rsi_trader.risk import RiskManager
from rsi_trader.strategy import MarketArrays, RsiStrategy, prepare

logger = logging.getLogger("rsi_trader.runner")


class PaperTradingRunner:
    def __init__(self, config: AppConfig, toss_client: TossApiClient | None = None) -> None:
        self.config = config
        self.toss_client = toss_client
        self.strategy = RsiStrategy(config.strategy)
        self.risk = RiskManager(config.risk)
        self.broker = PaperBroker(config.risk.initial_equity, config.costs)
        self.last_bar_time: dict[str, pd.Timestamp] = {}

    def process_symbol_bars(self, symbol: str, df_15m: pd.DataFrame) -> None:
        """새로 갱신된 N분봉 데이터프레임을 기반으로 실시간 신호 평가 및 가상 체결 수행."""
        if df_15m.empty:
            return

        df_prepared = prepare(df_15m, self.config.indicators, self.config.strategy)
        arrays = MarketArrays.from_frame(df_prepared)
        t = len(arrays) - 1  # 가장 최근 완성 봉

        current_time = pd.to_datetime(df_prepared["timestamp"].iloc[t]) if "timestamp" in df_prepared.columns else pd.Timestamp.now()
        
        # 이미 처리한 봉이면 스킵
        if symbol in self.last_bar_time and self.last_bar_time[symbol] >= current_time:
            return
        self.last_bar_time[symbol] = current_time

        self.risk.update_date(current_time.date())
        current_close = arrays.close[t]
        current_low = arrays.low[t]
        current_high = arrays.high[t]

        # 1. 포지션 보유 중인 경우 청산 검토
        if symbol in self.broker.positions:
            pos = self.broker.positions[symbol]
            if current_low <= pos.stop_price:
                rec = self.broker.sell(symbol, pos.stop_price, current_time.to_pydatetime(), "STOP_LOSS")
                if rec:
                    self.risk.record_pnl(rec.pnl)
                    logger.info(f"[{symbol}] 손절 체결: {rec.exit_price:,.0f}원 (PnL: {rec.pnl:,.0f}원)")
            elif current_high >= pos.target_price:
                rec = self.broker.sell(symbol, pos.target_price, current_time.to_pydatetime(), "TAKE_PROFIT")
                if rec:
                    self.risk.record_pnl(rec.pnl)
                    logger.info(f"[{symbol}] 익절 체결: {rec.exit_price:,.0f}원 (PnL: {rec.pnl:,.0f}원)")
            else:
                exit_reason = self.strategy.exit_signal(arrays, t)
                if exit_reason:
                    rec = self.broker.sell(symbol, current_close, current_time.to_pydatetime(), exit_reason)
                    if rec:
                        self.risk.record_pnl(rec.pnl)
                        logger.info(f"[{symbol}] 지표 시그널 청산({exit_reason}): {rec.exit_price:,.0f}원")

        # 2. 미보유 시 신규 진입 검토
        if symbol not in self.broker.positions:
            sig = self.strategy.entry_signal(arrays, t)
            if sig is not None:
                pos_size = self.risk.calculate_position_size(
                    entry=sig.entry,
                    stop=sig.stop,
                    active_positions_count=len(self.broker.positions),
                )
                if pos_size is not None and pos_size.shares > 0:
                    success = self.broker.buy(
                        symbol=symbol,
                        shares=pos_size.shares,
                        price=sig.entry,
                        stop_price=sig.stop,
                        target_price=sig.target,
                        time=current_time.to_pydatetime(),
                        setup_name=sig.setup.value,
                    )
                    if success:
                        logger.info(
                            f"[{symbol}] 매수 진입({sig.setup.value}): {pos_size.shares}주 @ {sig.entry:,.0f}원 "
                            f"(손절가: {sig.stop:,.0f}원, 목표가: {sig.target:,.0f}원)"
                        )
