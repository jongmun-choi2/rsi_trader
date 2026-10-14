"""가상 페이퍼 브로커 (Paper Broker).

슬리피지, 증권거래세, 위탁수수료를 엄격하게 반영하여 가상 주문 체결 및 잔고/포지션을 관리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from rsi_trader.config import CostConfig


@dataclass
class PaperPosition:
    symbol: str
    shares: int
    entry_price: float
    stop_price: float
    target_price: float
    entry_time: datetime
    setup_name: str


@dataclass
class TradeRecord:
    symbol: str
    setup_name: str
    entry_time: datetime
    exit_time: datetime
    shares: int
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    fees: float
    exit_reason: str


class PaperBroker:
    def __init__(self, initial_cash: float, costs: CostConfig) -> None:
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.costs = costs
        self.positions: dict[str, PaperPosition] = {}
        self.trade_history: list[TradeRecord] = []

    @property
    def total_equity(self) -> float:
        # 평가 손익을 제외한 현금 기준 (포지션 보유 시 진입가 기준 추정)
        pos_value = sum(p.shares * p.entry_price for p in self.positions.values())
        return self.cash + pos_value

    def buy(
        self,
        symbol: str,
        shares: int,
        price: float,
        stop_price: float,
        target_price: float,
        time: datetime,
        setup_name: str,
    ) -> bool:
        """매수 체결 (슬리피지 및 수수료 반영)."""
        if symbol in self.positions or shares <= 0:
            return False

        fill_price = price * (1.0 + self.costs.slippage_pct)
        gross_cost = shares * fill_price
        commission = gross_cost * self.costs.commission_rate
        total_required = gross_cost + commission

        if self.cash < total_required:
            return False

        self.cash -= total_required
        self.positions[symbol] = PaperPosition(
            symbol=symbol,
            shares=shares,
            entry_price=fill_price,
            stop_price=stop_price,
            target_price=target_price,
            entry_time=time,
            setup_name=setup_name,
        )
        return True

    def sell(
        self,
        symbol: str,
        price: float,
        time: datetime,
        reason: str,
    ) -> TradeRecord | None:
        """매도 체결 (슬리피지, 거래세 및 수수료 반영)."""
        pos = self.positions.pop(symbol, None)
        if not pos:
            return None

        fill_price = price * (1.0 - self.costs.slippage_pct)
        gross_revenue = pos.shares * fill_price

        commission = gross_revenue * self.costs.commission_rate
        tax = gross_revenue * self.costs.sell_tax_rate
        total_fees = commission + tax

        net_revenue = gross_revenue - total_fees
        self.cash += net_revenue

        # 손익 계산
        gross_cost = pos.shares * pos.entry_price
        pnl = net_revenue - gross_cost
        return_pct = (pnl / gross_cost) * 100.0

        record = TradeRecord(
            symbol=symbol,
            setup_name=pos.setup_name,
            entry_time=pos.entry_time,
            exit_time=time,
            shares=pos.shares,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            pnl=pnl,
            return_pct=return_pct,
            fees=total_fees,
            exit_reason=reason,
        )
        self.trade_history.append(record)
        return record
