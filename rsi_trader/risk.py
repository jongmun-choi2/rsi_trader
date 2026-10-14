"""리스크 관리 모듈.

포지션 사이징(고정 비율 위험 모델), 일일 최대 손실 제한, 동시 보유 종목 수 제한을 담당한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from rsi_trader.config import RiskConfig


@dataclass(frozen=True)
class PositionSize:
    shares: int
    entry_price: float
    stop_price: float
    risk_amount: float
    total_cost: float


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.current_equity = cfg.initial_equity
        self.day_start_equity = cfg.initial_equity
        self.current_date: date | None = None
        self.halted_for_day: bool = False

    def update_date(self, new_date: date) -> None:
        """날짜가 변경되면 일일 손익 계산 기준점을 갱신하고 서킷브레이커를 해제한다."""
        if self.current_date != new_date:
            self.current_date = new_date
            self.day_start_equity = self.current_equity
            self.halted_for_day = False

    def check_daily_loss(self) -> bool:
        """일일 최대 손실 한도 초과 여부를 점검한다."""
        if self.halted_for_day:
            return True
        loss = self.day_start_equity - self.current_equity
        if loss >= self.day_start_equity * self.cfg.daily_max_loss_pct:
            self.halted_for_day = True
            return True
        return False

    def calculate_position_size(
        self,
        entry: float,
        stop: float,
        active_positions_count: int,
    ) -> PositionSize | None:
        """진입가와 손절가를 기준으로 안전한 진입 수량을 계산한다.
        
        규칙:
        1. 당일 손실 한도 초과 시 진입 금지
        2. 최대 동시 보유 포지션 수 도달 시 진입 금지
        3. 손절폭이 0 이하이거나 유효하지 않으면 진입 금지
        4. 1회 거래 최대 허용 위험 금액(Equity * risk_per_trade)을 손절폭으로 나누어 수량 산출
        5. 총 투자 금액이 계좌의 max_position_pct 를 초과하지 않도록 캡 적용
        """
        if self.check_daily_loss():
            return None

        if active_positions_count >= self.cfg.max_positions:
            return None

        risk_per_share = entry - stop
        if risk_per_share <= 0:
            return None

        # 1회 거래 허용 손실액
        max_risk_amount = self.current_equity * self.cfg.risk_per_trade
        raw_shares = max_risk_amount / risk_per_share

        # 단일 종목 최대 투자 금액 한도
        max_capital = self.current_equity * self.cfg.max_position_pct
        shares_cap_by_capital = max_capital / entry

        final_shares = int(math.floor(min(raw_shares, shares_cap_by_capital)))
        if final_shares <= 0:
            return None

        total_cost = final_shares * entry
        risk_amount = final_shares * risk_per_share

        return PositionSize(
            shares=final_shares,
            entry_price=entry,
            stop_price=stop,
            risk_amount=risk_amount,
            total_cost=total_cost,
        )

    def record_pnl(self, realized_pnl: float) -> None:
        """실현 손익을 계좌 잔고에 반영하고 일일 손실 한도를 체크한다."""
        self.current_equity += realized_pnl
        self.check_daily_loss()
