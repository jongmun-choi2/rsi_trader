"""SOXL 전략 B 실시간 모니터링 및 카카오톡 알림 발송 스크립트.

미국 주식 장중 15분봉 단위로 지표를 계산하고,
- 지지선($115~$125) 부근에서 RSI 매수 시그널 발생 시 알림
- 1차 탈출 목표가($135~$138), 본전 탈출가($145~$148), 손절 기준선($115) 도달 시 알림
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime
import httpx
import pandas as pd
from pathlib import Path

from rsi_trader.config import load_config
from rsi_trader.strategy import prepare, MarketArrays, RsiStrategy
from rsi_trader.notifier.kakao import KakaoNotifier
from rsi_trader.notifier.telegram import TelegramNotifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("soxl_monitor")


def load_env_file(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def fetch_soxl_candles() -> pd.DataFrame:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/SOXL?range=5d&interval=15m"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = httpx.get(url, headers=headers, timeout=10.0)
    resp.raise_for_status()
    data = resp.json()["chart"]["result"][0]
    timestamps = [datetime.fromtimestamp(ts) for ts in data["timestamp"]]
    quote = data["indicators"]["quote"][0]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "volume": quote["volume"],
    }).dropna().reset_index(drop=True)
    return df


class SoxlStrategyBMonitor:
    def __init__(self, user_avg: float = 148.88) -> None:
        load_env_file()
        self.user_avg = user_avg
        self.config = load_config("config.yaml")
        self.strategy = RsiStrategy(self.config.strategy)
        self.kakao_notifier = KakaoNotifier()
        self.telegram_notifier = TelegramNotifier()
        self.last_candle_time: datetime | None = None
        self.last_alert_key: str | None = None

    def send_alert(self, msg: str) -> None:
        """설정된 채널(텔레그램, 카카오톡)로 알림 발송."""
        if self.telegram_notifier.bot_token:
            self.telegram_notifier.send_message(msg)
        if self.kakao_notifier.access_token:
            self.kakao_notifier.send_message(msg)

    def check_once(self) -> None:
        try:
            df = fetch_soxl_candles()
            if df.empty:
                return

            df_prepared = prepare(df, self.config.indicators, self.config.strategy)
            arrays = MarketArrays.from_frame(df_prepared)
            t = len(arrays) - 1

            candle_time = df["timestamp"].iloc[t]
            cur_close = arrays.close[t]
            cur_rsi = arrays.rsi[t]

            logger.info(
                f"[SOXL 모니터링] 캔들시각: {candle_time.strftime('%m-%d %H:%M')} | "
                f"현재가: ${cur_close:.2f} | RSI: {cur_rsi:.1f} | 평단가: ${self.user_avg:.2f}"
            )

            # 신규 15분봉 발생 시 신호 분석
            if self.last_candle_time == candle_time:
                return
            self.last_candle_time = candle_time

            # 1. 전략 B 매수 시그널 검사 (지지선 $115~$125 구간 & 셋업 A/B)
            sig = self.strategy.entry_signal(arrays, t)
            if sig is not None:
                setup_desc = "과매도 장악형 양봉" if "ENGULF" in sig.setup.value else "강세 다이버전스 + MACD 골든크로스"
                alert_key = f"BUY_{sig.setup.value}_{candle_time}"
                if alert_key != self.last_alert_key:
                    self.last_alert_key = alert_key
                    msg = (
                        f"🔔 [SOXL 전략 B 추가 매수 시그널]\n\n"
                        f"• 시그널: {setup_desc} ({sig.setup.value})\n"
                        f"• 현재가: ${cur_close:.2f}\n"
                        f"• RSI: {cur_rsi:.1f}\n"
                        f"• 추천 손절가: ${sig.stop:.2f}\n"
                        f"• 목표 반등가: ${sig.target:.2f}\n\n"
                        f"📌 행동 가이드: 1:1 추가 매수 시 평단가가 약 ${(self.user_avg + cur_close)/2:.2f}로 낮아집니다!"
                    )
                    logger.info(f"카카오톡 알림 전송: {msg}")
                    self.send_alert(msg)
                    return

            # 2. 목표가 / 손절선 알림
            if cur_close >= 145.0:
                alert_key = f"EXIT_BREAKEVEN_{candle_time.strftime('%Y%m%d')}"
                if alert_key != self.last_alert_key:
                    self.last_alert_key = alert_key
                    msg = (
                        f"🎉 [SOXL 본전 탈출 구간 도달!]\n\n"
                        f"• 현재가: ${cur_close:.2f}\n"
                        f"• 내 평단가: ${self.user_avg:.2f}\n"
                        f"• 상태: 평단가 도달! 전량 또는 절반 매도로 안전 탈출을 권장합니다."
                    )
                    self.send_alert(msg)

            elif cur_close >= 135.0 and cur_close < 138.0:
                alert_key = f"EXIT_PARTIAL_{candle_time.strftime('%Y%m%d')}"
                if alert_key != self.last_alert_key:
                    self.last_alert_key = alert_key
                    msg = (
                        f"⚡ [SOXL 1차 반등 목표가 도달]\n\n"
                        f"• 현재가: ${cur_close:.2f}\n"
                        f"• 손실률: {(cur_close - self.user_avg)/self.user_avg * 100:.1f}%\n"
                        f"• 가이드: 1차 저항선입니다. 물량의 30~50%를 부분 정리하여 현금을 확보하세요."
                    )
                    self.send_alert(msg)

            elif cur_close <= 115.0:
                alert_key = f"STOP_LOSS_{candle_time.strftime('%Y%m%d')}"
                if alert_key != self.last_alert_key:
                    self.last_alert_key = alert_key
                    msg = (
                        f"🚨 [SOXL 전저점 이탈 경고!]\n\n"
                        f"• 현재가: ${cur_close:.2f}\n"
                        f"• 기준 지지선: $115.00 하방 이탈\n"
                        f"• 가이드: $100선 추가 급락 위험이 있습니다. 비중 축소 또는 손절을 검토하세요."
                    )
                    self.send_alert(msg)

        except Exception as e:
            logger.error(f"모니터링 수행 중 에러: {e}")

    def run_loop(self, poll_interval_sec: int = 60) -> None:
        """반복 모니터링 실행."""
        logger.info(f"SOXL 전략 B 모니터링 시작 (주기: {poll_interval_sec}초)...")
        # 시작 안내 메시지 전송
        start_msg = (
            "🚀 <b>[SOXL 전략 B 모니터링 가동 시작]</b>\n\n"
            f"• 내 평단가: <b>${self.user_avg:.2f}</b>\n"
            "• 모니터링 타임프레임: <b>15분봉</b>\n"
            "• 감시 조건: 지지선($115~$125) RSI 반등 신호, 1차 목표가($135), 본전 탈출($145~$148), 손절선($115)\n\n"
            "미국 주식 시장 중 신호 발생 시 즉시 알려드립니다."
        )
        self.send_alert(start_msg)

        while True:
            self.check_once()
            time.sleep(poll_interval_sec)


if __name__ == "__main__":
    monitor = SoxlStrategyBMonitor()
    monitor.run_loop(poll_interval_sec=60)
