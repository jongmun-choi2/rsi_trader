"""텔레그램 봇 메시지 전송 모듈.

만료 없는 텔레그램 Bot API를 활용하여 트레이딩 시그널 및 모니터링 알림을 실시간 전송한다.
"""

from __future__ import annotations

import logging
import os
import httpx

logger = logging.getLogger("rsi_trader.notifier.telegram")


class TelegramNotifier:
    BASE_URL = "https://api.telegram.org"

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | int | None = None,
    ) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = str(chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()

    def get_chat_id(self) -> str | None:
        """사용자가 봇에게 /start 또는 아무 메시지를 보냈을 때 최근 chat_id를 자동 조회."""
        if not self.bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")
            return None

        url = f"{self.BASE_URL}/bot{self.bot_token}/getUpdates"
        try:
            with httpx.Client() as client:
                resp = client.get(url, timeout=10.0)
                resp.raise_for_status()
                res = resp.json()
                results = res.get("result", [])
                if not results:
                    return None
                # 가장 최근 메시지의 chat_id 반환
                last_update = results[-1]
                msg = last_update.get("message") or last_update.get("channel_post")
                if msg and "chat" in msg:
                    cid = str(msg["chat"]["id"])
                    self.chat_id = cid
                    return cid
        except Exception as e:
            logger.error(f"텔레그램 chat_id 조회 중 오류: {e}")
        return None

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """지정된 채팅방(chat_id)으로 텔레그램 알림 발송."""
        if not self.bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")
            return False

        if not self.chat_id:
            # chat_id가 없으면 자동으로 getUpdates로 조회 시도
            auto_cid = self.get_chat_id()
            if not auto_cid:
                logger.error("TELEGRAM_CHAT_ID를 찾을 수 없습니다. 봇에게 먼저 아무 메시지나 보내주세요.")
                return False

        url = f"{self.BASE_URL}/bot{self.bot_token}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            with httpx.Client() as client:
                resp = client.post(url, json=data, timeout=10.0)
                if resp.status_code == 200:
                    logger.info("텔레그램 알림 발송 성공!")
                    return True
                else:
                    logger.error(f"텔레그램 전송 실패 ({resp.status_code}): {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"텔레그램 전송 중 예외 발생: {e}")
            return False
