"""카카오톡 나에게 보내기 메시지 전송 모듈.

카카오 디벨로퍼스 REST API를 사용하여 신호 발생 시 스마트폰 카카오톡으로 즉시 알림을 발송한다.
"""

from __future__ import annotations

import json
import logging
import os
import httpx

logger = logging.getLogger("rsi_trader.notifier.kakao")


class KakaoNotifier:
    SEND_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    TOKEN_URL = "https://kauth.kakao.com/oauth/token"

    def __init__(
        self,
        access_token: str | None = None,
        refresh_token: str | None = None,
        rest_api_key: str | None = None,
    ) -> None:
        self.access_token = access_token or os.getenv("KAKAO_ACCESS_TOKEN", "")
        self.refresh_token = refresh_token or os.getenv("KAKAO_REFRESH_TOKEN", "")
        self.rest_api_key = rest_api_key or os.getenv("KAKAO_REST_API_KEY", "")

    def refresh_access_token(self) -> bool:
        """리프레시 토큰을 이용해 만료된 액세스 토큰을 자동 갱신."""
        if not self.refresh_token or not self.rest_api_key:
            return False

        data = {
            "grant_type": "refresh_token",
            "client_id": self.rest_api_key,
            "refresh_token": self.refresh_token,
        }
        try:
            with httpx.Client() as client:
                resp = client.post(self.TOKEN_URL, data=data, timeout=10.0)
                if resp.status_code == 200:
                    res_json = resp.json()
                    self.access_token = res_json["access_token"]
                    if "refresh_token" in res_json:
                        self.refresh_token = res_json["refresh_token"]
                    logger.info("카카오톡 액세스 토큰 갱신 성공")
                    return True
                else:
                    logger.error(f"카카오 토큰 갱신 실패: {resp.status_code} {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"카카오 토큰 갱신 중 예외 발생: {e}")
            return False

    def send_message(self, text: str, web_url: str = "https://finance.yahoo.com/quote/SOXL") -> bool:
        """카카오톡 '나에게 보내기' 텍스트 메시지 발송."""
        if not self.access_token:
            logger.warning("카카오 액세스 토큰(KAKAO_ACCESS_TOKEN)이 설정되지 않았습니다.")
            return False

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        template_object = {
            "object_type": "text",
            "text": text,
            "link": {
                "web_url": web_url,
                "mobile_web_url": web_url,
            },
            "button_title": "SOXL 시세 확인",
        }

        data = {"template_object": json.dumps(template_object, ensure_ascii=False)}

        try:
            with httpx.Client() as client:
                resp = client.post(self.SEND_URL, headers=headers, data=data, timeout=10.0)

                # 토큰 만료(401) 시 1회 자동 갱신 시도
                if resp.status_code == 401 and self.refresh_token:
                    logger.info("액세스 토큰 만료됨. 자동 갱신 후 재전송 시도...")
                    if self.refresh_access_token():
                        headers["Authorization"] = f"Bearer {self.access_token}"
                        resp = client.post(self.SEND_URL, headers=headers, data=data, timeout=10.0)

                if resp.status_code == 200:
                    logger.info("카카오톡 알림 발송 성공!")
                    return True
                else:
                    logger.error(f"카카오톡 전송 실패 ({resp.status_code}): {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"카카오톡 전송 중 예외 발생: {e}")
            return False
