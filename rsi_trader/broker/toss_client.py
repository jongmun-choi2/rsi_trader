"""토스증권 OpenAPI REST 클라이언트.

인증(OAuth2 Client Credentials Grant) 및 시세·캔들 데이터 조회 전용.
(실거래 주문 함수는 의도적으로 제외하여 안전성을 보장)
"""

from __future__ import annotations

import time
from typing import Any
import httpx


class TossApiClient:
    BASE_URL = "https://openapi.tossinvest.com"

    def __init__(self, client_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0

    def get_token(self, force_refresh: bool = False) -> str:
        """유효한 액세스 토큰을 반환한다. 만료 시 자동 갱신."""
        now = time.time()
        if not force_refresh and self._access_token and now < self._token_expires_at - 60:
            return self._access_token

        url = f"{self.BASE_URL}/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        with httpx.Client() as client:
            resp = client.post(url, headers=headers, data=data, timeout=10.0)
            resp.raise_for_status()
            res_json = resp.json()

            self._access_token = res_json["access_token"]
            expires_in = res_json.get("expires_in", 3600)
            self._token_expires_at = now + expires_in
            return self._access_token

    def _auth_headers(self) -> dict[str, str]:
        token = self.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def get_candles(
        self,
        symbol: str,
        interval: str = "1m",
        count: int = 200,
        before: str | None = None,
        adjusted: bool = True,
    ) -> list[dict[str, Any]]:
        """캔들 데이터 조회 API.

        - interval: '1m' 또는 '1d' (토스증권 API 제약)
        - count: 최대 200개
        - before: 특정 timestamp 이전 데이터 페이지네이션
        """
        url = f"{self.BASE_URL}/api/v1/candles"
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "count": min(count, 200),
            "adjusted": "true" if adjusted else "false",
        }
        if before:
            params["before"] = before

        with httpx.Client() as client:
            resp = client.get(url, headers=self._auth_headers(), params=params, timeout=10.0)
            resp.raise_for_status()
            # 최신순 정렬 반환
            return resp.json().get("candles", [])

    def get_current_price(self, symbol: str) -> dict[str, Any]:
        """현재가 조회 API."""
        url = f"{self.BASE_URL}/api/v1/prices"
        params = {"symbols": symbol}
        with httpx.Client() as client:
            resp = client.get(url, headers=self._auth_headers(), params=params, timeout=10.0)
            resp.raise_for_status()
            return resp.json()
