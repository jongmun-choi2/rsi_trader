"""대시보드 FastAPI 엔드포인트 단위 테스트."""

import pytest
from fastapi.testclient import TestClient

from rsi_trader.dashboard.app import app

client = TestClient(app)


def test_dashboard_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "TRADING DASHBOARD" in response.text
    assert "005930" in response.text


def test_dashboard_chart_data():
    response = client.get("/api/chart-data?symbol=005930")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "005930"
    assert len(data["candles"]) > 0
    assert "latest_rsi" in data
    assert len(data["bb_upper"]) > 0


def test_dashboard_metrics():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_value" in data
    assert "daily_pnl" in data
    assert "circuit_breaker" in data


def test_dashboard_orders():
    response = client.get("/api/orders")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
