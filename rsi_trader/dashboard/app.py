"""웹 대시보드 FastAPI 백엔드 서버.

시안 3(모듈러 스택 & 게이지 위젯) 기반 UI를 지원하기 위해
캔들 차트, RSI 게이지, PnL 메트릭, 주문 내역 API를 제공한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from rsi_trader.config import AppConfig, load_config
from rsi_trader.indicators import add_indicators
from rsi_trader.strategy import prepare, MarketArrays, RsiStrategy
from rsi_trader.broker.paper_broker import PaperBroker
from rsi_trader.risk import RiskManager

app = FastAPI(title="RSI Trader Dashboard")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

CONFIG_PATH = Path("/Users/jongmunchoi/project/rsi_trader/config.yaml")
config: AppConfig = load_config(CONFIG_PATH) if CONFIG_PATH.exists() else AppConfig()

broker = PaperBroker(config.risk.initial_equity, config.costs)
risk = RiskManager(config.risk)
strategy = RsiStrategy(config.strategy)

# 시뮬레이션용 데이터 생성 캐시 (실제 연동 시 토스 API 또는 DB에서 로딩)
def get_sample_candles(symbol: str = "005930") -> pd.DataFrame:
    np.random.seed(100)
    n = 150
    times = pd.date_range("2026-10-01 09:00", periods=n, freq="15min")
    walk = np.cumsum(np.random.randn(n) * 150)
    prices = 70000 + walk
    df = pd.DataFrame({
        "timestamp": times,
        "open": prices,
        "high": prices + np.random.uniform(50, 200, n),
        "low": prices - np.random.uniform(50, 200, n),
        "close": prices + np.random.randn(n) * 50,
        "volume": np.random.randint(1000, 20000, n).astype(float),
    })
    return df


@app.get("/api/chart-data")
def get_chart_data(symbol: str = "005930") -> dict[str, Any]:
    df = get_sample_candles(symbol)
    df_prep = prepare(df, config.indicators, config.strategy)
    
    # Lightweight-Charts 형식에 맞춘 변환
    candles = []
    rsi_data = []
    macd_data = []
    bb_upper = []
    bb_lower = []
    bb_mid = []

    for _, row in df_prep.iterrows():
        t = int(row["timestamp"].timestamp())
        candles.append({
            "time": t,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        })
        if pd.notna(row["rsi"]):
            rsi_data.append({"time": t, "value": float(row["rsi"])})
        if pd.notna(row["bb_upper"]):
            bb_upper.append({"time": t, "value": float(row["bb_upper"])})
            bb_lower.append({"time": t, "value": float(row["bb_lower"])})
            bb_mid.append({"time": t, "value": float(row["bb_mid"])})
        if pd.notna(row["macd"]):
            macd_data.append({
                "time": t,
                "macd": float(row["macd"]),
                "signal": float(row["macd_signal"]) if pd.notna(row["macd_signal"]) else 0.0,
                "hist": float(row["macd_hist"]) if pd.notna(row["macd_hist"]) else 0.0,
            })

    latest_rsi = float(df_prep["rsi"].dropna().iloc[-1]) if not df_prep["rsi"].dropna().empty else 50.0

    return {
        "symbol": symbol,
        "candles": candles,
        "rsi": rsi_data,
        "latest_rsi": latest_rsi,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_mid": bb_mid,
        "macd": macd_data,
    }


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    # 포트폴리오 메트릭 요약
    return {
        "total_value": broker.total_equity,
        "cash": broker.cash,
        "daily_pnl": 145000.0,
        "daily_pnl_pct": 1.45,
        "weekly_pnl": 420000.0,
        "total_return_pct": 4.20,
        "active_positions_count": len(broker.positions),
        "circuit_breaker": risk.halted_for_day,
    }


@app.get("/api/orders")
def get_orders() -> list[dict[str, Any]]:
    # 최근 가상 체결 및 포지션 내역
    res = []
    for tr in reversed(broker.trade_history[-10:]):
        res.append({
            "symbol": tr.symbol,
            "setup": tr.setup_name,
            "entry_price": tr.entry_price,
            "exit_price": tr.exit_price,
            "shares": tr.shares,
            "pnl": tr.pnl,
            "return_pct": tr.return_pct,
            "exit_reason": tr.exit_reason,
            "time": tr.exit_time.strftime("%Y-%m-%d %H:%M"),
        })
    return res


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Dashboard loading...</h1>"
