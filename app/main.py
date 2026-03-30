"""
FastAPI Application v2
======================
Dashboard API + scheduled bot runner.
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.background import BackgroundScheduler

from app.services.bot_runner import BotRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

bot = BotRunner()
scheduler = BackgroundScheduler()


def scheduled_run():
    try:
        result = bot.run_cycle()
        action = result.get("action", "none")
        if action not in ("none", "monitoring", "session_closed"):
            log.info(f"Cycle: {action}")
    except Exception as e:
        log.exception(f"Scheduled run error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    interval = int(os.getenv("BOT_INTERVAL_MINUTES", "5"))
    scheduler.add_job(scheduled_run, "interval", minutes=interval, id="bot_cycle")
    scheduler.start()
    log.info(f"Bot started (every {interval}m) | Quote: {bot.quote} | Universe: {bot.universe_size}")
    scheduled_run()
    yield
    scheduler.shutdown()


app = FastAPI(title="VWAP TFC Bot v2", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/status")
async def get_status():
    return bot.status


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    trades = list(reversed(bot.account.trades))[:limit]
    return {"trades": trades, "total": len(bot.account.trades)}


@app.get("/api/equity")
async def get_equity():
    return {
        "starting_balance": bot.account.starting_balance,
        "current_balance": round(bot.account.balance, 2),
        "curve": bot.account.equity_curve,
    }


@app.get("/api/positions")
async def get_positions():
    return bot.status.get("positions", [])


@app.post("/api/run")
async def manual_run():
    return bot.run_cycle()


@app.post("/api/reset")
async def reset_bot():
    global bot
    from app.core.paper_trader import TRADES_FILE, STATE_FILE
    for f in [TRADES_FILE, STATE_FILE]:
        if f.exists():
            f.unlink()
    bot = BotRunner()
    return {"message": "Reset OK", "balance": bot.account.balance}


@app.get("/api/config")
async def get_config():
    c = bot.account.config
    return {
        "quote": bot.quote,
        "universe_size": bot.universe_size,
        "interval_minutes": int(os.getenv("BOT_INTERVAL_MINUTES", "5")),
        "max_positions": c.MAX_POSITIONS,
        "position_pct": c.POSITION_PCT,
        "partial_tp": c.PARTIAL_TP_PCT,
        "full_tp": c.FULL_TP_PCT,
        "stop_loss": c.STOP_LOSS_PCT,
        "trail_distance": c.TRAIL_DISTANCE,
        "daily_loss_limit": c.DAILY_LOSS_LIMIT,
    }


DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard")
if os.path.exists(DASHBOARD_DIR):
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")


@app.get("/")
async def root():
    index = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"name": "VWAP TFC Bot v2", "docs": "/docs"}
