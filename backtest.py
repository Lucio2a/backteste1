from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from backtesting import Backtest, Strategy

SYMBOL = "BTCUSDT"
INTERVAL = "5m"
LIMIT = 1000
TIMEOUT_SECONDS = 20
INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.01

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "btcusdt_5m.csv"
SUMMARY_PATH = Path("results_summary.txt")
TRADES_PATH = Path("trades.csv")
EQUITY_CURVE_PATH = Path("equity_curve.png")


def _normalize_candles(candles: list[Any], source: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for candle in candles:
        if isinstance(candle, list) and len(candle) >= 6:
            ts, open_, high, low, close, volume = candle[:6]
        elif isinstance(candle, dict):
            ts = candle.get("time") or candle.get("t") or candle.get("openTime") or candle.get("timestamp")
            open_ = candle.get("open") or candle.get("o")
            high = candle.get("high") or candle.get("h")
            low = candle.get("low") or candle.get("l")
            close = candle.get("close") or candle.get("c")
            volume = candle.get("volume") or candle.get("v")
        else:
            continue

        rows.append(
            {
                "Date": ts,
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )

    if not rows:
        raise ValueError(f"{source}: no parseable candles returned")

    frame = pd.DataFrame(rows)
    frame["Date"] = pd.to_datetime(pd.to_numeric(frame["Date"], errors="coerce"), unit="ms", utc=True)
    return frame


def fetch_bingx() -> pd.DataFrame:
    url = "https://open-api.bingx.com/openApi/spot/v1/market/kline"
    params = {"symbol": "BTC-USDT", "interval": INTERVAL, "limit": LIMIT}
    resp = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)

    if resp.status_code == 451:
        raise RuntimeError("BingX blocked (HTTP 451)")
    if resp.status_code != 200:
        raise RuntimeError(f"BingX failed (HTTP {resp.status_code})")

    payload = resp.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict):
        data = data.get("items") or data.get("k") or data.get("candles")
    if not isinstance(data, list):
        raise ValueError("BingX payload shape is invalid")

    return _normalize_candles(data, "BingX")


def fetch_binance() -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
    resp = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)

    if resp.status_code == 451:
        raise RuntimeError("Binance blocked (HTTP 451)")
    if resp.status_code != 200:
        raise RuntimeError(f"Binance failed (HTTP {resp.status_code})")

    payload = resp.json()
    if not isinstance(payload, list):
        raise ValueError("Binance payload shape is invalid")

    return _normalize_candles(payload, "Binance")


def fetch_bybit() -> pd.DataFrame:
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": SYMBOL, "interval": "5", "limit": LIMIT}
    resp = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)

    if resp.status_code == 451:
        raise RuntimeError("Bybit blocked (HTTP 451)")
    if resp.status_code != 200:
        raise RuntimeError(f"Bybit failed (HTTP {resp.status_code})")

    payload = resp.json()
    data = payload.get("result", {}).get("list") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("Bybit payload shape is invalid")

    # Bybit returns newest first -> reverse
    data = list(reversed(data))
    return _normalize_candles(data, "Bybit")


def build_synthetic_data() -> pd.DataFrame:
    """Last-resort fallback so the script always produces outputs in CI."""
    print("All remote APIs failed. Falling back to synthetic BTCUSDT 5m data.")

    idx = pd.date_range(end=pd.Timestamp.utcnow().floor("5min"), periods=LIMIT, freq="5min", tz="UTC")
    rng = np.random.default_rng(42)
    drift = 0.00012
    vol = 0.002
    returns = rng.normal(drift, vol, size=LIMIT)

    close = 50000 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.maximum(0.0008 * close, 1.0)
    high = np.maximum(open_, close) + spread * rng.uniform(0.2, 1.2, size=LIMIT)
    low = np.minimum(open_, close) - spread * rng.uniform(0.2, 1.2, size=LIMIT)
    volume = rng.uniform(20, 250, size=LIMIT)

    return pd.DataFrame(
        {
            "Date": idx,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if frame.empty:
        raise ValueError("OHLCV dataframe is empty")

    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}")

    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"], utc=True, errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=required).sort_values("Date").drop_duplicates(subset=["Date"]).tail(LIMIT)

    if data.empty:
        raise ValueError("No valid OHLCV rows after cleanup")

    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Invalid OHLC values detected (<=0)")

    if len(data) < 100:
        raise ValueError("Not enough candles for backtest (minimum 100)")

    return data


def get_market_data() -> tuple[pd.DataFrame, str]:
    errors: list[str] = []

    for source, fn in [("BingX", fetch_bingx), ("Binance", fetch_binance), ("Bybit", fetch_bybit)]:
        try:
            df = validate_ohlcv(fn())
            print(f"Data source used: {source}")
            return df, source
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            msg = f"{source}: {exc}"
            errors.append(msg)
            print(f"[WARN] {msg}")

    synthetic = validate_ohlcv(build_synthetic_data())
    return synthetic, "synthetic"


def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    s = pd.Series(series)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50).to_numpy()


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    prev_close = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().to_numpy()


def bb_middle(series: np.ndarray, period: int = 20) -> np.ndarray:
    return pd.Series(series).rolling(period).mean().to_numpy()


def bb_upper(series: np.ndarray, period: int = 20, std: float = 2.0) -> np.ndarray:
    s = pd.Series(series)
    return (s.rolling(period).mean() + std * s.rolling(period).std(ddof=0)).to_numpy()


class BollingerRsiBreakout(Strategy):
    def init(self) -> None:
        self.mid = self.I(bb_middle, self.data.Close, 20)
        self.up = self.I(bb_upper, self.data.Close, 20, 2.0)
        self.rsi_v = self.I(rsi, self.data.Close, 14)
        self.atr_v = self.I(atr, self.data.High, self.data.Low, self.data.Close, 14)

    def next(self) -> None:
        if len(self.data.Close) < 30:
            return

        close = float(self.data.Close[-1])
        prev_close = float(self.data.Close[-2])
        upper = float(self.up[-1])
        prev_upper = float(self.up[-2])
        rsi_value = float(self.rsi_v[-1])
        atr_now = float(self.atr_v[-1])

        if not np.isfinite([upper, prev_upper, rsi_value, atr_now]).all() or atr_now <= 0:
            return

        breakout = prev_close <= prev_upper and close > upper
        momentum_ok = rsi_value > 55

        if self.position or not breakout or not momentum_ok:
            return

        stop_loss = close - 1.5 * atr_now
        take_profit = close + 3.0 * atr_now
        risk_per_unit = close - stop_loss
        if risk_per_unit <= 0:
            return

        risk_budget = self.equity * RISK_PER_TRADE
        size_units = max(1, int(risk_budget / risk_per_unit))

        self.buy(size=size_units, sl=stop_loss, tp=take_profit)


def compute_metrics(stats: pd.Series, equity_curve: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    total_trades = int(stats.get("# Trades", 0))
    winning_trades = int((trades.get("PnL", pd.Series(dtype=float)) > 0).sum()) if not trades.empty else 0
    losing_trades = int((trades.get("PnL", pd.Series(dtype=float)) <= 0).sum()) if not trades.empty else 0

    gross_profit = float(trades.loc[trades["PnL"] > 0, "PnL"].sum()) if not trades.empty else 0.0
    gross_loss = float(-trades.loc[trades["PnL"] < 0, "PnL"].sum()) if not trades.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    expectancy = float(trades["PnL"].mean()) if not trades.empty else 0.0
    capital_final = float(stats.get("Equity Final [$]", INITIAL_CAPITAL))
    pnl_total = capital_final - INITIAL_CAPITAL

    peak = equity_curve["Equity"].cummax()
    dd = (equity_curve["Equity"] - peak) / peak
    max_dd = float(dd.min()) if not dd.empty else 0.0

    return {
        "capital_initial": float(INITIAL_CAPITAL),
        "capital_final": capital_final,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "winrate_percent": float((winning_trades / total_trades) * 100 if total_trades else 0.0),
        "profit_factor": float(profit_factor),
        "expectancy": expectancy,
        "max_drawdown": max_dd,
        "pnl_total": pnl_total,
        "equity_curve": EQUITY_CURVE_PATH.as_posix(),
    }


def write_summary(metrics: dict[str, Any], source: str) -> None:
    lines = [
        f"data_source={source}",
        f"capital_initial={metrics['capital_initial']:.2f}",
        f"capital_final={metrics['capital_final']:.2f}",
        f"total_trades={metrics['total_trades']}",
        f"winning_trades={metrics['winning_trades']}",
        f"losing_trades={metrics['losing_trades']}",
        f"winrate_percent={metrics['winrate_percent']:.2f}",
        f"profit_factor={metrics['profit_factor']:.6f}",
        f"expectancy={metrics['expectancy']:.6f}",
        f"max_drawdown={metrics['max_drawdown']:.6f}",
        f"pnl_total={metrics['pnl_total']:.2f}",
        f"equity_curve={metrics['equity_curve']}",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    market_data, source = get_market_data()
    market_data.to_csv(CSV_PATH, index=False)

    bt_data = market_data.copy().set_index("Date")
    bt = Backtest(
        bt_data,
        BollingerRsiBreakout,
        cash=INITIAL_CAPITAL,
        commission=0.0006,
        exclusive_orders=True,
        trade_on_close=True,
    )

    stats = bt.run()
    trades = stats.get("_trades", pd.DataFrame())
    equity_curve = stats.get("_equity_curve", pd.DataFrame())

    if trades is None:
        trades = pd.DataFrame()
    if equity_curve is None or equity_curve.empty:
        raise RuntimeError("Backtest did not return an equity curve")

    trades.to_csv(TRADES_PATH, index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve["Equity"], color="tab:blue", linewidth=1.6)
    plt.title("BTCUSDT 5m - Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity (USDT)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(EQUITY_CURVE_PATH, dpi=150)
    plt.close()

    metrics = compute_metrics(stats, equity_curve, trades)
    write_summary(metrics, source)

    print("Backtest completed.")
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
