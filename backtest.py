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
TIMEOUT_SECONDS = 15
INITIAL_CAPITAL = 10_000
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "btcusdt_5m.csv"
SUMMARY_PATH = Path("results_summary.txt")
EQUITY_CURVE_PATH = Path("equity_curve.png")


def fetch_binance_klines() -> pd.DataFrame | None:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)

    if response.status_code == 451:
        print("Binance API region-restricted (451). Falling back to BingX.")
        return None

    if response.status_code != 200:
        print(f"Binance API failed with status {response.status_code}. Falling back to BingX.")
        return None

    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise ValueError("Binance API returned an invalid or empty payload.")

    columns = [
        "Open time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ]
    frame = pd.DataFrame(payload, columns=columns)
    frame = frame[["Open time", "Open", "High", "Low", "Close", "Volume"]]
    frame.rename(columns={"Open time": "Date"}, inplace=True)
    frame["Date"] = pd.to_datetime(frame["Date"], unit="ms", utc=True)
    return frame


def fetch_bingx_klines() -> pd.DataFrame:
    url = "https://open-api.bingx.com/openApi/spot/v1/market/kline"
    params = {"symbol": "BTC-USDT", "interval": INTERVAL, "limit": LIMIT}
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)

    if response.status_code != 200:
        raise RuntimeError(f"BingX API failed with status {response.status_code}.")

    payload: Any = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None

    if isinstance(data, dict):
        data = data.get("items") or data.get("k") or data.get("candles")

    if not isinstance(data, list) or not data:
        raise ValueError("BingX API returned an invalid or empty payload.")

    normalized_rows: list[dict[str, Any]] = []
    for candle in data:
        if isinstance(candle, list) and len(candle) >= 6:
            ts, open_, high, low, close, volume = candle[:6]
        elif isinstance(candle, dict):
            ts = candle.get("time") or candle.get("t") or candle.get("openTime")
            open_ = candle.get("open") or candle.get("o")
            high = candle.get("high") or candle.get("h")
            low = candle.get("low") or candle.get("l")
            close = candle.get("close") or candle.get("c")
            volume = candle.get("volume") or candle.get("v")
        else:
            continue

        normalized_rows.append(
            {
                "Date": ts,
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )

    if not normalized_rows:
        raise ValueError("BingX API payload contained no parseable candles.")

    frame = pd.DataFrame(normalized_rows)
    frame["Date"] = pd.to_datetime(pd.to_numeric(frame["Date"], errors="coerce"), unit="ms", utc=True)
    return frame


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = frame.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=required)
    frame = frame.sort_values("Date").drop_duplicates(subset=["Date"]).tail(LIMIT)

    if frame.empty:
        raise ValueError("No valid candles after validation.")

    if len(frame) < 50:
        raise ValueError("Not enough candles for indicators (need at least 50).")

    invalid_price = (frame[["Open", "High", "Low", "Close"]] <= 0).any().any()
    if invalid_price:
        raise ValueError("Invalid non-positive OHLC values found.")

    return frame


def rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    series = pd.Series(values)
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50).to_numpy()


def bollinger_middle(values: np.ndarray, period: int = 20) -> np.ndarray:
    return pd.Series(values).rolling(period).mean().to_numpy()


def bollinger_upper(values: np.ndarray, period: int = 20, std_mult: float = 2.0) -> np.ndarray:
    s = pd.Series(values)
    return (s.rolling(period).mean() + std_mult * s.rolling(period).std(ddof=0)).to_numpy()


class BollingerRsiBreakout(Strategy):
    def init(self) -> None:
        self.bb_mid = self.I(bollinger_middle, self.data.Close, 20)
        self.bb_upper = self.I(bollinger_upper, self.data.Close, 20, 2.0)
        self.rsi = self.I(rsi, self.data.Close, 14)

    def next(self) -> None:
        if len(self.data.Close) < 22:
            return

        if self.position:
            return

        close = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        upper = self.bb_upper[-1]
        prev_upper = self.bb_upper[-2]
        rsi_value = self.rsi[-1]

        breakout = prev_close <= prev_upper and close > upper
        if breakout and rsi_value > 55:
            self.buy(sl=close * 0.98, tp=close * 1.03)


def download_data() -> pd.DataFrame:
    try:
        frame = fetch_binance_klines()
    except requests.RequestException as exc:
        print(f"Binance API request error: {exc}. Falling back to BingX.")
        frame = None

    if frame is None:
        try:
            frame = fetch_bingx_klines()
        except requests.RequestException as exc:
            raise RuntimeError(f"BingX API request error: {exc}") from exc

    return validate_ohlcv(frame)


def save_summary(stats: pd.Series) -> None:
    summary_lines = [
        f"Capital initial: {INITIAL_CAPITAL:.2f} USDT",
        f"Capital final: {float(stats.get('Equity Final [$]', np.nan)):.2f} USDT",
        f"Nombre de trades: {int(stats.get('# Trades', 0))}",
        f"Winrate %: {float(stats.get('Win Rate [%]', np.nan)):.2f}",
        f"Profit factor: {float(stats.get('Profit Factor', np.nan)):.4f}",
        f"Expectancy: {float(stats.get('Expectancy [%]', np.nan)):.4f}",
        f"Max drawdown %: {float(stats.get('Max. Drawdown [%]', np.nan)):.2f}",
        f"PnL total %: {float(stats.get('Return [%]', np.nan)):.2f}",
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def save_equity_curve(stats: pd.Series) -> None:
    equity_curve = stats.get("_equity_curve")
    if equity_curve is None or equity_curve.empty:
        raise ValueError("Equity curve is missing from backtest statistics.")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve["Equity"], label="Equity", color="royalblue")
    plt.title("BTCUSDT 5m Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity (USDT)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(EQUITY_CURVE_PATH, dpi=150)
    plt.close()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    frame = download_data()
    frame.to_csv(CSV_PATH, index=False)

    bt_frame = frame.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]

    backtest = Backtest(
        bt_frame,
        BollingerRsiBreakout,
        cash=INITIAL_CAPITAL,
        commission=0.0,
        trade_on_close=True,
        exclusive_orders=True,
    )
    stats = backtest.run()

    save_summary(stats)
    save_equity_curve(stats)

    print("Backtest completed successfully.")
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
