from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy


OUTPUT_CSV = Path("data/btcusdt_5m.csv")
SUMMARY_FILE = Path("results_summary.txt")
EQUITY_PLOT = Path("equity_curve.png")


def generate_synthetic_ohlcv(rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2024-01-01", periods=rows, freq="5min")

    drift = 0.00002
    base_vol = 0.0015
    vol_noise = rng.normal(0, 0.0004, size=rows)
    dynamic_vol = np.clip(base_vol + np.convolve(vol_noise, np.ones(12) / 12, mode="same"), 0.0005, 0.006)

    returns = drift + rng.normal(0, dynamic_vol)

    close = np.empty(rows)
    close[0] = 42_000
    for i in range(1, rows):
        close[i] = max(100.0, close[i - 1] * (1 + returns[i]))

    open_ = np.empty(rows)
    open_[0] = close[0]
    open_[1:] = close[:-1]

    spread = np.maximum(close * (np.abs(rng.normal(0.0008, 0.0003, rows))), 2.0)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread

    base_volume = 600 + rng.gamma(shape=2.2, scale=110, size=rows)
    volume = base_volume * (1 + np.abs(returns) * 130)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    s = pd.Series(values)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50).to_numpy()


def bollinger_upper(values: np.ndarray, period: int = 20, stdev_mult: float = 2.0) -> np.ndarray:
    s = pd.Series(values)
    ma = s.rolling(period).mean()
    stdev = s.rolling(period).std(ddof=0)
    return (ma + stdev_mult * stdev).to_numpy()


def volume_filter(values: np.ndarray, period: int = 20) -> np.ndarray:
    s = pd.Series(values)
    vol_ma = s.rolling(period).mean()
    return (s > vol_ma).astype(int).fillna(0).to_numpy()


class BreakoutLongStrategy(Strategy):
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    sl_pct = 0.01
    tp_pct = 0.02
    size_pct = 0.10

    def init(self):
        self.upper = self.I(bollinger_upper, self.data.Close, self.bb_period, self.bb_std)
        self.rsi = self.I(rsi, self.data.Close, self.rsi_period)
        self.vol_ok = self.I(volume_filter, self.data.Volume, 20)

    def next(self):
        if len(self.data.Close) < 22:
            return

        close = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        upper = self.upper[-1]
        prev_upper = self.upper[-2]

        breakout = prev_close <= prev_upper and close > upper
        rsi_ok = self.rsi[-1] > 55
        volume_ok = self.vol_ok[-1] == 1

        if not self.position and breakout and rsi_ok and volume_ok:
            sl = close * (1 - self.sl_pct)
            tp = close * (1 + self.tp_pct)
            self.buy(size=self.size_pct, sl=sl, tp=tp)


def save_results(stats: pd.Series, initial_cash: float):
    final_cash = float(stats["Equity Final [$]"])
    pnl_total = final_cash - initial_cash

    summary = [
        f"Capital initial: {initial_cash:.2f}",
        f"Capital final: {final_cash:.2f}",
        f"Nombre de trades: {int(stats['# Trades'])}",
        f"Winrate (%): {float(stats['Win Rate [%]']):.2f}",
        f"Profit factor: {float(stats['Profit Factor']):.4f}",
        f"Expectancy: {float(stats['Expectancy [%]']):.4f}%",
        f"Max drawdown: {float(stats['Max. Drawdown [%]']):.2f}%",
        f"PnL total: {pnl_total:.2f}",
    ]
    SUMMARY_FILE.write_text("\n".join(summary) + "\n", encoding="utf-8")

    equity_curve = stats["_equity_curve"]
    plt.figure(figsize=(11, 5))
    plt.plot(equity_curve.index, equity_curve["Equity"], label="Equity")
    plt.title("Backtest Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(EQUITY_PLOT, dpi=150)
    plt.close()


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    generated = generate_synthetic_ohlcv(rows=10_000)
    generated.to_csv(OUTPUT_CSV, index_label="Date")

    data = pd.read_csv(OUTPUT_CSV, parse_dates=["Date"], index_col="Date")

    initial_cash = 100_000.0
    bt = Backtest(data, BreakoutLongStrategy, cash=initial_cash, commission=0.0007, exclusive_orders=True)
    stats = bt.run()

    save_results(stats, initial_cash)
    print("Backtest terminé.")
    print(SUMMARY_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
