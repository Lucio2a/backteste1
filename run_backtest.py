import ccxt
import numpy as np
import pandas as pd


SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
INITIAL_BALANCE = 10_000.0
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0


def fetch_data(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        raise RuntimeError("No OHLCV data returned from Binance.")

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def run_strategy(df: pd.DataFrame) -> dict:
    close = df["close"]
    middle = close.rolling(BOLLINGER_PERIOD).mean()
    std = close.rolling(BOLLINGER_PERIOD).std(ddof=0)
    upper = middle + (BOLLINGER_STD * std)

    cash = INITIAL_BALANCE
    position = 0.0
    entry_price = np.nan
    total_trades = 0
    winning_trades = 0

    for i in range(1, len(df)):
        if np.isnan(middle.iloc[i]) or np.isnan(upper.iloc[i]):
            continue

        current_close = close.iloc[i]
        prev_close = close.iloc[i - 1]
        current_upper = upper.iloc[i]
        prev_upper = upper.iloc[i - 1]
        current_middle = middle.iloc[i]
        prev_middle = middle.iloc[i - 1]

        breakout_up = prev_close <= prev_upper and current_close > current_upper
        back_below_mid = prev_close >= prev_middle and current_close < current_middle

        if position == 0 and breakout_up:
            position = cash / current_close
            cash = 0.0
            entry_price = current_close
        elif position > 0 and back_below_mid:
            exit_price = current_close
            cash = position * exit_price
            pnl = exit_price - entry_price
            total_trades += 1
            if pnl > 0:
                winning_trades += 1
            position = 0.0
            entry_price = np.nan

    final_balance = cash if position == 0 else position * close.iloc[-1]
    winrate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    return {
        "final_balance": round(float(final_balance), 2),
        "total_trades": int(total_trades),
        "winrate_percent": round(float(winrate), 2),
    }


def main() -> None:
    data = fetch_data(SYMBOL, TIMEFRAME)
    results = run_strategy(data)
    pd.DataFrame([results]).to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
