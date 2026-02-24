import os
import requests
import yfinance as yf
import pandas as pd
import logging
from dotenv import load_dotenv

load_dotenv()
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ==============================
# Einstellungen für 1m Indizes
# ==============================

HISTORY_PERIOD = "5d"    # 1m Daten nur wenige Tage verfügbar [web:32][web:52]
INTERVAL       = "1m"
MA_TYPE        = "SMA"
MA_LENGTH      = 50
MIN_BARS       = 100
NEAR_PCT       = 0.5     # engerer Near-Touch-Filter für 1m
FUTURES = {
    "ES=F":  "S&P 500 Future",        # E‑Mini S&P 500
    "NQ=F":  "Nasdaq 100 Future",     # E‑Mini Nasdaq‑100
    "YM=F":  "Dow Jones Future",      # E‑Mini Dow (YM) [web:61]
    "RTY=F": "Russell 2000 Future",   # E‑Mini Russell 2000 (RTY) [web:61]
}


# ==============================
# MA-Berechnung
# ==============================

def ma_pine_style(close: pd.Series, ma_type: str, length: int) -> pd.Series:
    ma_type = ma_type.upper()
    if ma_type == "SMA":
        return close.rolling(length, min_periods=length).mean()
    elif ma_type == "EMA":
        return close.ewm(span=length, adjust=False, min_periods=length).mean()
    else:
        return close.rolling(length, min_periods=length).mean()


# ==============================
# Bars Since MA Touch/Cross
# ==============================

def bars_since_ma_event(df, ma_type="SMA", length=50):
    df = df.copy()
    ma = ma_pine_style(df["Close"], ma_type, length)
    df["ma"] = ma

    touch      = (df["Low"] <= df["ma"]) & (df["ma"] <= df["High"])
    close_prev = df["Close"].shift(1)
    ma_prev    = df["ma"].shift(1)
    crossover  = (close_prev < ma_prev) & (df["Close"] >= df["ma"])
    crossunder = (close_prev > ma_prev) & (df["Close"] <= df["ma"])

    touch_or_cross = touch | crossover | crossunder

    bars_since = []
    last = None
    for is_event in touch_or_cross:
        if pd.isna(is_event):
            last = None
            bars_since.append(None)
        else:
            last = 0 if (is_event or last is None) else last + 1
            bars_since.append(last)

    df["bars_since"] = bars_since
    return df["bars_since"].iloc[-1]


# ==============================
# Near Touch + Abstand
# ==============================

def get_near_touch(df, ma_type="SMA", length=50, near_pct=1.0):
    df = df.copy()
    ma = ma_pine_style(df["Close"], ma_type, length)

    last_close   = df["Close"].iloc[-1]
    last_ma      = ma.iloc[-1]
    distance_pct = abs(last_close - last_ma) / last_ma * 100

    signal = "⚠️ Near Touch" if distance_pct <= near_pct else "-"
    return signal, round(distance_pct, 3)


# ==============================
# Screener 1m nur Indizes
# ==============================

def run_screener_1m_indices() -> pd.DataFrame:
    ticker_index_map = FUTURES  # nur ES=F und NQ=F

    data = yf.download(
        tickers=list(ticker_index_map.keys()),
        period=HISTORY_PERIOD,
        interval=INTERVAL,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False
    )

    results = []

    for ticker, index_name in ticker_index_map.items():
        try:
            df_ticker    = data[ticker].dropna()[["Close", "High", "Low"]]
            bars         = bars_since_ma_event(df_ticker, MA_TYPE, MA_LENGTH)
            signal, dist = get_near_touch(df_ticker, MA_TYPE, MA_LENGTH, NEAR_PCT)

            if bars is not None and bars > MIN_BARS:
                results.append({
                    "Ticker":              ticker,
                    "Index":               index_name,
                    "Bars_since_MA_event": int(bars),
                    "Signal":              signal,
                    "Abstand_%":           dist
                })
        except Exception:
            pass

    if not results:
        return pd.DataFrame(columns=[
            "Ticker", "Index", "Bars_since_MA_event", "Signal", "Abstand_%"
        ])

    return pd.DataFrame(results).sort_values(
        "Bars_since_MA_event",
        ascending=False
    ).reset_index(drop=True)


# ==============================
# Discord-Tabelle
# ==============================

def format_discord_message(df: pd.DataFrame) -> str:
    if df.empty:
        return f"⏱️ SMA{MA_LENGTH} 1m Index-Screener: Keine Futures mit > {MIN_BARS} Bars."

    header = [
        f"⏱️ **1m SMA{MA_LENGTH} Index-Screener – ES & NQ** (> {MIN_BARS} Bars | Near Touch ≤{NEAR_PCT}%)",
        "```",
        f"{'#':<4} {'Ticker':<8} {'Index':<20} {'Bars':>5}  {'Signal':<15} {'Abstand':>8}",
        f"{'-'*65}"
    ]

    rows = []
    for i, row in df.iterrows():
        rows.append(
            f"{i+1:<4} {row['Ticker']:<8} {row['Index']:<20} "
            f"{row['Bars_since_MA_event']:>5}  "
            f"{row['Signal']:<15} {row['Abstand_%']:>7}%"
        )

    footer = ["```"]

    message = "\n".join(header + rows + footer)
    return message


# ==============================
# Discord-Webhook
# ==============================

def post_to_discord_1m(content: str):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL_1M")
    if not webhook_url:
        raise RuntimeError("DISCORD_WEBHOOK_URL_1M not set")
    requests.post(webhook_url, json={"content": content}).raise_for_status()


# ==============================
# Entry Point
# ==============================

if __name__ == "__main__":
    df = run_screener_1m_indices()

    pd.set_option("display.max_rows", None)
    print(df.to_string(index=False))

    message = format_discord_message(df)
    post_to_discord_1m(message)
