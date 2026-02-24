import os
import requests
import yfinance as yf
import pandas as pd
import logging
from dotenv import load_dotenv

load_dotenv()
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# =====================================================
# Universen / Märkte
# =====================================================

UNIVERSE_US   = "US"
UNIVERSE_DE   = "DE"
UNIVERSE_ASIA = "ASIA"

# =====================================================
# Einstellungen
# =====================================================

HISTORY_PERIOD = "2y"
INTERVAL       = "1d"
MA_TYPE        = "SMA"
MA_LENGTH      = 50
MIN_BARS       = 80
NEAR_PCT       = 1.0   # Near Touch Schwelle in %

# =====================================================
# Hilfsfunktion: sichere URL-Ladung
# =====================================================

def safe_fetch_csv(url: str, timeout: int = 15) -> pd.DataFrame | None:
    """CSV per requests laden mit Timeout + User-Agent."""
    try:
        from io import StringIO
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return pd.read_csv(StringIO(resp.text))
    except Exception as e:
        print(f"[WARN] CSV-Laden fehlgeschlagen: {url} | {e}")
        return None


def safe_read_html(url: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(url)
    except Exception as e:
        print(f"[WARN] Konnte HTML nicht laden: {url} | {e}")
        return []


# =====================================================
# Index-Loader mit try/except
# =====================================================

def get_sp500_tickers():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = safe_fetch_csv(url)
    if df is None or "Symbol" not in df.columns:
        return []
    tickers = df["Symbol"].astype(str).tolist()
    return [t.replace(".", "-") for t in tickers]


def get_nasdaq100_tickers():
    url = "https://github.com/azymnis/python-stocks/raw/master/nasdaq100.csv"
    df = safe_fetch_csv(url)
    if df is None:
        return []
    col = "Ticker" if "Ticker" in df.columns else df.columns[0]
    return df[col].astype(str).tolist()


def get_dax40_tickers():
    return [
        "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE",
        "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE", "CON.DE",
        "DTG.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE",
        "EOAN.DE", "FME.DE", "FRE.DE", "G1A.DE", "HNR1.DE",
        "HEI.DE", "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE",
        "MTX.DE", "MUV2.DE", "P911.DE", "PAH3.DE", "QGEN",
        "RHM.DE", "RWE.DE", "SAP.DE", "SIE.DE", "SHL.DE",
        "SRT3.DE", "SY1.DE", "VNA.DE", "VOW3.DE", "ZAL.DE",
    ]

def get_nifty500_tickers():
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    df = safe_fetch_csv(url)
    if df is None or "Symbol" not in df.columns:
        print("[WARN] Nifty 500 nicht geladen – Fallback auf leere Liste")
        return []
    symbols = df["Symbol"].dropna().astype(str).tolist()
    # yfinance braucht .NS Suffix für NSE-Ticker
    return [s + ".NS" for s in symbols]


def get_mdax_tickers():
    # Platzhalter – mit echter Quelle auffüllen, falls gewünscht
    return []


def get_tecdax_tickers():
    # Platzhalter – mit echter Quelle auffüllen
    return []


def get_nifty50_tickers():
    return [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BHARTIARTL.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DIVISLAB.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "IOC.NS",
    "ITC.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHREECEM.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
]

def get_hscei_tickers():
    return [
        "0700.HK",   # Tencent
        "9988.HK",   # Alibaba
        "0939.HK",   # China Construction Bank
        "1398.HK",   # ICBC
        "1288.HK",   # Agricultural Bank of China
        "3988.HK",   # Bank of China
        "0941.HK",   # China Mobile
        "2318.HK",   # Ping An Insurance
        "0857.HK",   # PetroChina
        "2628.HK",   # China Life Insurance
        "3690.HK",   # Meituan
        "9618.HK",   # JD.com
        "1810.HK",   # Xiaomi
        "0883.HK",   # CNOOC
        "1211.HK",   # BYD
        "3968.HK",   # China Merchants Bank
        "1024.HK",   # Kuaishou
        "2331.HK",   # Li-Ning
        "1109.HK",   # China Resources Land
        "2020.HK",   # Anta Sports
        "0175.HK",   # Geely
        "9633.HK",   # Nongfu Spring
        "2688.HK",   # ENN Energy
        "0386.HK",   # Sinopec
        "9999.HK",   # NetEase
        "0688.HK",   # China Overseas Land
        "2382.HK",   # Sunny Optical
        "1093.HK",   # Shijiazhuang Pharma
        "0981.HK",   # SMIC
        "2313.HK",   # Shenzhou International
        "9888.HK",   # Baidu
        "6618.HK",   # JD Health
        "0267.HK",   # CITIC
        "0968.HK",   # Xinyi Solar
        "1658.HK",   # Postal Savings Bank
        "0992.HK",   # Lenovo
        "1177.HK",   # Sino Biopharmaceutical
        "0960.HK",   # Longfor Properties
        "1801.HK",   # Innovent Biologics
        "2601.HK",   # China Pacific Insurance
        "3328.HK",   # Bank of Communications
        "0384.HK",   # China Gas
        "6186.HK",   # China Feihe
        "2618.HK",   # JD Logistics
        "6098.HK",   # CG Services
        "6862.HK",   # Haidilao
        "0241.HK",   # Alibaba Health
        "9626.HK",   # Bilibili
        "2423.HK",   # KE Holdings (neu ab März 2026)
        "9660.HK",   # Horizon Robotics (neu ab März 2026)
    ]



def get_nikkei225_tickers():
    return [
        "7203.T",   # Toyota
        "6758.T",   # Sony
        "6861.T",   # Keyence
        "8306.T",   # Mitsubishi UFJ
        "9984.T",   # SoftBank Group
        "6098.T",   # Recruit Holdings
        "9432.T",   # NTT
        "8035.T",   # Tokyo Electron
        "4063.T",   # Shin-Etsu Chemical
        "6902.T",   # Denso
        "6501.T",   # Hitachi
        "7741.T",   # HOYA
        "4519.T",   # Chugai Pharma
        "6954.T",   # Fanuc
        "7974.T",   # Nintendo
        "9433.T",   # KDDI
        "6367.T",   # Daikin
        "8316.T",   # Sumitomo Mitsui FG
        "4502.T",   # Takeda
        "6981.T",   # Murata
        "8058.T",   # Mitsubishi Corp
        "7267.T",   # Honda
        "6594.T",   # Nidec
        "4568.T",   # Daiichi Sankyo
        "8031.T",   # Mitsui & Co
        "8766.T",   # Tokio Marine
        "4661.T",   # Oriental Land
        "6971.T",   # Kyocera
        "7832.T",   # Bandai Namco
        "3382.T",   # Seven & i Holdings
        "4543.T",   # Terumo
        "8001.T",   # Itochu
        "6762.T",   # TDK
        "6503.T",   # Mitsubishi Electric
        "9983.T",   # Fast Retailing (Uniqlo)
        "2802.T",   # Ajinomoto
        "4901.T",   # Fujifilm
        "7751.T",   # Canon
        "5108.T",   # Bridgestone
        "8411.T",   # Mizuho FG
        "6301.T",   # Komatsu
        "7269.T",   # Suzuki Motor
        "4578.T",   # Otsuka Holdings
        "6857.T",   # Advantest
        "8802.T",   # Mitsubishi Estate
        "9531.T",   # Tokyo Gas
        "2914.T",   # Japan Tobacco
        "4452.T",   # Kao
        "6326.T",   # Kubota
        "8053.T",   # Sumitomo Corp
    ]



def get_hscei_tickers():
    # HSCEI-Komponenten von TradingView (HTML-Tabelle)
    url = "https://www.tradingview.com/symbols/HSI-HSCEI/components/"
    tables = safe_read_html(url)
    if not tables:
        return []
    df = tables[0]
    # Spalte mit Ticker/Code finden
    col = None
    for c in df.columns:
        if "Symbol" in str(c) or "Ticker" in str(c) or "Code" in str(c):
            col = c
            break
    if col is None:
        return []
    raw = df[col].astype(str).tolist()
    # TradingView-Symbole sind oft wie "00700", "00939" – für yfinance ".HK"
    tickers = []
    for s in raw:
        s_clean = s.strip()
        if s_clean.endswith(".HK"):
            tickers.append(s_clean)
        elif s_clean.isdigit():
            tickers.append(s_clean.zfill(5) + ".HK")
        else:
            tickers.append(s_clean)
    return tickers


# =====================================================
# Universen bauen: {Ticker: Universe}
# =====================================================

def build_universes():
    tickers_us   = set(get_sp500_tickers())   | set(get_nasdaq100_tickers())
    tickers_de   = set(get_dax40_tickers())   | set(get_mdax_tickers()) | set(get_tecdax_tickers())
    tickers_asia = set(get_nifty500_tickers()) | set(get_nikkei225_tickers()) | set(get_hscei_tickers())

    print(f"US:   {len(tickers_us)} Ticker")
    print(f"DE:   {len(tickers_de)} Ticker")
    print(f"ASIA: {len(tickers_asia)} Ticker")

    universe_map: dict[str, str] = {}

    for t in tickers_us:
        universe_map[t] = UNIVERSE_US
    for t in tickers_de:
        universe_map[t] = UNIVERSE_DE
    for t in tickers_asia:
        universe_map[t] = UNIVERSE_ASIA

    return universe_map  # {ticker: "US"/"DE"/"ASIA"}


# =====================================================
# MA-Berechnung
# =====================================================

def ma_pine_style(close: pd.Series, ma_type: str, length: int) -> pd.Series:
    ma_type = ma_type.upper()
    if ma_type == "SMA":
        return close.rolling(length, min_periods=length).mean()
    elif ma_type == "EMA":
        return close.ewm(span=length, adjust=False, min_periods=length).mean()
    elif ma_type == "WMA":
        return close.rolling(length, min_periods=length).apply(
            lambda x: (x * pd.Series(range(1, len(x) + 1), index=x.index)).sum()
                      / pd.Series(range(1, len(x) + 1), index=x.index).sum(),
            raw=False
        )
    elif ma_type == "HMA":
        import numpy as np
        def wma(s, l):
            return s.rolling(l, min_periods=l).apply(
                lambda x: (x * pd.Series(range(1, len(x)+1), index=x.index)).sum()
                          / pd.Series(range(1, len(x)+1), index=x.index).sum(),
                raw=False
            )
        n = length
        return wma(2 * wma(close, int(n/2)) - wma(close, n), int(np.sqrt(n)))
    else:
        return pd.Series(index=close.index, dtype=float)


# =====================================================
# Bars Since MA Touch/Cross
# =====================================================

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


# =====================================================
# Near Touch + Abstand
# =====================================================

def get_near_touch(df, ma_type="SMA", length=50, near_pct=1.0):
    df = df.copy()
    ma = ma_pine_style(df["Close"], ma_type, length)

    last_close   = df["Close"].iloc[-1]
    last_ma      = ma.iloc[-1]
    distance_pct = abs(last_close - last_ma) / last_ma * 100

    signal = "⚠️ Near Touch" if distance_pct <= near_pct else "-"
    return signal, round(distance_pct, 2)


# =====================================================
# Screener
# =====================================================

def run_screener() -> pd.DataFrame:
    universe_map = build_universes()
    all_tickers  = list(universe_map.keys())

    print(f"Gesamt-Universum: {len(all_tickers)} Ticker")

    if not all_tickers:
        return pd.DataFrame(columns=["Ticker", "Universe", "Bars_since_MA_event", "Signal", "Abstand_%"])

    data = yf.download(
        tickers=all_tickers,
        period=HISTORY_PERIOD,
        interval=INTERVAL,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False
    )

    results = []

    for ticker in all_tickers:
        try:
            df_ticker = data[ticker].dropna()[["Close", "High", "Low"]]
            if df_ticker.empty:
                continue

            bars = bars_since_ma_event(df_ticker, MA_TYPE, MA_LENGTH)
            signal, dist = get_near_touch(df_ticker, MA_TYPE, MA_LENGTH, NEAR_PCT)

            if bars is not None and bars > MIN_BARS:
                results.append({
                    "Ticker":              ticker,
                    "Universe":            universe_map[ticker],  # US / DE / ASIA
                    "Bars_since_MA_event": int(bars),
                    "Signal":              signal,
                    "Abstand_%":           dist
                })
        except Exception as e:
            print(f"[WARN] Fehler bei {ticker}: {e}")
            continue

    if not results:
        return pd.DataFrame(columns=["Ticker", "Universe", "Bars_since_MA_event", "Signal", "Abstand_%"])

    df = pd.DataFrame(results).sort_values(
        ["Universe", "Bars_since_MA_event"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return df


# =====================================================
# Discord – drei Tabellen (US / DE / ASIA)
# =====================================================

def format_discord_message(df: pd.DataFrame) -> str:
    if df.empty:
        return f"SMA{MA_LENGTH} Screener: Keine Ergebnisse (> {MIN_BARS} Bars)."

    parts = []

    universe_titles = [
        (UNIVERSE_US,   "US"),
        (UNIVERSE_DE,   "DE"),
        (UNIVERSE_ASIA, "ASIA"),
    ]

    for uni, title in universe_titles:
        sub = df[df["Universe"] == uni]
        if sub.empty:
            continue

        sub = sub.reset_index(drop=True)

        lines = [
            f"🌍 {title} – SMA{MA_LENGTH} (> {MIN_BARS} Bars | Near Touch ≤{NEAR_PCT}%)",
            "```",
            f"{'#':<4} {'Ticker':<10} {'Bars':>5}  {'Signal':<15} {'Abstand':>7}",
            f"{'-'*50}"
        ]
        for i, row in sub.iterrows():
            lines.append(
                f"{i+1:<4} {row['Ticker']:<10} "
                f"{row['Bars_since_MA_event']:>5}  "
                f"{row['Signal']:<15} {row['Abstand_%']:>6}%"
            )
        lines.append("```")
        parts.append("\n".join(lines))

    if not parts:
        return f"SMA{MA_LENGTH} Screener: Keine Ergebnisse."

    message = "\n\n".join(parts)

    if len(message) > 1900:
        message = message[:1850] + "\n... (gekürzt)"

    return message


# =====================================================
# Discord-Webhook
# =====================================================

def post_to_discord(content: str):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL_1D")
    if not webhook_url:
        raise RuntimeError("DISCORD_WEBHOOK_URL_1D")
    resp = requests.post(webhook_url, json={"content": content})
    resp.raise_for_status()


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":
    df = run_screener()

    pd.set_option("display.max_rows", None)
    print(df.to_string(index=False))

    msg = format_discord_message(df)
    post_to_discord(msg)
