import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import time
import random
import concurrent.futures
import re
from datetime import datetime

# --- Constants ---

# "Momentum Universe" - High Beta, Liquid, & Thematic Leaders
SECTOR_DEFINITIONS = {
    "🌌 Space & Defense": [
        "PLTR", "RKLB", "LMT", "RTX", "NOC", "GD", "BA", "HII", "LDOS",
        "AXON", "VIRT", "KTOS", "AVAV", "SPCE", "ASTS", "LUNR", "SPIR", "BKSY", "MAXR", "SIDU",
        "JOBY", "ACHR", "EH", "PL", "RDW", "RCAT", "ONDS", "DPRO", "PDYN"
    ],
    "🧠 AI & Semi": [
        "NVDA", "AMD", "AVGO", "MU", "SMCI", "ARM", "TSM", "INTC", "QCOM", "TXN",
        "ADI", "KLAC", "LRCX", "AMAT", "MRVL", "ONTO", "COHR", "VRT", "ANET",
        "PSTG", "DELL", "HPE", "ORCL", "MSFT", "GOOGL", "META", "AMZN",
        "SOXL", "SMH", "USD", "TQQQ", "TECL", 'DOCN', 'IREN', 'WULF', 'CORZ', 'NBIS',
        'VST', 'CEG', 'NRG', 'GE', 'NVT', 'FIX', 'EMR', 'ETN', 'PWR', 'APH', 'GLW',
        'CRM', 'NOW', 'SNOW', 'DDOG', 'PATH', 'IONQ', 'RGTI', 'QBITS', 'QTUM', 'RXRX', 'SDGR', 'CRWD', 'PANW',
        'RBRK', 'ZS', 'MDB', 'CRWV', 'CIFR', 'APLD', 'ALAB', 'CRDO', 'NET', 'ASML', 'SKYT', 'AMKR', 'SNDK', 'WDC'
    ],
    "☢️ Energy & Resources": [
        "CCJ", "URA", "U.UN", "NXE", "DNN", "UEC", "UUUU", "LEU", "BWXT", "OKLO", "SMR",
        "COPX", "FCX", "SCCO", "HBM", "TECK", "RIO", "BHP", "VALE", 
        "XOM", "CVX", "SLB", "HAL", "OXY", "KMI", "WMB", "LNG"
    ],
    "🚜 Infra & Industry": [
        "CAT", "DE", "URI", "ETN", "PWR", "EME", "GE", "HON", "MMM", "ITW", 
        "PH", "CMI", "PCAR", "FAST", "XYL", "VMI", "GNRC"
    ],
    "🚗 Auto & EV": [
        "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM", "TM", "HMC",
        "ON", "STM", "MBLY", "QS", "ALB", "LTHM"
    ],
    "💸 FinTech & Crypto & Real Estate": [
        "COIN", "MSTR", "HOOD", "PYPL", "SQ", "AFRM", "UPST", "SOFI", "V", "MA",
        "JPM", "GS", "MS", "BAC", "C", "WFC", "BLK", "BX", "KKR", 
        "PLD", "AMT", "CCI", "O", "DLR", "EQIX", "PSA", "VICI"
    ],
    "💊 Consumer & Health & Bio": [
        "LLY", "NVO", "VRTX", "REGN", "AMGN", "GILD", "BIIB", "MRNA", "PFE", "JNJ",
        "XBI", "LABU", "COST", "WMT", "TGT", "HD", "LOW", "MCD", "SBUX", "CMG",
        "NKE", "LULU", "ONON", "DECK", "CROX", "DIS", "NFLX"
    ]
}

# Create Mapping
TICKER_TO_SECTOR = {}
for sector, tickers in SECTOR_DEFINITIONS.items():
    for t in tickers:
        TICKER_TO_SECTOR[t.upper()] = sector

STATIC_MOMENTUM_WATCHLIST = list(TICKER_TO_SECTOR.keys())

# --- Thematic ETF List (Metrics Benchmark) ---
THEMATIC_ETFS = {
    # --- 🤖 Future Tech (High Growth) ---
    "Cloud Computing (クラウド)": "CLOU",
    "Cybersecurity (サイバー)": "CIBR",
    "Robotics & AI (ロボット)": "BOTZ",
    "Semiconductors (半導体)": "SMH",
    "Genomics (ゲノム)": "GNOM",
    "Healthcare Providers (医療)": "IHF",
    "Medical Devices (医療機器)": "IHI",

    # --- 🛒 消費・トレンド (Consumer) ---
    "E-commerce (EC)": "IBUY",
    "Fintech (フィンテック)": "FINX",
    "Millennials (若者消費)": "MILN",
    "Homebuilders (住宅)": "XHB",
    
    # --- 🛡️ ディフェンシブ・マクロ (Defensive/Macro) ---
    "Healthcare (ヘルスケア全体)": "XLV",
    "Consumer Staples (必需品)": "XLP",
    "Utilities (公益)": "XLU",
    "High Dividend (高配当)": "VYM",
    "Treasury 20Y+ (米国債)": "TLT",
    "VIX Short-Term (恐怖指数)": "VIXY", 

    # --- ⛏️ コモディティ・暗号資産 (Hard Assets) ---
    "Gold (金)": "GLD",
    "Silver (銀)": "SLV",
    "Oil & Gas (石油)": "XOP",
    "Copper Miners (銅)": "COPX",
    "Bitcoin Strategy (ビットコイン)": "BITO"
}

# Extend Static List with ETFs
STATIC_MOMENTUM_WATCHLIST.extend(list(THEMATIC_ETFS.values()))

# --- Functions ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_momentum_candidates(mode="hybrid"):
    """
    Builds a 'Momentum Universe' candidates list.
    Performance Optimization: Parallelized scraping.
    Returns: List of unique ticker strings.
    """
    
    # 1. Dynamic Sources (Yahoo Finance)
    sources = [
        "https://finance.yahoo.com/markets/stocks/gainers/",
        "https://finance.yahoo.com/markets/stocks/most-active/",
        "https://finance.yahoo.com/markets/stocks/52-week-gainers/"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    all_candidates = set()
    
    # Add Static List first
    for t in STATIC_MOMENTUM_WATCHLIST:
        all_candidates.add(t)
    
    # Add ETFs
    for t in THEMATIC_ETFS.values():
        all_candidates.add(t)

    # Scrape Dynamic Movers (Parallel)
    # print("Scraping Dynamic Sources...")
    
    def fetch_source(url):
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            dfs = pd.read_html(StringIO(response.text))
            if dfs:
                return dfs[0]
        except Exception:
            # print(f"Source fetch failed {url}: {e}")
            return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch_source, url): url for url in sources}
        for future in concurrent.futures.as_completed(futures):
            df = future.result()
            if df is not None:
                 # Yahoo usually has 'Symbol' and 'Name'
                if 'Symbol' in df.columns:
                    # Take top 15
                    top_df = df.head(15)
                    for _, row in top_df.iterrows():
                        sym = str(row['Symbol']).split()[0]
                        all_candidates.add(sym)

    return list(all_candidates)

def calculate_momentum_metrics(tickers):
    """
    Calculates detailed metrics for the given tickers.
    Optimized for batch processing (offline).
    """
    if not tickers:
        return None, None

    # Download 1y data to calculate long-term MA and 1y return
    try:
        # Sleep slightly to be polite/safe
        time.sleep(random.uniform(1.0, 3.0))
        # Fetching for ALL candidates in one go
        # Group by ticker is safer for multi-ticker
        df = yf.download(tickers, period="1y", group_by='ticker', auto_adjust=True, progress=False, threads=False)
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return None, None

    stats_list = []
    history_dict = {}

    for t in tickers:
        try:
            # Handle df structure
            t_data = pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                if t in df.columns.levels[0]:
                    t_data = df[t]
                elif t in df.columns:
                     t_data = df[t]
            else:
                 t_data = df

            if t_data.empty: continue
            if 'Close' not in t_data.columns: continue

            t_data = t_data.dropna()
            if t_data.empty: continue
            
            # --- 1. Validation Filter (Integrated) ---
            # Check most recent data point
            current_price = t_data['Close'].iloc[-1]
            current_vol = t_data['Volume'].iloc[-1]
            
            # For Static List: We skipping filter (Assume they are valid)
            # Normalize to avoid key errors
            t_clean = t.upper()
            if t_clean not in STATIC_MOMENTUM_WATCHLIST:
                # For Dynamic: Strict Penny Filter
                if current_price < 2.0 or current_vol < 200000:
                    continue

            # --- 2. Calculations ---
            
            # Returns
            def get_ret(days):
                if len(t_data) < days: return 0.0
                return (current_price - t_data['Close'].iloc[-days]) / t_data['Close'].iloc[-days] * 100

            metrics = {}
            metrics['Ticker'] = t
            metrics['Price'] = current_price
            metrics['1d'] = get_ret(2)
            metrics['5d'] = get_ret(5)
            metrics['1mo'] = get_ret(21)
            metrics['3mo'] = get_ret(63)
            metrics['6mo'] = get_ret(126)
            
            # YTD
            current_year = t_data.index[-1].year
            ytd_data = t_data[t_data.index.year == current_year]
            if not ytd_data.empty:
                metrics['YTD'] = (current_price - ytd_data['Close'].iloc[0]) / ytd_data['Close'].iloc[0] * 100
            else:
                metrics['YTD'] = 0.0

            if len(t_data) >= 252:
                metrics['1y'] = get_ret(252)
            else:
                metrics['1y'] = (current_price - t_data['Close'].iloc[0]) / t_data['Close'].iloc[0] * 100
            
            # RVOL
            if len(t_data) > 21:
                avg_vol_20 = t_data['Volume'].iloc[-21:-1].mean()
                if pd.isna(avg_vol_20) or avg_vol_20 == 0:
                    rvol = 0
                else:
                    rvol = current_vol / avg_vol_20
            else:
                rvol = 0
            metrics['RVOL'] = rvol
            
            # Technicals
            if len(t_data) >= 50:
                sma50 = t_data['Close'].rolling(window=50).mean().iloc[-1]
                metrics['Above_SMA50'] = current_price > sma50
            else:
                metrics['Above_SMA50'] = False
            
            rsi_series = calculate_rsi(t_data['Close'], 14)
            metrics['RSI'] = rsi_series.iloc[-1] if not rsi_series.empty else 50
            
            # Signals
            signals = []
            if metrics['RVOL'] > 2.0: signals.append('⚡')
            if metrics['Above_SMA50'] and metrics['3mo'] > 0: signals.append('🐂')
            
            # 🛒 Dip Buy: Uptrend (Above SMA50) but Short-term cool (RSI < 45)
            if metrics['Above_SMA50'] and metrics['RSI'] < 45: signals.append('🛒')

            # 🐻 Bear Trend: Downtrend (Below SMA50) & Negative Mom
            if not metrics['Above_SMA50'] and metrics['3mo'] < 0: signals.append('🐻')

            if metrics['RSI'] > 70: signals.append('🔥')
            if metrics['RSI'] < 30: signals.append('🧊')
            metrics['Signal'] = "".join(signals)
            
            stats_list.append(metrics)
            
            # Save history
            norm_hist = (t_data['Close'] / t_data['Close'].iloc[0]) * 100
            history_dict[t] = norm_hist

        except Exception as e:
            # print(f"Error calc {t}: {e}")
            continue
            
    if not stats_list:
        return None, None
        
    df_metrics = pd.DataFrame(stats_list)
    cols = ['Ticker', 'Signal', 'Price', '1d', '5d', '1mo', '3mo', '6mo', 'YTD', '1y', 'RVOL', 'RSI']
    
    # Allow for flexible columns if something missing
    valid_cols = [c for c in cols if c in df_metrics.columns]
    df_metrics = df_metrics[valid_cols]
    
    # Just in case extra cols are needed for heatmap (Above_SMA50 etc) - no, existing code uses these via merge logic or re-calc?
    # Actually app.py uses df_metrics directly.
    # Note: `Above_SMA50` was used in `app.py` heatmap logic? 
    # Let's check app.py. The user code in prompt shows `cols = ['Ticker', ...]` filtering out others. 
    # If app.py needs `Above_SMA50` for sorting/filtering later, we might need to include it.
    # But sticking to user request for now.
    
    return df_metrics, history_dict
