import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import StringIO
import random
from datetime import datetime, date
import plotly.express as px
import re

# Page Config (Must be first Streamlit command)
st.set_page_config(
    page_title="Market Correlation Radar",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Market Correlation Radar\nPowered by AI Analyst"
    }
)

# --- Risk Management Helpers ---
def get_ticker_news(ticker, company_name=None):
    """
    Fetches top 3 news.
    Filters:
    1. Valid Title (Not empty)
    2. Recency (< 3 days)
    3. Relevance (Title must contain Ticker or Company Name)
    """
    try:
        news = yf.Ticker(ticker).news
        if not news: return []
        
        results = []
        now = datetime.now()
        
        # Prepare Regex for Ticker (Case-insensitive word boundary? No, Ticker usually CAPS, but let's be flexible)
        # Actually for Ticker, Case Sensitive is safer for short ones like 'BE' vs 'be'.
        # But some titles might lower case? "Bloom Energy (be) ..." Unlikely.
        # Let's simple check: 
        # 1. Ticker (Case Sensitive) in Title (Word Bound)
        # 2. Company Name (First Word) in Title (Case Insensitive)
        
        patterns = [r'\b{}\b'.format(re.escape(ticker))] # Exact Ticker Match
        
        if company_name:
            # Clean name: "Bloom Energy Corporation" -> "Bloom"
            # "NVIDIA Corp" -> "NVIDIA"
            # "Advanced Micro Devices" -> "Advanced" (Risk? "Advanced" is common word)
            # Maybe use full string up to common suffixes?
            
            # Simple heuristic: Split by space
            parts = company_name.split()
            if parts:
                main_name = parts[0]
                # If short basic word, maybe skip? But let's trust it for now.
                # Avoid very short words if they are not the ticker
                if len(main_name) > 2:
                    patterns.append(r'\b{}\b'.format(re.escape(main_name)))
                
                # Also try full name string (e.g. "Bloom Energy")
                if len(parts) > 1:
                     patterns.append(re.escape(company_name))

        for n in news:
            # 1. Normalize Logic
            content = n.get('content', n)
            title = content.get('title', '')
            
            if not title or title == "No Title":
                continue

            # --- SUBJECT FILTER ---
            # Check if any pattern matches title
            is_relevant = False
            for pat in patterns:
                if re.search(pat, title, re.IGNORECASE):
                    is_relevant = True
                    break
            
            if not is_relevant:
                # Debug print? No.
                continue
            # ----------------------

            # 2. Time Extraction
            pub_time = None
            if 'pubDate' in content:
                try:
                    ts_str = content['pubDate'].replace('Z', '')
                    pub_time = datetime.fromisoformat(ts_str)
                except: pass
            
            if not pub_time and 'providerPublishTime' in n:
                try:
                    pub_time = datetime.fromtimestamp(n['providerPublishTime'])
                except: pass
                    
            if not pub_time:
                pub_time = now # Skip if unknown? Or assume recent?
                # Let's skip to be strict
                continue

            # 3. Filter: Within 3 days
            if (now - pub_time).days > 3:
                continue

            dt = pub_time.strftime('%Y-%m-%d %H:%M')
            
            # 4. Link Extraction
            link = content.get('clickThroughUrl')
            if not link:
                link = content.get('link')
                if isinstance(link, dict): link = link.get('url')
                
            if not link: link = "#"

            results.append({
                'title': title,
                'publisher': content.get('publisher', 'Unknown'),
                'link': link,
                'time': dt
            })
            
            if len(results) >= 3:
                break
                
        return results
    except Exception as e:
        return []
STATIC_MENU_ITEMS = [
    "--- 🌏 指数・為替・債券 (Indices/Forex/Bonds) ---",
    'USDJPY=X', '^TNX', 'BTC-USD', 'GLD',
    "--- 💻 米国株：AI・ハイテク (US Tech/AI) ---",
    'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'AAPL', 'META', 'AMD', 'PLTR', 'AVGO',
    "--- 📊 米国ETF：セクター (US Sector ETFs) ---",
    'QQQ', 'SPY', 'SMH', 'VGT', 'XLV', 'XLP', 'XLE', 'XLF',
    "--- 🚀 テーマ別ETF (Thematic ETFs) ---",
    'URA', 'COPX', 'QTUM', 'ARKX', 'NLR'
]

# ... (rest of constants stays same until end of lists) ...

# --- Hardcoded Momentum Watchlist (Categorized) ---
# "Momentum Universe" - High Beta, Liquid, & Thematic Leaders
SECTOR_DEFINITIONS = {
   "Space & Defense": [
        "PLTR", "RKLB", "LMT", "RTX", "NOC", "GD", "BA", "HII", "LDOS",
        "AXON", "VIRT", "KTOS", "AVAV", "SPCE", "ASTS", "Lunr" 
   ],
   "AI & Semi (High Beta)": [
        "NVDA", "AMD", "AVGO", "MU", "SMCI", "ARM", "TSM", "INTC", "QCOM", "TXN",
        "ADI", "KLAC", "LRCX", "AMAT", "MRVL", "ONTO", "COHR", "VRT", "ANET",
        "PSTG", "DELL", "HPE", "ORCL", "MSFT", "GOOGL", "META", "AMZN",
        "SOXL", "SMH", "USD", "TQQQ", "TECL" 
   ],
   "Energy & Resources (Nuclear/Uranium/Copper)": [
        "CCJ", "URA", "U.UN", "NXE", "DNN", "UEC", "UUUU", "LEU", "BWXT", "OKLO", "SMR",
        "COPX", "FCX", "SCCO", "HBM", "TECK", "RIO", "BHP", "VALE", 
        "XOM", "CVX", "SLB", "HAL", "OXY", "KMI", "WMB", "LNG"
   ],
   "Infra & Industry (Caterpillar/Deere/Grid)": [
        "CAT", "DE", "URI", "ETN", "PWR", "EME", "GE", "HON", "MMM", "ITW", 
        "PH", "CMI", "PCAR", "FAST", "XYL", "VMI", "GNRC"
   ],
   "Auto & EV (Tesla/Rivian)": [
        "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM", "TM", "HMC",
        "ON", "STM", "MBLY", "QS", "ALB", "LTHM"
   ],
   "FinTech & Crypto & Real Estate": [
        "COIN", "MSTR", "HOOD", "PYPL", "SQ", "AFRM", "UPST", "SOFI", "V", "MA",
        "JPM", "GS", "MS", "BAC", "C", "WFC", "BLK", "BX", "KKR", 
        "PLD", "AMT", "CCI", "O", "DLR", "EQIX", "PSA", "VICI"
   ],
   "Consumer & Health & Bio": [
        "LLY", "NVO", "VRTX", "REGN", "AMGN", "GILD", "BIIB", "MRNA", "PFE", "JNJ",
        "XBI", "LABU", "COST", "WMT", "TGT", "HD", "LOW", "MCD", "SBUX", "CMG",
        "NKE", "LULU", "ONON", "DECK", "CROX", "DIS", "NFLX"
   ]
}

# --- Thematic ETF List (Metrics Benchmark) ---
THEMATIC_ETFS = {
    # --- 🤖 最先端ハイテク (Frontier Tech) ---
    "Semiconductors (半導体)": "SMH",
    "AI & Robotics (ロボ/AI)": "BOTZ",
    "Cybersecurity (サイバー)": "CIBR",
    "Cloud Computing (クラウド)": "CLOU",
    "Quantum (量子/次世代)": "QTUM",
    "Blockchain (ブロックチェーン)": "BLOK",
    "Metaverse/Gaming (ゲーム)": "HERO",

    # --- 🏭 インフラ・エネルギー (Physical World) ---
    "Nuclear/Uranium (原子力/ウラン)": "URA",
    "Data Center/Infra (DC/建設)": "SRVR", 
    "US Infrastructure (インフラ)": "PAVE",
    "Grid & Power (電力網)": "GRID",
    "Clean Energy (クリエネ)": "ICLN",
    "Water Resources (水資源)": "PHO",
    "Aerospace & Defense (防衛)": "ITA",
    "Space Exploration (宇宙)": "ARKX",

    # --- 🧬 ヘルスケア・バイオ (Life Science) ---
    "Biotech (バイオ)": "XBI",
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

# --- Risk Management Helpers ---
def get_earnings_next(ticker):
    """
    Fetches the next earnings date.
    Returns: formatted string (e.g., '⚠️ In 3 days' or '2025-10-30') or '-'
    """
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        
        # Handle dictionary return (newer yfinance)
        if isinstance(cal, dict):
            # Key varies: 'Earnings Date' or 'Earnings High' etc.
            # Usually 'Earnings Date' is a list of dates
            dates = cal.get('Earnings Date', [])
            if not dates:
                return "-"
            next_date = dates[0] # Take the first one
        
        # Handle DataFrame return (older yfinance)
        elif isinstance(cal, pd.DataFrame):
            if cal.empty: return "-"
            # Often index is 0, 1... and columns are dates
            # Or formatted differently. Let's try to grab the first date available.
            # This part is tricky without exact dataframe structure from recent fix, 
            # but usually it finds 'Earnings Date' in dict form now.
            return "-" 
        else:
            return "-"

        # Calculate days until
        if isinstance(next_date, (datetime, date)):
            d = next_date.date() if isinstance(next_date, datetime) else next_date
            today = date.today()
            delta = (d - today).days
            
            if 0 <= delta <= 7:
                return f"⚠️ In {delta} days"
            elif delta < 0:
                # Past earnings (sometimes API returns previous)
                return "-"
            else:
                return d.strftime("%Y-%m-%d")
        return "-"
    except:
        return "-"

def get_ticker_news(ticker, company_name=None):
    """
    Fetches top 3 news.
    Filters:
    1. Valid Title (Not empty)
    2. Recency (< 3 days)
    3. Relevance (Title must contain Ticker or Company Name)
    """
    try:
        news = yf.Ticker(ticker).news
        if not news: return []
        
        results = []
        now = datetime.now()
        
        # Prepare Regex for Ticker (Case-insensitive word boundary? No, Ticker usually CAPS, but let's be flexible)
        # Actually for Ticker, Case Sensitive is safer for short ones like 'BE' vs 'be'.
        # But some titles might lower case? "Bloom Energy (be) ..." Unlikely.
        # Let's simple check: 
        # 1. Ticker (Case Sensitive) in Title (Word Bound)
        # 2. Company Name (First Word) in Title (Case Insensitive)
        
        patterns = [r'\b{}\b'.format(re.escape(ticker))] # Exact Ticker Match
        
        if company_name:
            # Clean name: "Bloom Energy Corporation" -> "Bloom"
            # "NVIDIA Corp" -> "NVIDIA"
            # "Advanced Micro Devices" -> "Advanced" (Risk? "Advanced" is common word)
            # Maybe use full string up to common suffixes?
            
            # Simple heuristic: Split by space
            parts = company_name.split()
            if parts:
                main_name = parts[0]
                # If short basic word, maybe skip? But let's trust it for now.
                # Avoid very short words if they are not the ticker
                if len(main_name) > 2:
                    patterns.append(r'\b{}\b'.format(re.escape(main_name)))
                
                # Also try full name string (e.g. "Bloom Energy")
                if len(parts) > 1:
                     patterns.append(re.escape(company_name))

        for n in news:
            # 1. Normalize Logic (Handle New vs Old API)
            # New API has nested 'content' dict
            content = n.get('content', n) # Fallback to n if content missing
            
            title = content.get('title', '')
            if not title or title == "No Title":
                continue

            # --- SUBJECT FILTER ---
            # Check if any pattern matches title
            is_relevant = False
            for pat in patterns:
                if re.search(pat, title, re.IGNORECASE):
                    is_relevant = True
                    break
            
            if not is_relevant:
                # Debug print? No.
                continue
            # ----------------------

            # 2. Time Extraction
            pub_time = None
            
            # Try ISO String (New API)
            if 'pubDate' in content:
                try:
                    # e.g., 2026-01-13T14:30:00Z - remove Z for simple parsing
                    ts_str = content['pubDate'].replace('Z', '')
                    pub_time = datetime.fromisoformat(ts_str)
                except:
                    pass
            
            # Try Timestamp (Old API)
            if not pub_time and 'providerPublishTime' in n:
                try:
                    pub_time = datetime.fromtimestamp(n['providerPublishTime'])
                except:
                    pass
                    
            # Fallback
            if not pub_time:
                pub_time = now # If unknown, assume recent? Or skip? Let's skip to be safe.
                # Let's skip to be strict
                continue

            # 3. Filter: Within 3 days
            # Simple check ignoring offset for robustness
            if (now - pub_time).days > 3:
                continue

            dt = pub_time.strftime('%Y-%m-%d %H:%M')
            
            # 4. Link Extraction
            # 'clickThroughUrl' often works better for new API
            link = content.get('clickThroughUrl')
            if not link:
                link = content.get('link') # Old API
                if isinstance(link, dict): link = link.get('url') # Sometimes dict?
                
            if not link: link = "#"

            results.append({
                'title': title,
                'publisher': content.get('publisher', 'Unknown'), # New API might not have publisher
                'link': link,
                'time': dt
            })
            
            if len(results) >= 3:
                break
                
        return results
    except Exception as e:
        # print(f"News Error: {e}") 
        return []

# --- Logic Functions: Shared / Correlation (Existing) ---
def get_data(tickers, period):
    # Parse tickers
    if isinstance(tickers, list):
        ticker_list = [t.strip() for t in tickers if t.strip()]
    else:
        # Fallback for string input
        ticker_list = [t.strip() for t in tickers.split(',') if t.strip()]
        
    if not ticker_list:
        return None
    
    try:
        data_frames = []
        for t in ticker_list:
            if t.startswith('---'): continue # Skip separators just in case
            try:
                # Fetch one by one to avoid bulk download header/cache issues
                df = yf.download(t, period=period, auto_adjust=True, progress=False)
                
                # Check if data is empty
                if df is None or df.empty:
                    continue
                    
                # Standardize column to Ticker name
                if isinstance(df, pd.DataFrame):
                    # Should have 'Close'
                    if 'Close' in df.columns:
                        df = df[['Close']]
                    
                    # Force rename columns to simple string ticker
                    df.columns = [t]
                
                data_frames.append(df)
            except Exception as e:
                st.warning(f"Failed to fetch {t}: {e}")
                continue

        if not data_frames:
            return None

        # Concatenate all
        data = pd.concat(data_frames, axis=1)
        
        # Align data: Forward fill to handle mismatching trading days
        data = data.ffill()
        
        # Drop only if data is still missing (e.g. leading NaNs)
        aligned_data = data.dropna()
        
        return aligned_data
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return None

def calculate_stats(df_prices):
    """
    Calculates daily returns, correlation matrix, and cumulative returns.
    """
    if df_prices is None or df_prices.empty:
        return None, None, None
        
    # 1. Daily Returns (for Correlation)
    returns = df_prices.pct_change().dropna()
    
    # 2. Correlation Matrix
    corr_matrix = returns.corr()
    
    # 3. Cumulative Returns (for Performance Chart)
    # Rebase to 0%
    cumulative_returns = (df_prices / df_prices.iloc[0]) - 1
    
    return returns, corr_matrix, cumulative_returns

@st.cache_data(ttl=3600)
def get_dynamic_trending_tickers():
    """
    Fetches 'Most Active' tickers from Yahoo Finance.
    Existing logic for Correlation Radar default items.
    """
    fallback_tickers = ['RKLB', 'MU', 'OKLO', 'LLY', 'SOFI']
    url = "https://finance.yahoo.com/most-active"
    
    # Create exclusion set from static menu
    exclusion_set = {t for t in STATIC_MENU_ITEMS if not t.startswith('---')}
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        dfs = pd.read_html(StringIO(response.text))
        
        if dfs:
            df_scrape = dfs[0]
            if 'Symbol' in df_scrape.columns:
                candidates_raw = df_scrape['Symbol'].head(30).dropna().astype(str).tolist()
                candidates = [t.split()[0] for t in candidates_raw if t]

                # Quick filtering logic (simplified from original for brevity)
                filtered = [t for t in candidates if t not in exclusion_set]
                return filtered[:5]

        return fallback_tickers
        
    except Exception as e:
        print(f"Failed to fetch trending tickers: {e}")
        return fallback_tickers

def generate_insights(corr_matrix):
    insights = []
    
    # Define Asset Classes for Fake Hedge Detection
    defensive_assets = {'GLD', 'IAU', 'TLT', 'IEF', 'AGG', 'BND', 'XLP', 'XLV', 'XLU', 'LQD', 'USDJPY=X'}
    risky_assets = {'QQQ', 'TQQQ', 'NVDA', 'SOXL', 'SMH', 'BTC-USD', 'ETH-USD', 'MSTR', 'COIN', 'PLTR', 'TSLA', 'ARKK', 'SPY'}

    # 1. Pairwise checks
    processed_pairs = set()
    columns = corr_matrix.columns
    
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            ticker_a = columns[i]
            ticker_b = columns[j]
            val = corr_matrix.iloc[i, j]
            
            pair_key = tuple(sorted((ticker_a, ticker_b)))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Condition: Fake Hedge Detection (Priority)
            is_def_a = ticker_a in defensive_assets
            is_risk_a = ticker_a in risky_assets
            is_def_b = ticker_b in defensive_assets
            is_risk_b = ticker_b in risky_assets
            
            if (is_def_a and is_risk_b) or (is_risk_a and is_def_b):
                if val >= 0.5:
                    def_name = ticker_a if is_def_a else ticker_b
                    risk_name = ticker_b if is_def_a else ticker_a
                    
                    insights.append({
                        "type": "fake_hedge",
                        "display": f"🚨 **ヘッジ機能不全**: {def_name} と {risk_name} (相関: {val:.2f})",
                        "message": f"安全資産とされる {def_name} が、リスク資産 {risk_name} と強く連動しています。暴落時にクッションの役割を果たさない可能性があります。",
                        "score": abs(val) + 0.5
                    })

            # Condition A: High Correlation
            if val > 0.7:
                insights.append({
                    "type": "risk",
                    "display": f"⚠️ **集中リスク警告**: {ticker_a} と {ticker_b} (相関: {val:.2f})",
                    "message": "この2つは非常に似た動きをしています。分散効果が低いため、ポジション調整を検討してください。",
                    "score": abs(val)
                })
            
            # Condition B: Inverse Correlation
            elif val < -0.3:
                insights.append({
                    "type": "hedge",
                    "display": f"🛡️ **ヘッジ機能**: {ticker_a} と {ticker_b} (相関: {val:.2f})",
                    "message": "逆の動きをする傾向があります。ポートフォリオのリスク低減に役立っています。",
                    "score": abs(val)
                })

    # 2. Individual Asset check (Independence)
    for ticker in columns:
        encounters = corr_matrix[ticker].drop(ticker)
        max_corr = encounters.abs().max()
        if max_corr < 0.25:
             insights.append({
                "type": "independent",
                "display": f"🧘 **独立独歩**: {ticker}",
                "message": f"他の資産との連動性が低く（最大相関 {max_corr:.2f}）、独自の要因で動いています。分散投資の観点で優秀です。",
                "score": (1 - max_corr)
            })



    # --- Filtering Logic: Max 2 per Type ---
    # Sort by score descending to keep the "most important" ones
    insights.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    final_insights = []
    type_counts = {}
    
    for item in insights:
        t = item['type']
        count = type_counts.get(t, 0)
        
        if count < 2:
            final_insights.append(item)
            type_counts[t] = count + 1
            
    return final_insights

# --- Portfolio Logic (New) ---
def generate_ai_portfolios(df_sorted, corr_matrix, exclude_tickers=None):
    """
    Generates 3 Portfolio Models based on momentum & logic.
    Returns dict: {'Hunter': [...], 'Fortress': [...], 'Bento': [...]}
    Each item is dict: {Ticker, Price, Weight, LatestReturn}
    """
    portfolios = {}
    
    # Pre-filter exclusions (Short-term losers)
    if exclude_tickers:
        # Filter out excluded tickers from potential candidates
        pool = df_sorted[~df_sorted['Ticker'].isin(exclude_tickers)].copy()
    else:
        pool = df_sorted.copy()
    
    # --- Model A: 🐯 The Hunter (Aggressive) ---
    # Top 5 by 1mo, RVOL > 1.2
    hunter_pool = pool[pool['RVOL'] > 1.2].copy()
    hunter_pool = hunter_pool.sort_values(by='1mo', ascending=False).head(5)
    
    if len(hunter_pool) < 5:
        # Fallback: Just top 1mo if not enough RVOL
        fallback = pool.sort_values(by='1mo', ascending=False).head(5)
        hunter_pool = fallback 
        
    portfolios['Hunter'] = hunter_pool
    
    # --- Model B: 🏰 The Fortress (Consistent) ---
    # 3mo, 6mo, YTD all > 0. Sort by 3mo. Top 8.
    fortress_pool = pool[
        (pool['3mo'] > 0) & 
        (pool['6mo'] > 0) & 
        (pool['YTD'] > 0)
    ].copy()
    fortress_pool = fortress_pool.sort_values(by='3mo', ascending=False).head(8)
    
    if len(fortress_pool) < 5:
         # Fallback: Just top 3mo positive
         fortress_pool = pool[pool['3mo'] > 0].sort_values(by='3mo', ascending=False).head(8)
         
    portfolios['Fortress'] = fortress_pool
    
    
    # --- Model C: 🥗 The Bento Box (Diversified) ---
    # Pick Top 1 (by 1mo) from each Core Sector
    # Core Sectors defined in SECTOR_DEFINITIONS keys or simplified logic
    # Keys mappings based on SECTOR_DEFINITIONS
    
    # ... (Bento logic handled in next block, just inserting Sniper before or after? 
    # Let's insert Sniper BEFORE Bento to keep alphabetical or logic flow)
    # Actually user asked for "4th portfolio". I'll put it after Bento or before. 
    # Let's put it as Model D.
    
    # --- Model D: 🦅 The Sniper (Precision) ---
    # Like Hunter, but strictly NO Overheating (RSI < 70).
    # Ideal entry point: High Momentum + Volume + But not yet Overbought.
    # Base criteria: RSI < 70 AND 1mo > 0 (Must be rising)
    
    # 1. Strict: RVOL > 1.2
    sniper_pool = pool[
        (pool['RVOL'] > 1.2) & 
        (pool['RSI'] < 70) &
        (pool['1mo'] > 0)
    ].copy()
    
    # 2. Fallback if empty: Relax RVOL
    if len(sniper_pool) < 3:
        fallback_pool = pool[
            (pool['RSI'] < 70) &
            (pool['1mo'] > 0)
        ].copy()
        # Sort by 1mo to get "Strongest among non-overheated"
        sniper_pool = fallback_pool
    
    sniper_pool = sniper_pool.sort_values(by='1mo', ascending=False).head(5)
    
    portfolios['Sniper'] = sniper_pool

    # --- Model C: 🥗 The Bento Box (Diversified) ---
    
    # 1. Map Tickers to Broad Category
    # We already have TICKER_TO_SECTOR
    # Broad Categories:
    # 1. AI/Semi ("🧠 AI & Semi")
    # 2. Energy ("⚛️ Energy & Resources")
    # 3. FinTech/Crypto ("🏦 FinTech & Real Estate")
    # 4. Space/Defense ("🌌 Space & Defense")
    # 5. Consumer/Bio ("💊 Consumer & Health", "🚗 Auto & EV")
    
    bento_picks = []
    
    # Define Target Groups (Regex friendly or exact match)
    target_groups = [
        ["AI", "Semi"], 
        ["Energy", "Resources", "Infra"],
        ["FinTech", "Crypto"],
        ["Space", "Defense"],
        ["Consumer", "Health", "Auto"]
    ]
    
    used_tickers = set()
    
    for keywords in target_groups:
        # Filter df for tickers in this sector
        candidates = []
        for t in pool['Ticker']:
            sec = TICKER_TO_SECTOR.get(t, "")
            if any(k in sec for k in keywords):
                candidates.append(t)
                
        # Get subset
        subset = pool[pool['Ticker'].isin(candidates)].sort_values(by='1mo', ascending=False)
        
        # Pick best not already satisfying correlation check?
        # Simplified: Just pick Top 1 for now, correlation check is bonus
        if not subset.empty:
            pick = subset.iloc[0]
            bento_picks.append(pick)
            used_tickers.add(pick['Ticker'])
    
    # Check if we have 5?
    if len(bento_picks) < 5:
        # Fill with "Independent" stocks if missing sectors
        # Find low correlation stocks
        pass # Keep what we have
        
    portfolios['Bento'] = pd.DataFrame(bento_picks)
    
    return portfolios

def calculate_simulated_return(portfolio_df, weight_pct=1.0):
    # Virtual Return: Sum of (1mo return * weight)
    # Simple equal weight
    if portfolio_df.empty: return 0.0
    avg_ret = portfolio_df['1mo'].mean()
    return avg_ret # This is portfolio return over last month

# --- Logic Functions: Momentum Master (New) ---

# --- Constants: Momentum Universe ---
# "Always Watching" list to ensure main characters are never missed# ==========================================
# 📊 STATIC WATCHLIST & SECTOR MAP
# ==========================================

SECTOR_DEFINITIONS = {
    "🌌 Space & Defense": [
        'RKLB', 'ASTS', 'PLTR', 'SPIR', 'LUNR', 'BKSY', 'MAXR', 'SIDU',
        'RTX', 'LMT', 'NOC', 'GD', 'LHX', 'AXON', 'KTOS', 'AVAV', 'HII', 'BA', 'CAE', 'HEI',
        'JOBY', 'ACHR', 'EH',
        'PL', 'RDW', 'RCAT', 'ONDS', 'DPRO', 'PDYN' # From Gems
    ],
    "🧠 AI & Semi": [
        'MSFT', 'GOOGL', 'AMZN', 'META', 'ORCL', 'DOCN', 'IREN', 'WULF', 'CORZ', 'NBIS',
        'VST', 'CEG', 'NRG', 'GE', 'VRT', 'NVT', 'FIX', 'EMR', 'ETN', 'PWR', 'APH', 'COHR', 'GLW', 'PSTG',
        'NVDA', 'AMD', 'AVGO', 'MU', 'TSM', 'ARM', 'SMCI', 'AMAT', 'LRCX', 'KLAC', 'INTC', 'QCOM', 'TXN', 'ADI', 'MRVL', 'ON',
        'PLTR', 'CRM', 'NOW', 'SNOW', 'DDOG', 'PATH', 'IONQ', 'RGTI', 'QBITS', 'QTUM', 'RXRX', 'SDGR', 'CRWD', 'PANW',
        'RBRK', 'ZS', 'MDB', 'CRWV', 'CIFR', 'APLD', 'ALAB', 'CRDO', 'NET', 'ASML', 'SKYT', 'AMKR', 'SNDK', 'WDC' # From Gems
    ],
    "⚛️ Energy & Resources": [
        'CCJ', 'URA', 'UEC', 'OKLO', 'SMR', 'BWXT', 'LEU',
        'GLD', 'NEM', 'GOLD', 'SLV', 'PAAS', 'FCX', 'SCCO', 'COPX', 'AA', 'CENX', 'X', 'CLF', 'NUE', 'ALB', 'SQM', 'LAC', 'MP',
        'XOM', 'CVX', 'SHEL', 'COP', 'EOG', 'EQT', 'LNG', 'KMI', 'WMB', 'ARCH', 'HCC', 'BTU', 'SLB', 'HAL',
        'LIN', 'APD', 'DOW', 'CTVA', 'MOS', 'NTR', 'ADM', 'BG',
        'GEV', 'NEE', 'NNE', 'EOSE', 'BE', 'NVTS', 'UUUU', 'CRML', 'UAMY', 'ASPI' # From Gems
    ],
    "🏗️ Infra & Industry": [
        'CAT', 'DE', 'URI', 'PWR', 'J', 'EME', 'UNP', 'CSX', 'WAB', 'GATX', 'TRN',
        'ROK', 'PH', 'TER', 'KEY', 'CGNX', 'TSLA', 'GOOG', 'MBLY', 'VUZI',
        'UPS', 'FDX', 'JBHT', 'ODFL', 'XPO',
        'POWL', 'MOD', 'ERJ' # From Gems
    ],
    "🚗 Auto & EV": [
        'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'BYDDF', 'QS', 'ENVX', 'FREY',
        'GM', 'F', 'TM', 'HMC', 'STLA', 'MGA', 'APTV', 'BWA', 'CVNA', 'KMX', 'ORLY', 'AZO',
        'UBER', 'LYFT', 'GRAB'
    ],
    "🏦 FinTech & Real Estate": [
        'JPM', 'BAC', 'GS', 'MS', 'BLK', 'KKR', 'APO', 'NU', 'SOFI',
        'V', 'MA', 'PYPL', 'SQ', 'AFRM', 'UPST', 'COIN', 'HOOD', 'MSTR', 'MARA', 'CLSK', 'LMND',
        'CB', 'PGR', 'TRV', 'MET', 'PRU', 'AON', 'MMC',
        'DHI', 'LEN', 'PHM', 'TOL', 'LOW', 'HD', 'BLD', 'MAS', 'VMC', 'MLM'
    ],
    "💊 Consumer & Health": [
        'KO', 'PEP', 'MNST', 'STZ', 'BUD', 'PM', 'MO', 'TSN', 'GIS', 'K', 'SYY', 'MDLZ', 'HSY',
        'WMT', 'COST', 'TGT', 'AMZN', 'SHOP', 'MELI', 'NKE', 'LULU', 'ONON', 'TJX', 'ROST', 'LVMUY', 'TPR', 'EL', 'ULTA',
        'LLY', 'NVO', 'JNJ', 'PFE', 'MRK', 'ABBV', 'AMGN', 'VRTX', 'REGN', 'ISRG', 'SYK', 'TMO', 'DHR', 'MCK', 'COR', 'UNH', 'ELV', 'HCA',
        'NFLX', 'DIS', 'WBD', 'LYV', 'BKNG', 'ABNB', 'MAR', 'HLT', 'RCL', 'CCL', 'DAL', 'UAL', 'LUV',
        'VKTX', 'SMMT', 'ANF', 'SG', 'ROOT', 'GCT', 'WING', 'LNTH', 'ASP', 'CART', 'TSEM' # From Gems mixed
    ]
}

# 1. Flatten into Dict map for lookup: {'NVDA': '🧠 AI & Semi', ...}
TICKER_TO_SECTOR = {}
for sector, tickers in SECTOR_DEFINITIONS.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = sector

# 2. Complete Watchlist
STATIC_MOMENTUM_WATCHLIST = list(TICKER_TO_SECTOR.keys())

# --- Metadata Helpers ---
@st.cache_data(ttl=86400) # Cache metadata for a day
def get_ticker_metadata(ticker):
    """
    Fetches basic info (Short Name, Sector/Industry) for a single ticker.
    Used only for Top 5-10 to save API calls.
    Returns: (name, category_label)
    """
    # 1. Check Scraped Cache (Fastest)
    if 'dynamic_names' in st.session_state:
        if ticker in st.session_state['dynamic_names']:
             return st.session_state['dynamic_names'][ticker], '🌊 Market Mover'

    # 2. Fallback to API (Slow)
    try:
        t = yf.Ticker(ticker)
        info = t.info
        name = info.get('shortName', info.get('longName', ticker))
        
        # Priority: Industry > Sector > 'Unknown'
        # e.g. "Aerospace & Defense" is better than "Industrials"
        industry = info.get('industry')
        sector = info.get('sector')
        
        category = industry if industry else (sector if sector else '🌊 Market Mover')
        
        return name, category
    except:
        return ticker, '🌊 Market Mover'

import concurrent.futures

@st.cache_data(ttl=900) 
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
    dynamic_names = {} # Capture names during scrape to save API calls
    
    # Add Static List first
    for t in STATIC_MOMENTUM_WATCHLIST:
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
        except Exception as e:
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
                        
                        # Capture Name if available
                        if 'Name' in row and isinstance(row['Name'], str):
                            dynamic_names[sym] = row['Name']

    # Persist scraped names to Session State (for Name Metadata)
    if 'dynamic_names' not in st.session_state:
        st.session_state['dynamic_names'] = {}
    st.session_state['dynamic_names'].update(dynamic_names)

    return list(all_candidates)

@st.cache_data(ttl=300, show_spinner=False)
def calculate_momentum_metrics(tickers):
    """
    Calculates detailed metrics for the given tickers.
    Optimized: Handles Validation (Price/Vol) AND Metrics in one pass.
    Cached: 5 minutes TTL to avoid re-downloading on every interaction.
    """
    if not tickers:
        return None, None

    # Download 1y data to calculate long-term MA and 1y return
    try:
        # Fetching for ALL candidates in one go
        # Group by ticker is safer for multi-ticker
        df = yf.download(tickers, period="1y", group_by='ticker', auto_adjust=True, progress=False, threads=True)
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
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
            if t not in STATIC_MOMENTUM_WATCHLIST:
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
    df_metrics = df_metrics[cols]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_momentum_metrics(tickers):
    """
    Calculates detailed metrics for the given tickers.
    Optimized: Handles Validation (Price/Vol) AND Metrics in one pass.
    """
    if not tickers:
        return None, None

    # Download 1y data to calculate long-term MA and 1y return
    try:
        # Fetching for ALL candidates in one go
        # Group by ticker is safer for multi-ticker
        df = yf.download(tickers, period="1y", group_by='ticker', auto_adjust=True, progress=False, threads=False)
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
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
            if t not in STATIC_MOMENTUM_WATCHLIST:
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
    df_metrics = df_metrics[cols]
    
    return df_metrics, history_dict

# --- Main App ---
def main():
    # st.set_page_config is now called globally at line 15
    
    # --- Hide Streamlit Style (Hamburger Menu Friendly Version) ---
    hide_st_style = """
        <style>
        /* 1. フッター（下の Made with Streamlit）を完全に消す */
        footer {visibility: hidden !important;}
        [data-testid="stFooter"] {display: none !important;}
        
        /* 2. ヘッダーの「バー（背景色や装飾）」だけを透明にして見えなくする */
        /* ※header全体を消すとボタンも消えるため、背景透明化で対応 */
        /* [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0) !important;
            border-bottom: none !important;
        } */
        [data-testid="stDecoration"] {
            display: none !important; /* 上部の虹色の線を消す */
        }

        /* 3. 右上の「...」メニューとデプロイボタンを消す */
        /* 左上のハンバーガー（サイドバー開閉）は残ります */
        #MainMenu {visibility: hidden !important;}
        .stDeployButton {display:none !important;}
        [data-testid="stAppDeployButton"] {display:none !important;}
        [data-testid="stHeaderActionElements"] {display: none !important;}
        [data-testid="stStatusWidget"] {display:none !important;}

        /* 4. 余白調整 */
        /* ヘッダーの背景が消えた分、コンテンツを少し上に詰める */
        .block-container {
            padding-top: 2rem !important; 
            padding-bottom: 0rem !important;
        }
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # --- Sidebar: Global Navigation ---
    with st.sidebar:
        st.title("🤖 Market Analyst AI")
        
        mode = st.radio(
            "Select Mode",
            ["📊 Correlation Radar", "🚀 Momentum Master"],
            index=0
        )
        st.markdown("---")

    # --- Router ---
    if mode == "📊 Correlation Radar":
        render_correlation_radar()
    elif mode == "🚀 Momentum Master":
        render_momentum_master()

# --- View: Correlation Radar ---
def render_correlation_radar():
    st.title("📊 Market Correlation Radar")
    st.markdown("""
    **目的**: 為替、株式、債券、暗号資産など、異なるアセット間の「現在の連動性」を可視化します。
    単なる価格比較ではなく、**日次リターン（変化率）** に基づく純粋な相関を表示します。
    """)
    
    # Load Settings (Only once per session)
    if 'tickers' not in st.session_state:
        st.session_state['tickers'] = ["USDJPY=X", "^TNX", "GLD", "QQQ", "SMH", "BTC-USD", "XLP", "XLV"]
            
    if 'period' not in st.session_state:
        st.session_state['period'] = "1y"

    # --- Configuration ---
    with st.sidebar:
        st.header("⚙️ Radar Settings")
        
        # 1. Fetch Trending for Radar
        trending_tickers = get_dynamic_trending_tickers()
        popular_tickers = []
        if trending_tickers:
            popular_tickers.extend(["--- 🔥 Trending (Yahoo Finance) ---"] + trending_tickers)
        popular_tickers.extend(STATIC_MENU_ITEMS)
        
        # Merge saved tickers
        current_selection = st.session_state.get('tickers', [])
        options = list(popular_tickers)
        for t in current_selection:
            if t not in options:
                options.append(t)

        tickers_input = st.multiselect(
            "対象銘柄 (Tickers)",
            options=options,
            key="tickers",
            default=st.session_state['tickers'],
            max_selections=10
        )
        
        st.caption("※「---」ヘッダーは無視されます。")
        
        # Custom input
        def add_custom_ticker():
            new_ticker = st.session_state.new_ticker_input.strip().upper()
            if new_ticker:
                current = list(st.session_state['tickers'])
                if new_ticker not in current:
                    if len(current) < 10:
                        current.append(new_ticker)
                        st.session_state['tickers'] = current
        
        st.text_input(
            "➕ Add Ticker",
            key="new_ticker_input",
            on_change=add_custom_ticker
        )
        
        period_options = {
            '1y': '1 Year (長期)', '3mo': '3 Months', '1mo': '1 Month', '5d': '5 Days'
        }
        st.selectbox(
            "Analysis Period", 
            list(period_options.keys()), 
            key="period", 
            format_func=lambda x: period_options.get(x, x)
        )

    # --- Main Content ---
    if tickers_input:
        with st.spinner('Fetching Radar data...'):
            df_prices = get_data(tickers_input, st.session_state['period'])

        if df_prices is not None and not df_prices.empty:
            if len(df_prices) < 2:
                st.warning("データ不足。期間を延ばしてください。")
            else:
                returns, corr_matrix, cumulative_returns = calculate_stats(df_prices)
                
                # 1. Heatmap
                st.subheader("Correlation Matrix")
                if corr_matrix is not None:
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax_corr, square=True)
                    st.pyplot(fig_corr, use_container_width=False)
                
                st.markdown("---")
                
                # 2. Chart
                st.subheader("Relative Performance")
                if cumulative_returns is not None:
                    fig_perf, ax_perf = plt.subplots(figsize=(10, 5))
                    for column in cumulative_returns.columns:
                        ax_perf.plot(cumulative_returns.index, cumulative_returns[column] * 100, label=column)
                    ax_perf.set_ylabel("Return (%)")
                    ax_perf.grid(True, linestyle='--', alpha=0.6)
                    ax_perf.legend(loc='upper left', bbox_to_anchor=(1, 1))
                    plt.tight_layout()
                    st.pyplot(fig_perf, use_container_width=False)
                
                # 3. AI Insights
                st.markdown("---")
                st.subheader("📊 AI Analyst Insights")
                insights = generate_insights(corr_matrix)
                if insights:
                    for item in insights:
                        t = item['type']
                        msg = f"**{item['display']}**\n\n{item['message']}"
                        
                        if t == 'fake_hedge' or t == 'risk':
                            st.warning(msg, icon="⚠️")
                        elif t == 'hedge':
                            st.success(msg, icon="🛡️")
                        else:
                            st.info(msg, icon="ℹ️")
                else:
                    st.info("特筆すべき相関パターンはありません。")
        else:
            st.error("No data found.")


import random # Add at top if not exists (handling in instruction context)

# --- AI Comment Logic ---
def generate_dynamic_comment(ticker, row):
    """
    Generates a entertaining, randomized comment based on metrics.
    """
    rvol = row.get('RVOL', 0)
    rsi = row.get('RSI', 50)
    sma_ok = row.get('Above_SMA50', False)
    ret_3mo = row.get('3mo', 0)
    
    # Templates
    # 1. Volume Surge (Priority 1)
    if rvol > 3.0:
        templates = [
            f"🚀 {ticker}の出来高が異常値！何か漏れてるかも？",
            f"⚡ 機関が動いたぞ。{ticker}の初動に乗り遅れるな。",
            f"💰 マネーゲーム開始の合図。{ticker}を監視せよ。",
            f"📢 {ticker}に何かが起きている...イナゴタワー建設予定地か？"
        ]
        return random.choice(templates)
        
    # 2. Overbought (RSI > 75)
    if rsi > 75:
        templates = [
            f"🔥 {ticker}はアチアチだ。火傷する前に逃げとけ。",
            f"⚠️ 欲張りすぎ。そろそろ{ticker}は調整入るぞ。",
            f"🛑 利確千人力。天井で掴むのは素人だけだ。",
            f"🎢 ジェットコースターの頂上かも。{ticker}から降りる準備を。"
        ]
        return random.choice(templates)
        
    # 3. Oversold (RSI < 30)
    if rsi < 30:
        templates = [
            f"🧊 売られすぎ。そろそろ{ticker}のリバウンドあるで。",
            f"🎣 落ちるナイフ？いや、{ticker}はバーゲンセールかも。",
            f"💎 誰も見てない今こそ{ticker}を拾うチャンス。"
        ]
        return random.choice(templates)

    # 4. Dip Buy (Uptrend but cool RSI)
    # Price > SMA50 (Bullish) but RSI < 45 (Not hot)
    if sma_ok and rsi < 45:
        templates = [
            f"🛒 {ticker}は上昇トレンド中の押し目。絶好の拾い場。",
            f"📉 健全な調整だ。{ticker}のバーゲンは長くは続かない。",
            f"🐂 休憩中の{ticker}を拾っておくのが賢い投資家。"
        ]
        return random.choice(templates)

    # 5. Strong Uptrend (SMA OK + Positive Mom + RSI OK)
    if sma_ok and ret_3mo > 0:
        templates = [
            f"🐂 綺麗なチャートだ。{ticker}は素直に買い。",
            f"📈 逆らう理由がない。トレンドフォローこそ正義。",
            f"🚀 {ticker}は青天井モード突入か？握力高めていけ。"
        ]
        return random.choice(templates)
        
    # 6. Bear Trend (Price < SMA50 & Negative Mom)
    if not sma_ok and ret_3mo < 0:
        templates = [
            f"🐻 完全な下降トレンド。{ticker}には触るな。",
            f"⛈️ 雨降って地固まる...まで{ticker}は様子見が無難。",
            f"📉 まだ掘るぞ。{ticker}の底はここじゃない。",
            f"💀 デッドクロス継続中。{ticker}のロングは自殺行為。"
        ]
        return random.choice(templates)
        
    # 7. Default/Neutral
    templates = [
        f"👀 {ticker}はまだ様子見。次の動きを待て。",
        f"😴 出来高が足りない。{ticker}は寝かせておこう。",
        f"🤔 {ticker}の方向性が定まらない。触るな危険。"
    ]
    return random.choice(templates)

# --- View: Momentum Master ---
def render_momentum_master():
    st.title("🚀 Momentum Master (順張り分析)")
    st.markdown("""
    **目的**: 指定した期間において、最もパフォーマンスが良い「最強の10銘柄」を瞬時に特定します。
    (Pool: Hybrid Scan | Real-time Market Movers + Static High-Beta Watchlist)
    """)
    
    with st.sidebar:
        st.header("Action")
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.cache_data.clear()
            st.rerun()

    # --- UI: Control Panel ---
    st.markdown("### 🎯 Focus Period Selector")
    
    period_map = {
        '1d': '1 Day (本日)',
        '5d': '5 Days (週間)',
        '1mo': '1 Month (月間)',
        '3mo': '3 Months (四半期)',
        '6mo': '6 Months (半年)',
        'YTD': 'YTD (年初来)',
        '1y': '1 Year (年間)'
    }
    
    # Default to 1d
    selected_period = st.selectbox(
        "どの期間のモメンタムを見ますか？",
        options=list(period_map.keys()),
        index=0, 
        format_func=lambda x: period_map[x]
    )

    with st.spinner(f'Scanning approx 100+ candidates for {selected_period} momentum...'):
        # 1. Get Candidates (Hybrid)
        candidates = get_momentum_candidates("hybrid")
        
        if not candidates:
            st.error("現在、十分なデータが取得できませんでした。時間をおいて再試行してください。")
            return

        # 2. Calculate Metrics
        # This will calculate ALL periods for these candidates
        df_metrics, history_dict = calculate_momentum_metrics(candidates)
        
        if df_metrics is None or df_metrics.empty:
            st.error("Metric calculation failed.")
            return
            
    # --- UI: Top 5 Filter ---
    
    # Ensure column exists
    if selected_period not in df_metrics.columns:
        st.error(f"Data for {selected_period} is missing.")
        return

    # Sort Descending
    df_sorted = df_metrics.sort_values(selected_period, ascending=False)

    # Filter: Market Movers (Dynamic) only for 1d?
    # NO: User requested to allow Market Movers for all periods, BUT with a "Consistency Filter".
    # Logic: If selected_period is long (e.g. 1mo), exclude stocks where shorter period return > long period return.
    # This filters out "Pump & Dump" or recent spikes that aren't consistent with the long term trend.
    
    # Hierarchy definition
    period_hierarchy = ['1d', '5d', '1mo', '3mo', '6mo', '1y']
    
    # Dynamic Filter
    if selected_period != '1d' and selected_period in period_hierarchy:
        # Get index
        target_idx = period_hierarchy.index(selected_period)
        
        # Check all shorter periods
        shorter_periods = period_hierarchy[:target_idx]
        
        # Filter Condition: Keep only if Return(Shorter) <= Return(Target)
        # We need to apply this to df_sorted.
        # Note: df_metrics contains all period columns.
        
        valid_indices = []
        for idx, row in df_sorted.iterrows():
            is_consistent = True
            target_ret = row[selected_period]
            
            # Skip if target is NaN
            if pd.isna(target_ret):
                continue
                
            for sp in shorter_periods:
                if sp in row and not pd.isna(row[sp]):
                    short_ret = row[sp]
                    # STRICT FILTER: If Short Return > Target Return, it implies momentum is fading or it was a spike.
                    # User: "1d(100%) > 1m(30%) -> OUT"
                    # User: "1d(10%) < 1m(30%) -> IN"
                     
                    # Tolerance? Let's use strict for now as requested.
                    if short_ret > target_ret:
                        is_consistent = False
                        break
            
            if is_consistent:
                valid_indices.append(idx)
                
        df_sorted = df_sorted.loc[valid_indices]
    
    # Also, we do NOT filter by STATIC_MOMENTUM_WATCHLIST anymore if it's consistent.
    # Unless... wait, if it's NOT in static list, it MUST be a market mover.
    # So we are now allowing Market Movers into the main ranking provided they are consistent.
    
    # Take Top 10
    top_10 = df_sorted.head(10).copy() # Copy to avoid SettingWithCopyWarning
    
    # Enrich with Name, Sector, AI Strategy, AND Earnings
    names = []
    sectors = []
    strategies = []
    earnings_dates = []
    
    for _, row in top_10.iterrows():
        t = row['Ticker']
        
        # 1. Metadata Fetch
        static_sec = TICKER_TO_SECTOR.get(t)
        d_name, d_cat = get_ticker_metadata(t)
        
        names.append(d_name)
        
        if static_sec:
            sectors.append(static_sec)
        elif "🌊" in d_cat:
             # Logic fix: d_cat already has emoji if it comes from get_ticker_metadata default
            sectors.append(d_cat)
        else:
            sectors.append(f"🌊 {d_cat}")
            
        # 2. AI Strategy
        strategies.append(generate_dynamic_comment(t, row))
        
        # 3. Earnings Date (Lazy fetch for Top 10 only)
        earnings_dates.append(get_earnings_next(t))
        
    top_10['Name'] = names
    top_10['Sector'] = sectors
    top_10['AI Strategy'] = strategies
    top_10['Earnings'] = earnings_dates
    
    st.markdown(f"### 🏆 Top 10 Strongest Stocks ({period_map[selected_period]})")
    
    # Legend
    with st.expander("ℹ️ Signal Icon Legend (アイコンの意味)", expanded=False):
        st.markdown("""
        - ⚡ **Volume Surge**: 出来高急増 (RVOL > 2.0)
        - 🐂 **Bull Mode**: 上昇トレンド (SMA50より上 & 3ヶ月+)。順張り。
        - 🛒 **Dip Buy**: 押し目買いチャンス (上昇中だがRSI<45で調整中)
        - 🔥 **Hot**: 買われすぎ (RSI > 70)。天井警戒。
        - 🐻 **Bear Trend**: 下降トレンド (SMA50より下 & 3ヶ月-)。戻り売り。
        - 🧊 **Oversold**: 売られすぎ (RSI < 30)。リバウンド狙い。
        """)

    if selected_period != '1d':
        st.caption("※長期分析（5d以上）のため、監視リスト（Static Watchlist）内のみでランキングしています。")
    
    with st.expander("🛠️ Debug Information (Candidates)", expanded=False):
         st.write(f"Candidates Found: {len(candidates)}")
         st.write(f"List: {', '.join(candidates)}")
    
    # Styling
    def highlight_focus(val):
        return 'background-color: #ffeb3b; color: black; font-weight: bold;' 

    # Show Table
    column_config = {
        "Ticker": st.column_config.TextColumn("Ticker", width="small", pinned=True),
        "Name": st.column_config.TextColumn("Company", width="medium"),
        "Sector": st.column_config.TextColumn("Sector", width="medium"),
        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "Signal": st.column_config.TextColumn(
            "Signal", 
            width="medium",
            help="⚡:出来高 | 🐂:上昇 | 🛒:押し目 | 🔥:加熱 | 🐻:下降 | 🧊:底値"
        ),
        "AI Strategy": st.column_config.TextColumn("🤖 AI Analysis", width="large"),
        "Earnings": st.column_config.TextColumn("Earnings (Next)", width="medium"),
        selected_period: st.column_config.NumberColumn(
            f"{selected_period.upper()} Return", 
            format="%.2f%%",
        )
    }
    
    context_cols = ['Ticker', 'Name', 'Sector', 'Price', 'Signal', 'AI Strategy', 'Earnings', selected_period]
    
    st.dataframe(
        top_10[context_cols].style.applymap(
            highlight_focus, subset=[selected_period]
        ).format({
            selected_period: "{:+.2f}%",
            'Price': "${:.2f}"
        }),
        column_config=column_config,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # --- UI: Chart ---
    st.subheader(f"📈 Performance Comparison (Top 10: {selected_period})")
    
    top_tickers = top_10['Ticker'].tolist()
    
    if top_tickers:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Decide chart window based on period (approx trading days)
        window_map = {
            '1d': 2, '5d': 5, '1mo': 22, '3mo': 65, '6mo': 130, 'YTD': 252, '1y': 252
        }
        days = window_map.get(selected_period, 65)
        
        for t in top_tickers:
            if t in history_dict:
                s = history_dict[t]
                
                # Slice data to relevant period + padding
                # If dataframe is shorter than days, take all
                slice_data = s.tail(days)
                if slice_data.empty: continue
                
                # Rebase to 0% at start of chart
                rebased = (slice_data / slice_data.iloc[0] - 1) * 100
                ax.plot(rebased.index, rebased, label=t)
        
        ax.set_ylabel("Return (%)")
        ax.set_title(f"Relative Performance (Last ~{days} Trading Days)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    # --- UI: News Section for Top Stocks ---
    st.markdown("---")
    st.subheader("📰 Latest News & Analysis")
    
    # Select box default to top 1
    default_ix = 0 if len(top_tickers) > 0 else None
    
    if top_tickers:
        news_ticker = st.selectbox("Select Ticker to View News:", top_tickers, index=default_ix)
        
        if news_ticker:
            # Retrieve company name from the dataframe or metadata
            # We already have it in top_10['Name'] but need to look it up
            # Simpler: just get it again or look in df
            selected_row = top_10[top_10['Ticker'] == news_ticker]
            if not selected_row.empty:
                c_name = selected_row.iloc[0]['Name']
            else:
                c_name, _ = get_ticker_metadata(news_ticker)

            with st.spinner(f"Fetching news for {news_ticker} ({c_name})..."):
                news_items = get_ticker_news(news_ticker, company_name=c_name)
                
                if news_items:
                    for item in news_items:
                        with st.expander(f"📰 {item['title']} ({item['publisher']})", expanded=True):
                            st.write(f"**Published**: {item['time']}")
                            st.write(f"[Read Article]({item['link']})")
                else:
                    st.info(f"No specific news found for {news_ticker} in the last 3 days.")
    
    # --- Part 1.5: Worst 10 Stocks ---
    st.markdown("---")
    # Take Bottom 10 (Worst Performers)
    bottom_10 = df_sorted.tail(10).sort_values(selected_period, ascending=True).copy()
    
    # Enrichment
    b_names = []
    b_sectors = []
    b_strategies = []
    b_earnings = []
    
    for _, row in bottom_10.iterrows():
        t = row['Ticker']
        static_sec = TICKER_TO_SECTOR.get(t)
        d_name, d_cat = get_ticker_metadata(t)
        
        b_names.append(d_name)
        if static_sec:
            b_sectors.append(static_sec)
        elif "🌊" in d_cat:
            b_sectors.append(d_cat)
        else:
            b_sectors.append(f"🌊 {d_cat}")
        
        b_strategies.append(generate_dynamic_comment(t, row))
        b_earnings.append(get_earnings_next(t))
        
    bottom_10['Name'] = b_names
    bottom_10['Sector'] = b_sectors
    bottom_10['AI Strategy'] = b_strategies
    bottom_10['Earnings'] = b_earnings
    
    st.subheader(f"📉 Worst 10 Performers ({period_map[selected_period]})")
    
    st.dataframe(
        bottom_10[context_cols].style.applymap(
            lambda x: 'background-color: #ffebee; color: black;', subset=[selected_period]
        ).format({
            selected_period: "{:+.2f}%",
            'Price': "${:.2f}"
        }),
        column_config=column_config,
        use_container_width=True,
        hide_index=True
    )

    # --- Part 2: Thematic ETF Analysis ---
    st.markdown("---")
    st.header("🌍 Global Theme & Sector Analysis")
    st.markdown("市場の資金がどの「テーマ・セクター」に流れているかをマクロ視点で分析します。")

    with st.spinner('Analyzing 40+ Thematic ETFs...'):
        # 1. Prepare ETF list
        etf_tickers = list(THEMATIC_ETFS.values())
        
        # 2. Calculate ETF Metrics (Re-using existing function)
        df_etf, _ = calculate_momentum_metrics(etf_tickers)
        
        if df_etf is not None and not df_etf.empty:
            # 3. Process & Sort
            # Map Ticker back to Theme Name
            # Build reverse map: Ticker -> Theme Label
            ticker_to_theme = {v: k for k, v in THEMATIC_ETFS.items()}
            
            df_etf['Theme'] = df_etf['Ticker'].map(ticker_to_theme)
            
            # Sort by selected period
            if selected_period in df_etf.columns:
                df_etf_sorted = df_etf.sort_values(selected_period, ascending=False)
                
                # Top 10 Themes
                top_etf = df_etf_sorted.head(10).copy()
                
                # Bottom 10 Themes
                bottom_etf = df_etf_sorted.tail(10).sort_values(selected_period, ascending=True).copy()
                
                # Display Top Table
                st.subheader(f"🔥 Hottest Themes ({period_map[selected_period]})")
                
                etf_cols = {
                    "Theme": st.column_config.TextColumn("Theme (Sector)", width="medium"),
                    "Ticker": st.column_config.TextColumn("ETF", width="small"),
                    "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "Signal": st.column_config.TextColumn("Signal", width="small"),
                    selected_period: st.column_config.NumberColumn(
                        f"{selected_period.upper()} Return", 
                        format="%.2f%%",
                    )
                }
                
                etf_display_cols = ['Theme', 'Ticker', 'Price', 'Signal', selected_period]
                
                st.dataframe(
                    top_etf[etf_display_cols].style.applymap(
                        highlight_focus, subset=[selected_period]
                    ).format({
                        selected_period: "{:+.2f}%",
                        'Price': "${:.2f}"
                    }),
                    column_config=etf_cols,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Display Bottom Table
                st.subheader(f"🥶 Coldest Themes ({period_map[selected_period]})")
                
                st.dataframe(
                    bottom_etf[etf_display_cols].style.applymap(
                        lambda x: 'background-color: #ffebee; color: black;', subset=[selected_period]
                    ).format({
                        selected_period: "{:+.2f}%",
                        'Price': "${:.2f}"
                    }),
                    column_config=etf_cols,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.error("ETF Data missing for selected period.")
        else:
            st.warning("Could not fetch ETF data.")

    
    # --- Part 4: 🤖 AI Portfolio Builder ---
    st.markdown("---")
    st.subheader("🤖 AI Portfolio Builder (Alpha)")
    st.caption("現在の市場環境（Momentum/Trend/Correlation）に基づき、AIが推奨する3つのポートフォリオ案です。")
    
    
    # Generate Portfolios
    # Need correlation matrix for Bento Box
    # Construct from history_dict
    with st.spinner("Calculating portfolio correlations..."):
        # Align histories
        try:
             # Create DataFrame from dict (keys=tickers, values=Series)
             # Values are normalized history. 
             # We need to make sure they share index or align. 
             # history_dict contains normalized series with DateTime index.
             
             # Filter only relevant candidates to speed up? 
             # Or just use all candidates history.
             
             price_history_df = pd.DataFrame(history_dict)
             
             # Some series might be shorter, align on recent date?
             # corr() handles NaNs by ignoring pairs.
             corr_matrix = price_history_df.corr()
             
             # If empty (unexpected)
             if corr_matrix.empty:
                 corr_matrix = pd.DataFrame()
        except Exception as e:
            # st.error(f"Correlation calc failed: {e}")
            corr_matrix = pd.DataFrame()

    # Identify Short-term Losers (to exclude from AI Portfolios)
    # Worst 10 for 1d and 5d
    # Note: df_metrics contains '1d' and '5d' for ALL candidates
    exclude_list = set()
    try:
        if '1d' in df_metrics.columns:
            worst_1d = df_metrics.sort_values('1d', ascending=True).head(10)['Ticker'].tolist()
            exclude_list.update(worst_1d)
        if '5d' in df_metrics.columns:
            worst_5d = df_metrics.sort_values('5d', ascending=True).head(10)['Ticker'].tolist()
            exclude_list.update(worst_5d)
    except:
        pass # Safety

    ai_portfolios = generate_ai_portfolios(df_sorted, corr_matrix, exclude_tickers=exclude_list)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🐯 The Hunter", "🦅 The Sniper", "🏰 The Fortress", "🥗 The Bento Box"])
    
    def render_portfolio_tab(name, df, emoji, desc):
        if df.empty:
            st.warning("条件に合致する銘柄が見つかりませんでした。")
            return
            
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown(f"### {emoji} {name}")
            st.caption(desc)
            
            # Display Table
            display_cols = ['Ticker', 'Price', '1mo', '3mo', 'RVOL', 'RSI', 'Signal']
            # Ensure cols exist
            valid_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[valid_cols].style.format({
                'Price': "{:.2f}",
                '1mo': "{:+.2f}%",
                '3mo': "{:+.2f}%",
                'RVOL': "{:.2f}",
                'RSI': "{:.1f}"
            }), hide_index=True)
            
            # Virtual Performance
            sim_return = calculate_simulated_return(df)
            st.metric("📊 過去1ヶ月の仮想リターン (直近実績)", f"{sim_return:+.2f}%")
            
        with col2:
            # Pie Chart
            # Equal weight for now
            df['Weight'] = 100 / len(df)
            fig = px.pie(df, values='Weight', names='Ticker', title=f"{name} Allocation", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    with tab1:
        render_portfolio_tab("The Hunter (短期集中)", ai_portfolios['Hunter'], "🐯", 
                             "**攻撃型:** リターン・出来高重視。加熱感（RSI高）を問わず、とにかく「今強い」銘柄に乗る戦略。※高値掴み注意")
        
    with tab2:
        render_portfolio_tab("The Sniper (精密射撃)", ai_portfolios['Sniper'], "🦅", 
                             "**厳選型:** Hunterと同様に強いモメンタムを持ちつつ、RSI < 70 の「まだ加熱していない」銘柄に絞った戦略。安全マージン重視。")
                             
    with tab3:
        render_portfolio_tab("The Fortress (堅実トレンド)", ai_portfolios['Fortress'], "🏰",
                             "**順張り型:** 3ヶ月、6ヶ月、年初来がすべてプラスの「負けない」トレンド銘柄。安定した上昇気流に乗るための構成。")
        
    with tab4:
        render_portfolio_tab("The Bento Box (セクター分散)", ai_portfolios['Bento'], "🥗",
                             "**バランス型:** 主要テーマ（AI・エネ・金融・宇宙・消費）からそれぞれ最強の1銘柄をピックアップ。相関係数を抑えつつリターンを狙う幕の内弁当。")
                             
    # --- Footer: Disclaimer ---
    st.markdown("---")
    st.caption("⚠️ **免責事項**: 本アプリケーションは情報提供のみを目的としており、投資勧誘や助言を意図するものではありません。表示されるデータやAIによる分析結果は過去の実績に基づいており、将来の運用成果を保証するものではありません。投資判断はご自身の責任において行ってください。")

if __name__ == "__main__":
    main()
