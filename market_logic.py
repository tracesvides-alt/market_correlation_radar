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
    # ---------------------------------------------------------
    # 1. AI & Semiconductor (Hardware / Cloud)
    # ---------------------------------------------------------
    "🖥️ AI: Hardware & Cloud Infra": [
        "CRWV", "NVDA", "AMD", "SMCI", "VRT", "ANET", "PSTG", "DELL", "HPE", 
        "TSM", "AVGO", "ARM", "MU", "QCOM", "AMAT", "LRCX", "GFS", "STM", 
        "UMC", "ASX", "WDC", "ENTG", "AMKR", "ALAB", "NVTS", "SWKS", "KLAR", 
        "MCHP", "TXN", "ADI", "ON", "Q", "APLD"
    ],

    # ---------------------------------------------------------
    # 2. AI Software & Services
    # ---------------------------------------------------------
    "🧠 AI: Software & SaaS": [
        "FIG", "PLTR", "MSFT", "GOOGL", "GOOG", "META", "NOW", "DDOG", "SNOW", 
        "MDB", "PATH", "AI", "BBAI", "SOUN", "ESTC", "CDNS", "SNPS", "ZETA", 
        "SYM", "DOCN", "APP", "TTD", "TEAM", "HUBS", "GTLB", "CFLT", "NET", 
        "OKTA", "CRWD", "PANW", "FTNT", "ZS", "ORCL", "SAP", "IBM", "INTU", 
        "ADBE", "CRM", "WDAY", "DOCU", "ZM", "GEN", "BOX", "DBX", "ASAN", 
        "VRNS", "CCC", "FRSH", "KVYO", "UPWK", "MGNI", "OPCH", "PRO", "PAYC", 
        "TYL", "RDDT", "DJT"
    ],

    # ---------------------------------------------------------
    # 3. Crypto & FinTech
    # ---------------------------------------------------------
    "💸 Crypto & FinTech": [
        "CRCL", "XYZ", "MSTR", "COIN", "MARA", "RIOT", "HOOD", "PYPL", "XYZ", 
        "SOFI", "AFRM", "UPST", "BILL", "TOST", "FOUR", "PAYX", "ADP", "FIS", 
        "FISV", "GPN", "FLUT", "DKNG", "RELY", "INTR", "PAGS", "WU", "STNE", 
        "XP", "NU", "LC", "DLO", "BLSH", "GLXY","CORZ", "IREN", "WULF", 
        "CIFR", "CLSK", "BTDR",  "HIVE", "BITF", "HUT"
    ],

    # ---------------------------------------------------------
    # 4. Space & Defense
    # ---------------------------------------------------------
    "🌌 Space & Defense": [
        "RKLB", "ASTS", "LUNR", "JOBY", "ACHR", "BA", "PL", "SPIR", "SPCE", 
        "IRDM", "SATS", "ONDS", "RTX", "KTOS", "HWM", "LMT", "GD", "NOC", 
        "LHX", "AMTM", "AVAV", "AXON", "BWXT"
    ],

    # ---------------------------------------------------------
    # 5. Energy: Nuclear (AI Power Theme)
    # ---------------------------------------------------------
    "☢️ Energy: Nuclear": [
        "OKLO", "SMR", "UEC", "UUUU", "CCJ", "NXE", "LEU", "DNN", "NNE", "GEV"
    ],

    # ---------------------------------------------------------
    # 6. Energy: Power & Renewables
    # ---------------------------------------------------------
    "⚡ Energy: Power & Renewables": [
        "VST", "CEG", "NRG", "NEE", "DUK", "SO", "AEP", "EXC", "PEG", "PPL", 
        "SRE", "CNP", "ED", "EIX", "ETR", "LNT", "NI", "WEC", "WTRG", "CMS", 
        "ES", "XEL", "PCG", "AES", "FLNC", "BE", "ENPH", "SEDG", "RUN", "NXT"
    ],

    # ---------------------------------------------------------
    # 7. Energy: Oil & Gas
    # ---------------------------------------------------------
    "🛢️ Energy: Oil & Gas": [
        "PR", "XOM", "CVX", "OXY", "EOG", "SLB", "HAL", "BKR", "COP", "DVN", 
        "VLO", "MPC", "PSX", "PBR", "PBR-A", "BP", "SU", "EC", "EQNR", "YPF", 
        "TRP", "KMI", "WMB", "ET", "EPD", "CTRA", "AR", "EQT", "SM", "OKE", 
        "FTI", "DINO", "PBF", "MUR", "AM", "LBRT", "CNQ", "APA", "SHEL", "VZLA", 
        "MTDR", "CHYM"
    ],

    # ---------------------------------------------------------
    # 8. BioPharma (Major & Obesity)
    # ---------------------------------------------------------
    "💊 BioPharma: Big Pharma & Obesity": [
        "LLY", "NVO", "VKTX", "PFE", "MRK", "AMGN", "BMY", "ABBV", "JNJ", 
        "GILD", "AZN", "SNY", "TEVA"
    ],

    # ---------------------------------------------------------
    # 9. BioPharma (Biotech & Gene)
    # ---------------------------------------------------------
    "🧬 BioPharma: Biotech & Gene": [
        "CRSP", "BEAM", "ARWR", "SRPT", "VRTX", "ALKS", "INCY", "EXEL", "LEGN", 
        "RPRX", "HALO", "ADMA", "BBIO", "SMMT", "FOLD", "TVTX", "ROIV", "NTLA", 
         "APQT", "LQDA", "NUVB", "ERAS", "SNDK", "TAK", "INSM", "BMRN", 
        "BMNR", "AXSM", "VVV", "INDV", "OCUL", "RNA", "ADPT", "KOD", "ARQT", 
        "CPRX", "VIR", "BNTX"
    ],

    # ---------------------------------------------------------
    # 10. MedTech & Health Services
    # ---------------------------------------------------------
    "🏥 MedTech & Health": [
        "UNH", "CVS", "ABT", "DHR", "TMO", "SYK", "BSX", "EW", "MDT", "DXCM", 
        "ZTS", "GEHC", "CNC", "DOCS", "ALHC", "NVST", "BRKR", "OGN", "BAX", 
        "XRAY", "CAH", "BHC", "SHC", "COO", "HIMS", "WRBY", "NEOG", "OSCR", 
        "ALGN", "RMD", "HCA", "ELV", "CI", "HUM", "MCK", "COR"
    ],

    # ---------------------------------------------------------
    # 11. Consumer: Food & Beverage
    # ---------------------------------------------------------
    "🍔 Consumer: Food & Bev": [
        "MICC", "KO", "PEP", "MNST", "CELH", "MCD", "SBUX", "CMG", "CAVA", 
        "HRL", "KHC", "MDLZ", "CPB", "CAG", "GIS", "TAP", "BUD", "STZ", "MO", 
        "PM", "BTI"
    ],

    # ---------------------------------------------------------
    # 12. Consumer: Retail & E-Commerce
    # ---------------------------------------------------------
    "🛒 Consumer: Retail & E-Com": [
        "AMZN", "WMT", "COST", "TGT", "LOW", "TJX", "ROST", "ETSY", "EBAY", 
        "CHWY", "CART", "DASH", "UBER", "LYFT", "GRND", "MTCH", "W", "BBY", 
        "ANF", "AEO", "KSS", "M", "VSCO", "BROS", "YMM", "PDD", "BABA", "JD", 
        "VIPS", "CPNG"
    ],

    # ---------------------------------------------------------
    # 13. Consumer: Apparel & Leisure
    # ---------------------------------------------------------
    "👗 Consumer: Apparel & Leisure": [
        "NKE", "LULU", "DECK", "ONON", "BIRK", "VFC", "LEVI", "CPRI", "UA", 
        "UAA", "RCL", "CCL", "NCLH", "VIK", "LVS", "MGM", "CZR", "DIS", "NFLX", 
        "SPOT", "PINS", "SNAP", "TTWO", "EA", "ROKU", "LYV", "IHRT", "CNK", 
        "GENI", "SBET", "STUB"
    ],

    # ---------------------------------------------------------
    # 14. Auto & EV
    # ---------------------------------------------------------
    "🚗 Auto & EV": [
        "TSLA", "RIVN", "LCID", "LI", "XPEV", "NIO", "ZETA", "PSNY", "F", 
        "GM", "STLA", "TM", "HMC", "CNH", "GNTX", "APTV", "GT", "LKQ", "CVNA", 
        "KMX", "ALV", "BWA", "QS", "GTX", "HOG"
    ],

    # ---------------------------------------------------------
    # 15. Real Estate & REITs
    # ---------------------------------------------------------
    "🏘️ Real Estate & REITs": [
        "MRP", "PLD", "AMT", "CCI", "O", "VICI", "GLPI", "WELL", "VTR", "ARE", 
        "CUBE", "REXR", "INVH", "AMH", "EQR", "UDR", "IRM", "WY", "Z", "OPEN", 
        "CSGP", "BEKE", "HR", "APLE", "STWD", "AGNC", "NLY", "RITM", "MPW", 
        "DBRG", "IRT", "DOC", "COLD", "SBRA", "BRX", "PDI", "COMP", "HST"
    ],

    # ---------------------------------------------------------
    # 16. Finance: Banks & Capital Markets
    # ---------------------------------------------------------
    "🏦 Finance: Banks & Capital": [
        "JPM", "BAC", "WFC", "C", "MS", "GS", "SCHW", "BLK", "AXP", "V", "MA", 
        "BRK-B", "AIG", "MET", "KKR", "BX", "APO", "ARES", "STT", "BK", "USB", 
        "PNC", "TFC", "COF", "FITB", "RF", "KEY", "CFG", "HBAN", "CADE", "FNB", 
        "IBKR", "DB", "UBS", "BCS", "LYG", "ITUB", "MFC", "TD", "AFL", "WRB", 
        "PGR", "EQH", "GNW", "SEI", "GOF", "ARCC", "OBDC", "BGC", "BANC", "EBC", 
        "MFG", "SMFG", "MUFG", "JEF", "PRMB", "BPRE", "COLB"
    ],

    # ---------------------------------------------------------
    # 17. Industrials & Transport
    # ---------------------------------------------------------
    "🏗️ Industrials & Transport": [
        "TIC", "CAT", "ETN", "EMR", "PWR", "URI", "JCI", "IR", "OTIS", "CARR", 
        "FIX", "GVA", "MLM", "VMC", "MMM", "GE", "HON", "ITW", "PH", "FAST", 
        "MAS", "BLDR", "CRH", "ESI", "AL", "CPRT", "RHI", "FTV", "FLR", "NVT", 
        "GTES", "APG", "TTEK", "FLEX", "CLS", "AXTA", "PCAR", "DAL", "UAL", 
        "AAL", "LUV", "ALK", "CSX", "UNP", "CP", "FDX", "UPS", "ODFL", "KNX", 
        "RXO", "ZIM", "FRO", "FLY", "QXO"
    ],

    # ---------------------------------------------------------
    # 18. Resources & Materials
    # ---------------------------------------------------------
    "⛏️ Resources & Materials": [
        "FCX", "VALE", "RIO", "BHP", "CLF", "NEM", "GOLD", "AEM", "ALB", "AA", 
        "SCCO", "MP", "CENX", "CDE", "HL", "AG", "EXK", "TGB", "FSM", "SSRM", 
        "IAG", "SVM", "PAAS", "TECK", "HBM", "GFI", "AU", "NG", "AGI", "ORLA", 
        "CC", "OLN", "IFF", "BALL", "IP", "GPK", "SUZ", "CE", "EMN", "HUN", 
        "MOS", "NTR", "HYMC", "VZLA", "AMCR", "DOW", "LIN", "DD", "LYB", "PHYS", 
        "PSLV", "IE", "NGD"
    ],

    # ---------------------------------------------------------
    # 19. Tech: Other / Emerging / Quantum
    # ---------------------------------------------------------
    "🔮 Tech: Other & Emerging": [
        "VISN", "IONQ", "QBTS", "RGTI", "QUBT", "MBLY", "HSAI", "VNET", "FYBR", 
        "LUMN", "VIAV", "CIEN", "BILI", "TME", "RUM"
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

# Extend Static List with ETFs & Register to Sector Map
for name, ticker in THEMATIC_ETFS.items():
    if ticker not in STATIC_MOMENTUM_WATCHLIST:
        STATIC_MOMENTUM_WATCHLIST.append(ticker)
    TICKER_TO_SECTOR[ticker] = name

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
