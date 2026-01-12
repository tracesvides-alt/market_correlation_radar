import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests



# --- Constants ---
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

# --- Logic Functions (Separated for Testing) ---
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
        
        # Align data: Forward fill to handle mismatching trading days (E.g. Crypto vs Stocks)
        # This treats non-trading days as "no price change" instead of dropping the row
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

# --- Dynamic Ticker Fetching ---
@st.cache_data(ttl=3600)
def get_dynamic_trending_tickers():
    """
    Fetches 'Most Active' tickers from Yahoo Finance.
    Filters by:
    1. Trading Value (Close * Volume) > to exclude penny stocks
    2. Duplication > exclude tickers already in STATIC_MENU_ITEMS
    Returns a list of top 5 tickers.
    """
    fallback_tickers = ['RKLB', 'MU', 'OKLO', 'LLY', 'SOFI']
    url = "https://finance.yahoo.com/most-active"
    
    # Create exclusion set from static menu
    exclusion_set = {t for t in STATIC_MENU_ITEMS if not t.startswith('---')}
    
    try:
        # User-Agent is often required to avoid 403 Forbidden
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse tables
        from io import StringIO
        dfs = pd.read_html(StringIO(response.text))
        
        if dfs:
            df_scrape = dfs[0]
            if 'Symbol' in df_scrape.columns:
                # 1. Get broader list (Top 30)
                # Ensure symbols are strings and drop NaNs
                candidates_raw = df_scrape['Symbol'].head(30).dropna().astype(str).tolist()
                candidates = [t.split()[0] for t in candidates_raw if t]

                # 2. Fetch data to calculate Trading Value
                try:
                    df = yf.download(candidates, period="1d", progress=False, group_by='column', auto_adjust=True)
                    
                    if not df.empty and isinstance(df.columns, pd.MultiIndex):
                        closes = df['Close'].iloc[-1]
                        volumes = df['Volume'].iloc[-1]
                        
                        # Calculate Trading Value
                        trading_values = closes * volumes
                        
                        # Sort descending
                        sorted_candidates = trading_values.sort_values(ascending=False).index.tolist()
                        
                        # 3. Filter out exclusions and take Top 5
                        final_list = []
                        for t in sorted_candidates:
                            if t not in exclusion_set:
                                final_list.append(t)
                                if len(final_list) >= 5:
                                    break
                                    
                        if final_list:
                            return final_list
                            
                    elif not df.empty and 'Close' in df.columns:
                        # Single ticker case
                        t = candidates[0]
                        if t not in exclusion_set:
                            return [t]
                        return []
                        
                except Exception as e:
                    print(f"Validation download failed: {e}")
                    # Fallback filtering on scraped list
                    filtered = [t for t in candidates if t not in exclusion_set]
                    return filtered[:5]

        return fallback_tickers
        
    except Exception as e:
        print(f"Failed to fetch trending tickers: {e}")
        return fallback_tickers

# ... (rest of functions)

# --- Main App ---
def main():
    # ... (setup)
    
    # ... (load settings)

    # --- Sidebar: Configuration ---
    with st.sidebar:
        # ... (header and markdown)
        
        # ... (update_settings func)

        # 1. Fetch Trending Tickers
        trending_tickers = get_dynamic_trending_tickers()

        # 2. Categorized Popular Tickers
        popular_tickers = []
        
        if trending_tickers:
            popular_tickers.extend(["--- 🔥 Trending (Yahoo Finance) ---"] + trending_tickers)

        # Use global constant
        popular_tickers.extend(STATIC_MENU_ITEMS)
        
        # ... (rest of sidebar)

# --- Insight Logic ---
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
            # Check if one is defensive and other is risky
            is_def_a = ticker_a in defensive_assets
            is_risk_a = ticker_a in risky_assets
            is_def_b = ticker_b in defensive_assets
            is_risk_b = ticker_b in risky_assets
            
            # (Defensive vs Risky) OR (Risky vs Defensive)
            if (is_def_a and is_risk_b) or (is_risk_a and is_def_b):
                if val >= 0.5:
                    def_name = ticker_a if is_def_a else ticker_b
                    risk_name = ticker_b if is_def_a else ticker_a
                    
                    insights.append({
                        "type": "fake_hedge",
                        "display": f"🚨 **ヘッジ機能不全**: {def_name} と {risk_name} (相関: {val:.2f})",
                        "message": f"安全資産とされる {def_name} が、リスク資産 {risk_name} と強く連動しています。暴落時にクッションの役割を果たさない可能性があります。",
                        "score": abs(val) + 0.5 # Boost priority
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
                "score": (1 - max_corr) # Higher score = more independent (lower max corr)
            })

    return insights

# --- Main App ---
def main():
    # 1. UI/UX: Set Page Config
    st.set_page_config(page_title="Market Correlation Radar", layout="wide")

    st.title("Market Correlation Radar")
    
    # Hide Streamlit standard UI elements (Header, Footer, Menu, Deploy)
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display:none;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.markdown("""
    **目的**: 為替、株式、債券、暗号資産など、異なるアセット間の「現在の連動性」を可視化します。
    単なる価格比較ではなく、**日次リターン（変化率）** に基づく純粋な相関を表示します。
    """)
    
    # Load Settings (Only once per session)
    # Initialize Session State (Per-user temporary settings)
    if 'tickers' not in st.session_state:
        # Default tickers
        st.session_state['tickers'] = ["USDJPY=X", "^TNX", "GLD", "QQQ", "SMH", "BTC-USD", "XLP", "XLV"]
            
    if 'period' not in st.session_state:
        st.session_state['period'] = "1y"

    # --- Sidebar: Configuration ---
    with st.sidebar:
        st.header("設定 (Settings)")
        
        st.markdown("""
        **銘柄の指定について**
        - リストから**選択**するか、直接キーボードで**入力**して追加できます。
        - 自由に入れ替え可能ですが、**最大10銘柄**の上限があります。
        """)
        
        # Callback to save settings when changed
        # Callback to save settings when changed (Removed for Web Version)
        def update_settings():
            pass # No-op as we handle state in memory only

        # 1. Fetch Trending Tickers
        trending_tickers = get_dynamic_trending_tickers()

        # 2. Categorized Popular Tickers
        popular_tickers = []
        
        if trending_tickers:
            popular_tickers.extend(["--- 🔥 Trending (Yahoo Finance) ---"] + trending_tickers)

        popular_tickers.extend(STATIC_MENU_ITEMS)
        
        # Merge saved tickers with popular tickers for options
        current_selection = st.session_state.get('tickers', [])
        
        # Create final options list
        options = list(popular_tickers)
        for t in current_selection:
            if t not in options:
                options.append(t)

        tickers_input = st.multiselect(
            "対象銘柄 (Tickers)",
            options=options,
            # default=current_selection, # Removed to fix warning: value is handled by session_state key
            key="tickers",
            max_selections=10,
            on_change=update_settings
        )
        
        st.caption("※「---」で始まる項目は分類用ヘッダーです。選択しても計算には含まれません。")
        
        # New Period Options Mapping
        period_options = {
            '5d': '5 Days (超短期・今週の動き)',
            '1mo': '1 Month (短期トレンド)',
            '3mo': '3 Months (四半期・決算)',
            '6mo': '6 Months (中期)',
            'ytd': 'YTD (年初来)',
            '1y': '1 Year (長期)',
        }
        
        # Ensure the session state value is valid
        if st.session_state['period'] not in period_options:
             st.session_state['period'] = '1y'

        period_keys = list(period_options.keys())
        selected_period = st.selectbox(
            "Analysis Period", 
            period_keys, 
            key="period", 
            format_func=lambda x: period_options[x],
            on_change=update_settings
        )
        
        st.caption("Common Tickers examples:\n- `USDJPY=X` (USD/JPY)\n- `^TNX` (US 10Y Yield)\n- `SPY` (S&P 500)\n- `BTC-USD` (Bitcoin)")

    if tickers_input:
        with st.spinner('Fetching data...'):
            df_prices = get_data(tickers_input, selected_period)

        if df_prices is not None and not df_prices.empty:
            if len(df_prices) < 2:
                st.warning("データポイントが不足しています。期間を延ばしてください。")
            else:
                # --- Calculations ---
                returns, corr_matrix, cumulative_returns = calculate_stats(df_prices)

                # --- Visualization ---
                # --- Visualization ---
                # Removed columns for better mobile visibility (Vertical Stack)
                
                # --- Visualization ---
                # Fixed Size Charts using Matplotlib
                
                # 1. Heatmap (Fixed Size)
                st.subheader("Correlation Matrix (Heatmap)")
                st.caption("日次変化率（%）に基づく相関係数 (-1.0 to 1.0)")
                
                if corr_matrix is not None:
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 8)) # Fixed pixel size ratio
                    sns.heatmap(
                        corr_matrix, 
                        annot=True, 
                        fmt=".2f", 
                        cmap='coolwarm', 
                        vmin=-1, 
                        vmax=1, 
                        center=0,
                        ax=ax_corr,
                        square=True,
                        linewidths=.5
                    )
                    ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha='right')
                    ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0)
                    st.pyplot(fig_corr, use_container_width=False) # Important: False keeps fixed size

                st.markdown("---") 

                # 2. Performance Chart (Also Fixed Size to match Heatmap)
                st.subheader("Relative Performance")
                st.caption("期間初日を 0% とした累積リターン")
                
                if cumulative_returns is not None:
                    # Create a fixed-size matplotlib figure instead of interactive st.line_chart
                    fig_perf, ax_perf = plt.subplots(figsize=(10, 5)) # Wide aspect ratio
                    
                    # Plot logic
                    for column in cumulative_returns.columns:
                        ax_perf.plot(cumulative_returns.index, cumulative_returns[column] * 100, label=column)
                    
                    ax_perf.set_ylabel("Return (%)")
                    ax_perf.grid(True, linestyle='--', alpha=0.6)
                    ax_perf.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Legend outside
                    
                    plt.tight_layout() # Prevent cutoff
                    st.pyplot(fig_perf, use_container_width=False) # Fixed size, no resizing logic

                # --- AI Analyst Insights ---
                if corr_matrix is not None:
                    st.markdown("---")
                    st.subheader("📊 AI Analyst Insights (投資戦略)")
                    
                    insights = generate_insights(corr_matrix)
                    
                    if not insights:
                        st.info("特筆すべき強い相関や乖離は見当たりません。バランスの取れた状態、あるいは全体的に相関が薄い状態です。")
                    else:
                        # Helper to render insight card
                        def render_insight(item):
                            if item['type'] == 'fake_hedge':
                                st.warning(f"{item['display']}\n\n{item['message']}", icon="🚨")
                            elif item['type'] == 'risk':
                                st.warning(f"{item['display']}\n\n{item['message']}")
                            elif item['type'] == 'hedge':
                                st.success(f"{item['display']}\n\n{item['message']}")
                            elif item['type'] == 'independent':
                                st.info(f"{item['display']}\n\n{item['message']}")

                        # Group by type
                        grouped_insights = {'fake_hedge': [], 'risk': [], 'hedge': [], 'independent': []}
                        for item in insights:
                            if item['type'] in grouped_insights:
                                grouped_insights[item['type']].append(item)

                        # Display logic: Priority Order: Fake Hedge -> Risk -> Hedge -> Independent
                        labels = {
                            'fake_hedge': '🚨 ヘッジ機能不全 (Fake Hedge Alert)',
                            'risk': '⚠️ 集中リスク (Concentration Risk)', 
                            'hedge': '🛡️ ヘッジ候補 (Possible Hedges)', 
                            'independent': '🧘 独立した動き (Uncorrelated Assets)'
                        }
                        
                        # Define detailed display order
                        display_order = ['fake_hedge', 'risk', 'hedge', 'independent']
                        
                        for type_key in display_order:
                            items = grouped_insights[type_key]
                            if not items:
                                continue
                                
                            # Sort by score (descending)
                            items.sort(key=lambda x: x['score'], reverse=True)
                            
                            # Show top 2
                            for item in items[:2]:
                                render_insight(item)
                                
                            # Show rest in expander
                            remaining = items[2:]
                            if remaining:
                                with st.expander(f"▼ その他 {len(remaining)}件の {labels[type_key]} を表示"):
                                    for item in remaining:
                                        render_insight(item)

                # --- Data Preview ---
                with st.expander("Show Raw Data"):
                    st.subheader("Aligned Prices")
                    st.dataframe(df_prices.tail())
                    if corr_matrix is not None:
                        st.subheader("Correlation Matrix")
                        st.dataframe(corr_matrix)
        else:
            st.info("データが見つかりませんでした。ティッカーを確認してください。")
    else:
        st.info("Enter tickers in the sidebar.")

    st.markdown("---")
    st.caption("""
    **免責事項 (Disclaimer)**
    本アプリケーションは情報提供のみを目的としており、投資助言や勧誘を意図するものではありません。
    表示されるデータ、相関、AIインサイトは過去の実績や統計に基づくものであり、将来の市場動向やリターンを保証するものではありません。
    投資判断はご自身の責任において行ってください。
    """)

if __name__ == "__main__":
    main()
