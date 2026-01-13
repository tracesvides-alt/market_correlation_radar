# Market Correlation Radar & AI Momentum Master 🚀

**最先端の金融分析ツールへようこそ。**
このアプリケーションは、従来の「相関分析」に加え、AIによる「モメンタム投資診断」と「自動ポートフォリオ提案」を統合した包括的なマーケット分析プラットフォームです。

![App Banner](https://via.placeholder.com/800x400?text=Market+Correlation+Radar+&+Momentum+Master)

---

## ✨ 主な機能 (Key Features)

### 1. 🤖 AI Portfolio Builder (New!)
現在の市場環境（トレンド・相関・ボラティリティ）を分析し、AIが最適なポートフォリオを提案します。

| ポートフォリオ | 戦略タイプ | 特徴 |
| :--- | :--- | :--- |
| **🐯 The Hunter** | 短期集中・攻撃型 | **「今強い」銘柄に乗る。** 直近リターンと出来高急増（RVOL）を重視。過熱感があってもトレンドを追うイケイケ戦略。 |
| **🦅 The Sniper** | 精密射撃・厳選型 | **「加熱していない最強株」を撃ち抜く。** 強いモメンタムを持ちつつ、RSI < 70 の銘柄のみを厳選。高値掴みを避けるスマート戦略。 |
| **🏰 The Fortress** | 堅実トレンド型 | **「負けない」順張り投資。** 3ヶ月・6ヶ月・年初来すべてがプラスの銘柄のみで構成。安定した上昇気流に乗る。 |
| **🥗 The Bento Box** | セクター分散型 | **バランス重視。** AI・エネルギー・金融・宇宙・消費の主要5セクターから、それぞれ最強の1銘柄をピックアップ。 |

### 2. 🚀 Momentum Master (モメンタム分析)
「どの株が今、一番強いのか？」を瞬時に可視化します。

- **Multi-Timeframe Analysis**: 1日、5日、1ヶ月、3ヶ月、6ヶ月、YTDの期間別ランキング。
- **Consistency Filter (一貫性フィルター)**:
    - 単なる「瞬間最大風速」の銘柄は除外します。
    - 「1ヶ月リターン < 5日リターン」のような一時的な急騰（Pop-and-Dump）を排除し、**「時間とともにリターンが積み上がっている本物のトレンド」** だけを表示します。
- **Market Movers Integration**:
    - 今日の急動意株（Market Movers）が、長期目線でも一貫して強いかどうかをチェック可能。
- **Smart Signals**:
    - ⚡ **Volume Surge**: 出来高急増
    - 🐂 **Bull Mode**: 強気トレンド
    - 🔥 **Hot**: 加熱注意 (RSI > 70)
    - 🧊 **Oversold**: 売られすぎ (RSI < 30)

### 3. 📊 Correlation Radar (相関分析)
ポートフォリオのリスク管理の要となる機能です。

- **Correlation Matrix**: 資産間の相関係数をヒートマップ化。
- **AI Risk Insights**:
    - **⚠️ 集中リスク**: 知らないうちに「同じ動きをする銘柄」ばかり買っていないか警告。
    - **🚨 Fake Hedge**: 「ヘッジしたつもり」が機能していない状態（逆相関の崩れ）を検知。
    - **🛡️ True Hedge**: 本当にリスク分散になる銘柄を発見。

---

## 📦 使用技術 (Tech Stack)

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Source**: [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance API)
- **Visualization**: [Plotly](https://plotly.com/), Matplotlib, Seaborn
- **Analysis**: Pandas, NumPy

## ⚠️ 免責事項 (Disclaimer)

本ツールは投資判断の参考情報を提供するためのものであり、投資勧誘を目的としたものではありません。
「AI Portfolio」や「Momentum Ranking」は過去のデータに基づくアルゴリズム的な提案であり、将来のリターンを保証しません。
投資は自己責任で行ってください。

---
MIT License
