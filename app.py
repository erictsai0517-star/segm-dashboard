import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

st.set_page_config(page_title="SEGM 投資儀表板 & 回測", layout="wide")
st.title("🚀 SEGM 投資儀表板 (終極防崩潰穩定版)")

# ==================== 1. 設定與參數 ====================
symbols = {
    "QQQ": "QQQ",          # 科技股大盤
    "BTC": "BTC-USD",      # 比特幣
    "TQQQ": "TQQQ",        # 3倍做多 QQQ
    "SQQQ": "SQQQ",        # 3倍做空 QQQ
    "SH": "SH",            # 1倍做空 S&P500
    "TLT": "TLT",          # 20年美債
    "GLD": "GLD",          # 黃金
    "USO": "USO",          # 原油
    "SPY": "SPY",          # 標普 500 作為基準
    "VIX": "^VIX"          # 恐慌指數
}

st.sidebar.header("⚙️ 策略與回測設定")
momentum_period = st.sidebar.number_input("動能週期 (天數)", min_value=10, max_value=200, value=60)
btc_ma_period = st.sidebar.number_input("BTC 均線週期 (天數)", min_value=10, max_value=200, value=100)

cash_annual_rate = st.sidebar.number_input("現金部位年化利率 (%)", min_value=0.0, max_value=10.0, value=2.0)
cash_daily_rate = (1 + cash_annual_rate/100)**(1/252) - 1

# 回測日期設定 (預設 2026/03/01)
default_start_date = datetime.date(2026, 3, 1)
start_date = st.sidebar.date_input("回測起始日期", default_start_date)
end_date = st.sidebar.date_input("回測結束日期", datetime.date.today())

if start_date >= end_date:
    st.sidebar.error("錯誤：起始日期必須早於結束日期。")

# ==================== 2. 完美的資料下載與攤平 ====================
@st.cache_data(ttl=3600)
def load_data():
    data = {}
    for name, ticker in symbols.items():
        try:
            # 抓取最大範圍，確保有足夠的歷史資料來計算動能與均線
            df = yf.download(ticker, period="max", progress=False)
            if df.empty:
                continue
                
            # 強制處理所有可能的 yfinance 多層索引
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.levels:
                    series = df['Close']
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]
                    data[name] = series
                else:
                    series = df.xs('Close', axis=1, level=1)
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]
                    data[name] = series
            else:
                data[name] = df["Close"]
                
        except Exception as e:
            st.sidebar.warning(f"下載 {name} ({ticker}) 失敗: {e}")
            
    df_all = pd.DataFrame(data)
    # 用正向填補與反向填補解決所有的假日期/周末開盤空值問題
    df_all = df_all.ffill().bfill()
    return df_all

try:
    with st.spinner('⏳ 正在從 Yahoo Finance 下載數據並進行智能對齊...'):
        df_all = load_data()
    
    # 檢查是否有核心數據
    if "SPY" not in df_all.columns or "VIX" not in df_all.columns:
        st.error("❌ 讀取失敗：無法獲取 SPY 或 VIX 等關鍵標的！")
        st.info("💡 解決方案：請嘗試點擊網頁右上角的 `⋮` -> `Clear cache`。")
    else:
        # 計算技術指標
        momentum = df_all.pct_change(momentum_period)
        df_all["BTC_MA100"] = df_all["BTC"].rolling(btc_ma_period).mean()
        
        # 日期篩選
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        backtest_data = df_all.loc[start_ts:end_ts]
        momentum_sliced = momentum.loc[start_ts:end_ts]
        
        if len(backtest_data) < 2:
            st.error(f"❌ 查無資料：在所選的日期區間（{start_date} ~ {end_date}）內資料量不足！")
            st.info("💡 解決方案：由於 2026/03/01 剛過不久，資料太少。請在左側邊欄，將【回測起始日期】手動調前到 2025 年。")
        else:
            # ==================== 3. 今日策略面板 ====================
            today = backtest_data.index[-1]
            
            vix = backtest_data.loc[today, "VIX"]
            btc = backtest_data.loc[today, "BTC"]
            btc_ma = backtest_data.loc[today, "BTC_MA100"]
            
            valid_assets = [s for s in symbols.keys() if s not in ["VIX", "SPY"]]
            mom_today = momentum_sliced.loc[today, valid_assets].sort_values(ascending=False)
            
            p1 = mom_today.index[0]
            p2 = mom_today.index[1]
            
            p1_mom_val = mom_today.iloc[0]
            p2_mom_val = mom_today.iloc[1]
            
            # 動能小於 0，強制轉現金
            if p1_mom_val < 0: p1 = "CASH"
            if p2_mom_val < 0: p2 = "CASH"
                
            # 比特幣均線濾網
            if btc < btc_ma:
                if p1 == "BTC": p1 = "QQQ" if momentum_sliced.loc[today, "QQQ"] >= 0 else "CASH"
                if p2 == "BTC": p2 = "QQQ" if momentum_sliced.loc[today, "QQQ"] >= 0 else "CASH"
                    
            # 槓桿機制
            leverage = 2.5
            if vix > 40: leverage = 0.5
            elif vix > 30: leverage = 0.6
            elif vix > 20: leverage = 0.8
                
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 今日策略訊號")
                st.write(f"📅 訊號日期: {today.strftime('%Y-%m-%d')}")
                
                p1_display = f"{p1_mom_val:.2%}" if p1 != "CASH" else "N/A (轉入防禦)"
                p2_display = f"{p2_mom_val:.2%}" if p2 != "CASH" else "N/A (轉入防禦)"
                
                st.write(f"🥇 第一名（60%）: **{p1}** (動能: {p1_display})")
                st.write(f"🥈 第二名（40%）: **{p2}** (動能: {p2_display})")
                st.markdown(f"### 🔥 當前建議槓桿：{leverage}x")
                
            with col2:
                st.subheader("⚠️ 風控狀態")
                st.write(f"當前 VIX 指數: **{vix:.2f}**")
                st.write(f"BTC 價格: **{btc:.2f}** / {btc_ma_period}日均線: **{btc_ma:.2f}**")
                if vix > 30:
                    st.error("🚨 市場極度恐慌（觸發防禦性降槓）")
                elif vix > 20:
                    st.warning("⚠️ 市場波動加劇（適度調降槓桿）")
                else:
                    st.success("✅ 市場狀態正常（維持積極槓桿）")
                    
            # ==================== 4. 回測區域 ====================
            st.markdown("---")
            st.subheader(f"📈 策略回測 (從 {start_date.strftime('%Y-%m-%d')} 開始) vs. S&P 500")
            
            # 使用每日百分比變動 (T+1 的報酬率)
            daily_pct = backtest_data[valid_assets].pct_change().shift(-1)
            strategy_returns = []
            
            # 遍歷回測區間（排除最後一天，因為最後一天沒有 T+1 日的報酬率）
            for date in backtest_data.index[:-1]:
                vix_t = backtest_data.loc[date, "VIX"]
                btc_t = backtest_data.loc[date, "BTC"]
                btc_ma_t = backtest_data.loc[date, "BTC_MA100"]
                
                mom_t = momentum_sliced.loc[date, valid_assets].sort_values(ascending=False)
                
                pick1 = mom_t.index[0]
                pick2 = mom_t.index[1]
                
                # 負動能轉現金
                if mom_t.iloc[0] < 0: pick1 = "CASH"
                if mom_t.iloc[1] < 0: pick2 = "CASH"
                    
                # 比特幣濾網
                if btc_t < btc_ma_t:
                    if pick1 == "BTC": pick1 = "QQQ" if momentum_sliced.loc[date, "QQQ"] >= 0 else "CASH"
                    if pick2 == "BTC": pick2 = "QQQ" if momentum_sliced.loc[date, "QQQ"] >= 0 else "CASH"
                    
                # 槓桿
                lev_t = 2.5
                if vix_t > 40: lev_t = 0.5
                elif vix_t > 30: lev_t = 0.6
                elif vix_t > 20: lev_t = 0.8
                    
                # 獲取 T+1 的報酬
                ret1 = daily_pct.loc[date, pick1] if pick1 != "CASH" else cash_daily_rate
                ret2 = daily_pct.loc[date, pick2] if pick2 != "CASH" else cash_daily_rate
                
                # 若 T+1 剛好是空值 (通常不會發生，保險起見)，用 0 取代
                if np.isnan(ret1): ret1 = 0
                if np.isnan(ret2): ret2 = 0
                
                day_ret = (0.6 * ret1 + 0.4 * ret2) * lev_t
                strategy_returns.append({'Date': date, 'Strategy_Return': day_ret})
                
            strat_df = pd.DataFrame(strategy_returns).set_index('Date')
            
            # 獨立計算 SPY (不參與 join 排除，防止因假日過濾掉資料)
            spy_pct = backtest_data['SPY'].pct_change().loc[strat_df.index].fillna(0)
            strat_df['SPY_Return'] = spy_pct
            
            # 計算累積報酬
            strat_df['Cumulative_Strategy'] = (1 + strat_df['Strategy_Return']).cumprod() - 1
            strat_df['Cumulative_SPY'] = (1 + strat_df['SPY_Return']).cumprod() - 1
            
            display_df = strat_df[['Cumulative_Strategy', 'Cumulative_SPY']] * 100
            
            # 終極過濾：確保圖表 DataFrame 不是空的
            if display_df.dropna().empty:
                st.warning("⚠️ 產生的回測數據全為空值。可能是所選日期區間無有效交易數據。")
            else:
                st.line_chart(display_df)
                
                # -------------------- 績效指標 --------------------
                days = len(strat_df)
                years = days / 252 if days > 0 else 1
                
                def calculate_metrics(returns_series):
                    if returns_series.dropna().empty:
                        return 0.0, 0.0, 0.0
                    # 年化報酬
                    cagr = ((1 + returns_series).prod())**(1/years) - 1
                    # 年化波動
                    vol = returns_series.std() * np.sqrt(252)
                    # 夏普值 (取 0 波動防呆)
                    sharpe = (cagr - (cash_annual_rate/100)) / vol if vol > 0 else 0
                    # 最大回撤
                    cum_ret = (1 + returns_series).cumprod()
                    peak = cum_ret.cummax()
                    drawdown = (cum_ret - peak) / peak
                    mdd = drawdown.min()
                    return cagr, sharpe, mdd

                strat_cagr, strat_sharpe, strat_mdd = calculate_metrics(strat_df['Strategy_Return'])
                spy_cagr, spy_sharpe, spy_mdd = calculate_metrics(strat_df['SPY_Return'])
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("平均年化報酬率 (CAGR)", f"{strat_cagr:.2%}", f"對比 SPY: {spy_cagr:.2%}")
                with m2:
                    st.metric("夏普值 (Sharpe Ratio)", f"{strat_sharpe:.2f}", f"對比 SPY: {spy_sharpe:.2f}")
                with m3:
                    st.metric("最大回撤 (MDD)", f"{strat_mdd:.2%}", f"對比 SPY: {spy_mdd:.2%}")
                    
                final_strat = display_df['Cumulative_Strategy'].iloc[-1]
                final_spy = display_df['Cumulative_SPY'].iloc[-1]
                
                c1, c2 = st.columns(2)
                c1.metric("SEGM 策略總累積報酬", f"{final_strat:.2f}%", f"{(final_strat - final_spy):.2f}% (勝過 SPY)")
                c2.metric("S&P 500 (SPY) 總累積報酬", f"{final_spy:.2f}%")

except Exception as e:
    st.error(f"⚠️ 發生底層執行錯誤: {e}")
