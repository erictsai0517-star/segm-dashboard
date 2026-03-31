import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

st.set_page_config(page_title="SEGM 投資儀表板 & 回測", layout="wide")
st.title("🚀 SEGM 投資儀表板 (含多空標的、現金防禦與回測)")

# ==================== 1. 設定與參數 ====================
symbols = {
    "QQQ": "QQQ",          # 科技股大盤
    "BTC": "BTC-USD",      # 比特幣
    "TQQQ": "TQQQ",        # 3倍做多 QQQ
    "SQQQ": "SQQQ",        # 3倍做空 QQQ (做空類)
    "SH": "SH",            # 1倍做空 S&P500 (做空類)
    "TLT": "TLT",          # 20年美債 (避險類)
    "VIX": "^VIX"          # 恐慌指數
}

st.sidebar.header("⚙️ 策略與回測設定")
momentum_period = st.sidebar.number_input("動能週期 (天數)", min_value=10, max_value=200, value=60)
btc_ma_period = st.sidebar.number_input("BTC 均線週期 (天數)", min_value=10, max_value=200, value=100)

cash_annual_rate = st.sidebar.number_input("現金部位年化利率 (%)", min_value=0.0, max_value=10.0, value=0.0)
cash_daily_rate = (1 + cash_annual_rate/100)**(1/252) - 1

# 回測日期設定
default_start_date = datetime.date(2026, 3, 1)
start_date = st.sidebar.date_input("回測起始日期", default_start_date)
end_date = st.sidebar.date_input("回測結束日期", datetime.date.today())

if start_date >= end_date:
    st.sidebar.error("錯誤：起始日期必須早於結束日期。")

# ==================== 2. 極度穩定的資料下載與對齊 ====================
@st.cache_data(ttl=3600)
def load_data():
    data = {}
    for name, ticker in symbols.items():
        try:
            # 抓取 period="max" 確保有足夠的歷史資料來計算動能與移動平均線
            df = yf.download(ticker, period="max", progress=False)
            
            if df.empty:
                st.sidebar.warning(f"⚠️ 標的 {name} 沒有抓到歷史資料，請確認網路或 Ticker。")
                continue
                
            # 【終極欄位解構】處理 yfinance 新舊版本多層索引的各種狀況
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.levels[0]:
                    # 如果 Close 在第一層
                    series = df['Close']
                    if isinstance(series, pd.DataFrame):
                        # 如果裡面依然有股票代號層，取第一個 column
                        series = series.iloc[:, 0]
                    data[name] = series
                else:
                    # 如果 Close 在第二層
                    series = df.xs('Close', axis=1, level=1)
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]
                    data[name] = series
            else:
                # 傳統單層索引
                data[name] = df["Close"]
                
        except Exception as e:
            st.sidebar.warning(f"下載 {name} ({ticker}) 失敗: {e}")
            
    # 合併所有 Series
    df_all = pd.DataFrame(data)
    
    # 【關鍵修正】不要用 dropna()！改用 ffill 填補假日的空值 (例如周六日美股休市，但比特幣有開市)
    df_all = df_all.ffill().bfill()
    
    return df_all

try:
    with st.spinner('⏳ 正在從 Yahoo Finance 下載完整歷史數據（請稍候，防閃退處理中）...'):
        df_all = load_data()
    
    if df_all.empty or "VIX" not in df_all.columns:
        st.error("❌ 讀取失敗：無法從 Yahoo Finance 獲取關鍵標的！")
        st.info("💡 建議：請檢查本地網路，或是重啟 Streamlit 伺服器清除快取。")
    else:
        # 1. 在完整的歷史數據上計算指標 (防止日期被切後長度不足算不出來)
        momentum = df_all.pct_change(momentum_period)
        df_all["BTC_MA100"] = df_all["BTC"].rolling(btc_ma_period).mean()
        
        # 2. 進行日期切片 (僅篩選回測時段)
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        backtest_data = df_all.loc[start_ts:end_ts]
        momentum_sliced = momentum.loc[start_ts:end_ts]
        
        if backtest_data.empty:
            st.error(f"❌ 查無資料：在所選的日期區間（{start_date} ~ {end_date}）內沒有資料！")
            st.info("💡 解決方案：請嘗試在側邊欄將【回測起始日期】往前調整到 2025 年甚至更早。")
        else:
            # ==================== 3. 今日策略面板 ====================
            today = backtest_data.index[-1]
            
            vix = backtest_data.loc[today, "VIX"]
            btc = backtest_data.loc[today, "BTC"]
            btc_ma = backtest_data.loc[today, "BTC_MA100"]
            
            valid_assets = [s for s in symbols.keys() if s != "VIX"]
            mom_today = momentum_sliced.loc[today, valid_assets].sort_values(ascending=False)
            
            p1 = mom_today.index[0]
            p2 = mom_today.index[1]
            
            p1_mom_val = mom_today[p1]
            p2_mom_val = mom_today[p2]
            
            # 現金防衛
            if p1_mom_val < 0: p1 = "CASH"
            if p2_mom_val < 0: p2 = "CASH"
                
            # 均線防衛
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
            st.subheader(f"📈 策略回測 (從 {start_date.strftime('%Y-%m-%d')} 開始)")
            
            daily_pct = backtest_data[valid_assets].pct_change().shift(-1)
            strategy_returns = []
            
            for date in backtest_data.index[:-1]:
                vix_t = backtest_data.loc[date, "VIX"]
                btc_t = backtest_data.loc[date, "BTC"]
                btc_ma_t = backtest_data.loc[date, "BTC_MA100"]
                
                mom_t = momentum_sliced.loc[date, valid_assets].sort_values(ascending=False)
                pick1 = mom_t.index[0]
                pick2 = mom_t.index[1]
                
                if mom_t[pick1] < 0: pick1 = "CASH"
                if mom_t[pick2] < 0: pick2 = "CASH"
                    
                if btc_t < btc_ma_t:
                    if pick1 == "BTC": pick1 = "QQQ" if momentum_sliced.loc[date, "QQQ"] >= 0 else "CASH"
                    if pick2 == "BTC": pick2 = "QQQ" if momentum_sliced.loc[date, "QQQ"] >= 0 else "CASH"
                    
                lev_t = 2.5
                if vix_t > 40: lev_t = 0.5
                elif vix_t > 30: lev_t = 0.6
                elif vix_t > 20: lev_t = 0.8
                    
                ret1 = daily_pct.loc[date, pick1] if pick1 != "CASH" else cash_daily_rate
                ret2 = daily_pct.loc[date, pick2] if pick2 != "CASH" else cash_daily_rate
                
                day_ret = (0.6 * ret1 + 0.4 * ret2) * lev_t
                strategy_returns.append({'Date': date, 'Strategy_Return': day_ret})
                
            strat_df = pd.DataFrame(strategy_returns).set_index('Date').dropna()
            strat_df['Cumulative_Strategy'] = (1 + strat_df['Strategy_Return']).cumprod() - 1
            
            qqq_pct = backtest_data['QQQ'].pct_change().loc[strat_df.index]
            strat_df['Cumulative_QQQ'] = (1 + qqq_pct).cumprod() - 1
            
            display_df = strat_df[['Cumulative_Strategy', 'Cumulative_QQQ']] * 100
            st.line_chart(display_df)
            
            final_strat = display_df['Cumulative_Strategy'].iloc[-1]
            final_qqq = display_df['Cumulative_QQQ'].iloc[-1]
            
            c1, c2 = st.columns(2)
            c1.metric("SEGM 策略總報酬率", f"{final_strat:.2f}%", f"{(final_strat - final_qqq):.2f}% (勝過 QQQ)")
            c2.metric("QQQ 基準總報酬率", f"{final_qqq:.2f}%")

except Exception as e:
    st.error(f"⚠️ 發生程式執行錯誤: {e}")
