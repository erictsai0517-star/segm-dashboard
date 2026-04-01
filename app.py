import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

st.set_page_config(page_title="SEGM 投資儀表板 & 回測", layout="wide")
st.title("🚀 SEGM 投資儀表板 (含多空標的、現金防禦與回測)")

# ==================== 1. 設定與參數 ====================
# 新增做空與避險類型的標的
symbols = {
    "QQQ": "QQQ",          # 科技股大盤
    "BTC": "BTC-USD",      # 比特幣
    "TQQQ": "TQQQ",        # 3倍做多 QQQ
    "SQQQ": "SQQQ",        # 3倍做空 QQQ (做空類)
    "SH": "SH",            # 1倍做空 S&P500 (做空類)
    "TLT": "TLT",          # 20年美債 (避險類)
    "VIX": "^VIX"          # 恐慌指數
}

# 側邊欄設定
st.sidebar.header("⚙️ 策略與回測設定")
momentum_period = st.sidebar.number_input("動能週期 (天數)", min_value=10, max_value=200, value=60)
btc_ma_period = st.sidebar.number_input("BTC 均線週期 (天數)", min_value=10, max_value=200, value=100)

# 現金(無風險)年化利率設定 (例如 4%)
cash_annual_rate = st.sidebar.number_input("現金部位年化利率 (%)", min_value=0.0, max_value=10.0, value=0.0)
cash_daily_rate = (1 + cash_annual_rate/100)**(1/252) - 1

# 回測日期設定 (預設從 2026-03-01 開始)
default_start_date = datetime.date(2026, 3, 1)
start_date = st.sidebar.date_input("回測起始日期", default_start_date)
end_date = st.sidebar.date_input("回測結束日期", datetime.date.today())

if start_date >= end_date:
    st.sidebar.error("錯誤：起始日期必須早於結束日期。")

# ==================== 2. 資料下載與處理 ====================
@st.cache_data(ttl=3600)
def load_data():
    data = {}
    # 下載足夠長的歷史資料以滿足移動平均線與動能計算
    for name, ticker in symbols.items():
        df = yf.download(ticker, period="max", progress=False)
        # 處理 yfinance 新版的 multi-index 欄位問題
        if isinstance(df.columns, pd.MultiIndex):
            data[name] = df["Close"][ticker]
        else:
            data[name] = df["Close"]
            
    df_all = pd.DataFrame(data).dropna()
    return df_all

try:
    df_all = load_data()
    
    # 計算策略所需的指標
    momentum = df_all.pct_change(momentum_period)
    df_all["BTC_MA100"] = df_all["BTC"].rolling(btc_ma_period).mean()
    
    # 篩選出回測日期範圍的資料
    backtest_data = df_all.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)].dropna()
    
    if backtest_data.empty:
        st.warning("⚠️ 選擇的日期範圍內沒有足夠的資料，請調整日期或等待美股開盤。")
    else:
        # ==================== 3. 今日策略面板 ====================
        today = backtest_data.index[-1]
        
        vix = backtest_data.loc[today, "VIX"]
        btc = backtest_data.loc[today, "BTC"]
        btc_ma = backtest_data.loc[today, "BTC_MA100"]
        
        # 取得有效風險資產的動能
        valid_assets = [s for s in symbols.keys() if s != "VIX"]
        mom_today = momentum.loc[today, valid_assets].sort_values(ascending=False)
        
        p1 = mom_today.index[0]
        p2 = mom_today.index[1]
        
        # 【新增機制】若動能為負，則替換為現金 (CASH)
        p1_mom_val = mom_today[p1]
        p2_mom_val = mom_today[p2]
        
        if p1_mom_val < 0:
            p1 = "CASH"
        if p2_mom_val < 0:
            p2 = "CASH"
            
        # 濾網：若 BTC 低於均線，且排名前二有 BTC，則用 QQQ 取代（若 QQQ 動能也小於 0 則會變成 CASH）
        if btc < btc_ma:
            if p1 == "BTC":
                p1 = "QQQ" if momentum.loc[today, "QQQ"] >= 0 else "CASH"
            if p2 == "BTC":
                p2 = "QQQ" if momentum.loc[today, "QQQ"] >= 0 else "CASH"
                
        # VIX 槓桿濾網
        leverage = 2.5
        if vix > 40:
            leverage = 0.5
        elif vix > 30:
            leverage = 0.6
        elif vix > 20:
            leverage = 0.8
            
        # 顯示今日面板
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
                
        # ==================== 4. 2026/03/01 起回測區域 ====================
        st.markdown("---")
        st.subheader(f"📈 策略回測 (從 {start_date.strftime('%Y-%m-%d')} 開始)")
        
        # 計算每日策略報酬率
        # 為了模擬真實情況，策略會在 T 日收盤計算動能，並在 T+1 日獲取報酬
        daily_pct = backtest_data[valid_assets].pct_change().shift(-1)
        
        strategy_returns = []
        
        # 逐日計算回測
        for date in backtest_data.index[:-1]:
            vix_t = backtest_data.loc[date, "VIX"]
            btc_t = backtest_data.loc[date, "BTC"]
            btc_ma_t = backtest_data.loc[date, "BTC_MA100"]
            
            # 排行
            mom_t = momentum.loc[date, valid_assets].sort_values(ascending=False)
            pick1 = mom_t.index[0]
            pick2 = mom_t.index[1]
            
            # 負動能轉現金機制
            if mom_t[pick1] < 0:
                pick1 = "CASH"
            if mom_t[pick2] < 0:
                pick2 = "CASH"
                
            # BTC 均線濾網
            if btc_t < btc_ma_t:
                if pick1 == "BTC": 
                    pick1 = "QQQ" if momentum.loc[date, "QQQ"] >= 0 else "CASH"
                if pick2 == "BTC": 
                    pick2 = "QQQ" if momentum.loc[date, "QQQ"] >= 0 else "CASH"
                
            # VIX 槓桿設定
            lev_t = 2.5
            if vix_t > 40: lev_t = 0.5
            elif vix_t > 30: lev_t = 0.6
            elif vix_t > 20: lev_t = 0.8
                
            # 取得 T+1 日的標的報酬率 (若是 CASH 則使用 daily_rate)
            ret1 = daily_pct.loc[date, pick1] if pick1 != "CASH" else cash_daily_rate
            ret2 = daily_pct.loc[date, pick2] if pick2 != "CASH" else cash_daily_rate
            
            # 計算當日總報酬
            day_ret = (0.6 * ret1 + 0.4 * ret2) * lev_t
            strategy_returns.append({'Date': date, 'Strategy_Return': day_ret})
            
        # 轉換為 DataFrame 並繪製圖表
        strat_df = pd.DataFrame(strategy_returns).set_index('Date')
        strat_df = strat_df.dropna()
        
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
    st.error(f"發生程式執行錯誤: {e}")
    st.info("提示：若遇到 Yahoo Finance 讀取失敗，請重新整理頁面。")
