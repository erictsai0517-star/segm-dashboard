import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="SEGM 2.0 終極儀表板", layout="wide")
st.title("🚀 SEGM 2.0 穩定極端成長模型")
st.write("目前狀態：**自動化即時監控中**")

# =========================
# 1️⃣ 強力抓數據 (分開抓取避開 MultiIndex 錯誤)
# =========================
@st.cache_data(ttl=3600)
def get_clean_data():
    # 資產清單
    tickers = {"BTC": "BTC-USD", "QQQ": "QQQ", "GOLD": "GLD", "VIX": "^VIX"}
    combined = pd.DataFrame()
    
    for name, sym in tickers.items():
        try:
            temp = yf.download(sym, period="2y", interval="1d", progress=False)['Close']
            # yfinance 有時返回 Series，有時 DataFrame，強制轉為 Series
            if isinstance(temp, pd.DataFrame):
                temp = temp.iloc[:, 0]
            combined[name] = temp
        except:
            st.warning(f"⚠️ {name} 抓取微延遲，嘗試修正中...")
            
    return combined.dropna()

df = get_clean_data()

# 檢查數據是否完整
if df.empty or "VIX" not in df.columns:
    st.error("❌ 數據庫連線異常。原因：Yahoo Finance 暫時封鎖請求或網路不穩。請重新整理網頁。")
    st.stop()

# =========================
# 2️⃣ 指標計算
# =========================
df['BTC_MA100'] = df['BTC'].rolling(100).mean()
# 60天動量 (動能選股)
mom_df = df[['BTC', 'QQQ', 'GOLD']].pct_change(60)

# 最新數據
now = df.iloc[-1]
now_mom = mom_df.iloc[-1].sort_values(ascending=False)

# =========================
# 3️⃣ 核心邏輯：SEGM 2.0 指令
# =========================
vix_val = now['VIX']
leverage = 2.5
risk_msg = "✅ 市場穩定：全力進攻"

# VIX 階梯降槓
if vix_val > 40:
    leverage, risk_msg = 0.5, "🔥 極端恐慌：強制減倉 50%"
elif vix_val > 30:
    leverage, risk_msg = 0.6, "⚠️ 高度風險：防禦模式 0.6x"
elif vix_val > 20:
    leverage, risk_msg = 0.8, "⚡ 中度警戒：縮減槓桿 0.8x"

# 動態選股 (第一名/第二名)
p1, p2 = now_mom.index[0], now_mom.index[1]

# BTC 硬過濾 (2.0 核心)
if now['BTC'] < now['BTC_MA100']:
    if p1 == "BTC": p1 = "GOLD" if now_mom['GOLD'] > now_mom['QQQ'] else "QQQ"
    if p2 == "BTC": p2 = "QQQ"

# =========================
# 4️⃣ 介面顯示 (路人看板)
# =========================
col1, col2, col3 = st.columns(3)
col1.metric("今日建議槓桿", f"{leverage}x")
col2.metric("恐慌指數 VIX", f"{round(vix_val, 1)}")
col3.subheader(risk_msg)

st.divider()

c1, c2 = st.columns(2)
c1.info(f"🥇 第一名重倉 (60%)：**{p1}**")
c2.info(f"🥈 第二名配倉 (40%)：**{p2}**")

# =========================
# 5️⃣ 實戰 10 萬美金模擬
# =========================
st.subheader("💰 SEGM 2.0 實戰淨值走勢 (10萬美金起點)")
returns = df[['BTC', 'QQQ', 'GOLD']].pct_change()
equity = [100000.0]

for i in range(100, len(df)):
    d_vix = df.iloc[i]['VIX']
    d_lev = 2.5
    if d_vix > 40: d_lev = 0.5
    elif d_vix > 30: d_lev = 0.6
    elif d_vix > 20: d_lev = 0.8
    
    # 簡化當日回報計算
    d_mom = mom_df.iloc[i].sort_values(ascending=False)
    m1, m2 = d_mom.index[0], d_mom.index[1]
    
    # BTC 過濾
    if df.iloc[i]['BTC'] < df.iloc[i]['BTC_MA100']:
        if m1 == "BTC": m1 = "GOLD"
        if m2 == "BTC": m2 = "QQQ"
        
    day_ret = (returns.iloc[i][m1] * 0.6 + returns.iloc[i][m2] * 0.4) * d_lev
    equity.append(equity[-1] * (1 + day_ret))

equity_df = pd.DataFrame({"Net Worth": equity}, index=df.index[99:])
st.line_chart(equity_df)

cur_val = int(equity[-1])
st.write(f"### 🚀 目前帳戶總值：**${cur_val:,} USD**")
