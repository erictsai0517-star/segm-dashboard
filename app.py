import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="SEGM 2.0 投資儀表板", layout="wide")

st.title("🚀 SEGM 2.0 穩定極端成長模型")
st.caption("16歲量化開發版 - 自動化執行指令")

# =========================
# 1️⃣ 抓數據（改進版：確保欄位正確）
# =========================
# 注意：VIX 在 yfinance 建議用 '^VIX' 原始指數，VIXY 是 ETF 會有損耗
symbols = {
    "QQQ": "QQQ",
    "BTC": "BTC-USD",
    "VIX": "^VIX",
    "GOLD": "GLD",
    "UUP": "UUP"
}

@st.cache_data(ttl=3600) # 快取一小時，不用每次重新下載
def get_data():
    raw_df = yf.download(list(symbols.values()), period="2y", interval="1d", progress=False)
    
    # 處理 yfinance 的 MultiIndex 結構
    if isinstance(raw_df.columns, pd.MultiIndex):
        close_df = raw_df['Close']
    else:
        close_df = raw_df[['Close']]
    
    # 將代號換回我們好讀的名字 (BTC-USD -> BTC)
    inv_map = {v: k for k, v in symbols.items()}
    close_df = close_df.rename(columns=inv_map)
    return close_df.dropna()

df = get_data()

if df.empty or len(df.columns) < len(symbols):
    st.error("❌ 市場資料抓取失敗或不完整，請檢查網路或稍後再試")
    st.stop()

# =========================
# 2️⃣ 計算指標 (MA & Momentum)
# =========================
df['BTC_MA100'] = df['BTC'].rolling(100).mean()
# 計算 60 天報酬率作為動量排名
momentum_df = df[['QQQ', 'BTC', 'GOLD']].pct_change(60) 

# 取得最新一天的資料
today_idx = df.index[-1]
today_data = df.iloc[-1]
today_mom = momentum_df.iloc[-1].sort_values(ascending=False)

# =========================
# 3️⃣ 策略執行邏輯
# =========================
# 初始排名
p1 = today_mom.index[0]
p2 = today_mom.index[1]

# BTC 100MA 硬過濾
if today_data['BTC'] < today_data['BTC_MA100']:
    if p1 == "BTC": p1 = "GOLD" if today_mom['GOLD'] > today_mom['QQQ'] else "QQQ"
    if p2 == "BTC": p2 = "GOLD" if today_mom['GOLD'] > today_mom['QQQ'] else "QQQ"

# VIX 槓桿邏輯
vix_val = today_data['VIX']
leverage = 2.5
risk_level = "✅ 低風險"
status_color = "success"

if vix_val > 40:
    leverage, risk_level, status_color = 0.5, "🔥 極端恐慌 (減倉)", "error"
elif vix_val > 30:
    leverage, risk_level, status_color = 0.6, "⚠️ 高風險 (防禦)", "warning"
elif vix_val > 20:
    leverage, risk_level, status_color = 0.8, "⚡ 中等風險 (警戒)", "info"

# =========================
# 4️⃣ 顯示路人看板
# =========================
col1, col2 = st.columns(2)

with col1:
    st.metric("當前建議槓桿", f"{leverage}x")
    st.write(f"當前 VIX 指數: **{round(vix_val, 2)}**")

with col2:
    if status_color == "success": st.success(risk_level)
    elif status_color == "info": st.info(risk_level)
    elif status_color == "warning": st.warning(risk_level)
    else: st.error(risk_level)

st.divider()

st.subheader("📍 今日操作指令")
c1, c2 = st.columns(2)
c1.info(f"🥇 第一名 (60%): **{p1}**")
c2.info(f"🥈 第二名 (40%): **{p2}**")

# =========================
# 5️⃣ 💰 實戰資產模擬 (10萬)
# =========================
st.divider()
st.subheader("📈 SEGM 2.0 實戰模擬 (初始 $100,000)")

returns = df[['QQQ', 'BTC', 'GOLD', 'UUP']].pct_change()
capital = 100000
equity = [capital]

# 從有資料的地方開始跑回測
for i in range(100, len(df)):
    current_vix = df.iloc[i]['VIX']
    current_btc = df.iloc[i]['BTC']
    current_ma = df.iloc[i]['BTC_MA100']
    
    # 決定當天槓桿
    lev = 2.5
    if current_vix > 40: lev = 0.5
    elif current_vix > 30: lev = 0.6
    elif current_vix > 20: lev = 0.8
    
    # 決定當天持倉 (簡化模擬)
    daily_mom = momentum_df.iloc[i].sort_values(ascending=False)
    m1, m2 = daily_mom.index[0], daily_mom.index[1]
    
    # BTC 過濾
    if current_btc < current_ma:
        if m1 == "BTC": m1 = "GOLD"
        if m2 == "BTC": m2 = "QQQ"

    # 計算當日回報
    day_ret = (returns.iloc[i][m1] * 0.6 + returns.iloc[i][m2] * 0.4) * lev
    equity.append(equity[-1] * (1 + day_ret))

equity_series = pd.Series(equity, index=df.index[99:])
st.line_chart(equity_series)

final_value = equity_series.iloc[-1]
st.write(f"### 💰 當前資產淨值: **${int(final_value):,} USD**")

# 計算最大回撤
roll_max = equity_series.cummax()
dd = (equity_series / roll_max - 1).min()
st.write(f"📉 歷史最大回撤: **{round(dd*100, 2)}%**")
