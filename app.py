import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="SEGM 投資儀表板", layout="centered")

st.title("🚀 SEGM 投資儀表板")

# =========================
# 1️⃣ 抓數據（穩定版）
# =========================
symbols = {
    "QQQ": "QQQ",
    "BTC": "BTC-USD",
    "VIX": "VIXY"
}

data = {}

for name, ticker in symbols.items():
    try:
        df_temp = yf.download(ticker, period="1y", progress=False)

        # ❗ 完全防呆
        if df_temp is None or df_temp.empty:
            continue

        # 處理 MultiIndex（雲端常見）
        if isinstance(df_temp.columns, pd.MultiIndex):
            close = df_temp.xs("Close", level=0, axis=1)
        else:
            if "Close" not in df_temp.columns:
                continue
            close = df_temp["Close"]

        # ❗ 確保是 Series
        if isinstance(close, pd.Series):
            data[name] = close

    except Exception as e:
        st.write(f"{name} error:", e)

# ❗ 至少要兩個資產
if len(data) < 2:
    st.error("❌ 無法取得足夠市場資料")
    st.stop()

df = pd.DataFrame(data)

# ❗ 再保險
if df.empty:
    st.error("❌ DataFrame 為空")
    st.stop()

df = df.dropna()

# =========================
# 3️⃣ 策略邏輯
# =========================
vix = df.loc[today, "VIX"]
btc = df.loc[today, "BTC"]
btc_ma = df.loc[today, "BTC_MA100"]

mom = momentum.loc[today].dropna().sort_values(ascending=False)

p1 = mom.index[0]
p2 = mom.index[1]

# BTC 過濾
if btc < btc_ma:
    if p1 == "BTC":
        p1 = "QQQ"
    if p2 == "BTC":
        p2 = "QQQ"

# 槓桿
leverage = 2.5
if vix > 40:
    leverage = 0.5
elif vix > 30:
    leverage = 0.6
elif vix > 20:
    leverage = 0.8

# =========================
# 4️⃣ 顯示策略
# =========================
st.subheader("📊 今日策略")

st.markdown(f"### 🥇 第一名（60%）：**{p1}**")
st.markdown(f"### 🥈 第二名（40%）：**{p2}**")

st.markdown(f"# 🔥 槓桿：{leverage}x")

# 風險狀態
if vix > 40:
    st.error("🔥 極端恐慌（極限防禦）")
elif vix > 30:
    st.warning("⚠️ 高風險（降槓）")
elif vix > 20:
    st.info("⚡ 中等風險")
else:
    st.success("✅ 低風險（正常）")

# =========================
# 5️⃣ 💰 資金模擬（10萬）
# =========================
st.subheader("💰 虛擬資金模擬（10萬美金）")

returns = df.pct_change()

capital = 100000
equity = [capital]

for i in range(100, len(df)):
    mom = momentum.iloc[i].dropna().sort_values(ascending=False)

    if len(mom) < 2:
        equity.append(equity[-1])
        continue

    p1 = mom.index[0]
    p2 = mom.index[1]

    btc = df.iloc[i]["BTC"]
    btc_ma = df.iloc[i]["BTC_MA100"]

    if btc < btc_ma:
        if p1 == "BTC":
            p1 = "QQQ"
        if p2 == "BTC":
            p2 = "QQQ"

    weights = {p1: 0.6, p2: 0.4}

    vix = df.iloc[i]["VIX"]

    lev = 2.5
    if vix > 40:
        lev = 0.5
    elif vix > 30:
        lev = 0.6
    elif vix > 20:
        lev = 0.8

    daily_r = 0
    for asset, w in weights.items():
        if asset in returns.columns:
            daily_r += returns.iloc[i][asset] * w

    daily_r *= lev

    equity.append(equity[-1] * (1 + daily_r))

equity_series = pd.Series(equity)

# 顯示
st.line_chart(equity_series)

st.markdown(f"### 💰 當前資產：{int(equity_series.iloc[-1]):,} USD")

# 最大回撤
roll_max = equity_series.cummax()
drawdown = equity_series / roll_max - 1
mdd = drawdown.min()

st.markdown(f"### 📉 最大回撤：{round(mdd*100,2)}%")
