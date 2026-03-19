import streamlit as st
import yfinance as yf
import pandas as pd

st.title("🚀 SEGM 投資儀表板")

symbols = {
    "QQQ": "QQQ",
    "BTC": "BTC-USD",
    "VIX": "^VIX"
}

data = {}
for name, ticker in symbols.items():
    df = yf.download(ticker, period="200d", progress=False)
    data[name] = df["Close"]

df = pd.DataFrame(data).dropna()

momentum = df.pct_change(60)
df["BTC_MA100"] = df["BTC"].rolling(100).mean()

today = df.index[-1]

vix = df.loc[today, "VIX"]
btc = df.loc[today, "BTC"]
btc_ma = df.loc[today, "BTC_MA100"]

mom = momentum.loc[today].sort_values(ascending=False)

p1 = mom.index[0]
p2 = mom.index[1]

if btc < btc_ma:
    if p1 == "BTC":
        p1 = "QQQ"
    if p2 == "BTC":
        p2 = "QQQ"

leverage = 2.5
if vix > 40:
    leverage = 0.5
elif vix > 30:
    leverage = 0.6
elif vix > 20:
    leverage = 0.8

st.subheader("📊 今日策略")

st.write(f"🥇 第一名（60%）: {p1}")
st.write(f"🥈 第二名（40%）: {p2}")

st.markdown(f"# 🔥 槓桿：{leverage}x")

if vix > 30:
    st.error("⚠️ 市場恐慌（降槓）")
else:
    st.success("✅ 正常狀態")
