import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. 設置標題為 SEGM
st.set_page_config(page_title="SEGM 投資儀表板", layout="wide")
st.title("🚀 SEGM 投資模型")
st.caption("穩定極端成長模型 - 實戰監控版")

# =========================
# 數據抓取功能 (修正 TypeError)
# =========================
@st.cache_data(ttl=3600)
def get_clean_data():
    tickers = {"BTC": "BTC-USD", "QQQ": "QQQ", "GOLD": "GLD", "VIX": "^VIX", "UUP": "UUP"}
    data = pd.DataFrame()
    for name, sym in tickers.items():
        try:
            # 抓取最近 2 年數據以計算 100MA
            t = yf.download(sym, period="2y", interval="1d", progress=False)['Close']
            if isinstance(t, pd.DataFrame): t = t.iloc[:, 0]
            data[name] = t
        except:
            continue
    return data.dropna()

df = get_clean_data()

if df.empty:
    st.error("❌ 無法獲取市場數據，請檢查網路連線。")
    st.stop()

# =========================
# 策略邏輯運算
# =========================
df['BTC_MA100'] = df['BTC'].rolling(100).mean()
mom_df = df[['BTC', 'QQQ', 'GOLD']].pct_change(60) # 60天動量

# 獲取「今天」的即時狀態
now = df.iloc[-1]
now_mom = mom_df.iloc[-1].sort_values(ascending=False)

# 判斷槓桿與風險
vix_val = now['VIX']
leverage = 2.5
if vix_val > 40: leverage = 0.5
elif vix_val > 30: leverage = 0.6
elif vix_val > 20: leverage = 0.8

# 動態選股 (第一名/第二名)
p1, p2 = now_mom.index[0], now_mom.index[1]
if now['BTC'] < now['BTC_MA100']:
    if p1 == "BTC": p1 = "GOLD" if now_mom['GOLD'] > now_mom['QQQ'] else "QQQ"
    if p2 == "BTC": p2 = "QQQ"

# =========================
# 交易歷史與指標紀錄
# =========================
returns = df[['BTC', 'QQQ', 'GOLD', 'UUP']].pct_change()
equity = [100000.0]
history = []

for i in range(100, len(df)):
    d_vix = df.iloc[i]['VIX']
    d_lev = 2.5
    if d_vix > 40: d_lev = 0.5
    elif d_vix > 30: d_lev = 0.6
    elif d_vix > 20: d_lev = 0.8
    
    d_mom = mom_df.iloc[i].sort_values(ascending=False)
    m1, m2 = d_mom.index[0], d_mom.index[1]
    
    if df.iloc[i]['BTC'] < df.iloc[i]['BTC_MA100']:
        if m1 == "BTC": m1 = "GOLD"
        if m2 == "BTC": m2 = "QQQ"

    day_ret = (returns.iloc[i][m1] * 0.6 + returns.iloc[i][m2] * 0.4) * d_lev
    new_equity = equity[-1] * (1 + day_ret)
    equity.append(new_equity)
    
    # 紀錄換倉歷史 (僅紀錄變動時)
    if i == 100 or m1 != history[-1]['第一名']:
        history.append({"日期": df.index[i].strftime('%Y-%m-%d'), "第一名": m1, "第二名": m2, "槓桿": d_lev, "淨值": int(new_equity)})

# 計算核心指標
equity_series = pd.Series(equity)
total_return = (equity_series.iloc[-1] / 100000) - 1
annual_return = (1 + total_return) ** (252 / len(equity_series)) - 1
mdd = (equity_series / equity_series.cummax() - 1).min()

# =========================
# 視覺化介面
# =========================
st.subheader(f"📅 今日指令：{df.index[-1].strftime('%Y-%m-%d')}")
c1, c2, c3 = st.columns(3)
c1.metric("建議槓桿", f"{leverage}x")
c2.info(f"🥇 重倉(60%): {p1}")
c3.info(f"🥈 配倉(40%): {p2}")

st.divider()

# 績效看板
k1, k2, k3 = st.columns(3)
k1.metric("累計資產", f"${int(equity[-1]):,} USD")
k2.metric("預估年化報酬", f"{round(annual_return*100, 2)}%")
k3.metric("歷史最大回撤", f"{round(mdd*100, 2)}%", delta_color="inverse")

st.line_chart(pd.Series(equity, index=df.index[99:]))

# 顯示每筆交易歷史
with st.expander("📜 查看交易紀錄與換倉歷史"):
    st.table(pd.DataFrame(history).tail(10)) # 顯示最近10次變動
