import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ------------------------
# 1️⃣ 基礎設定
# ------------------------
st.set_page_config(page_title="SEGM 投資儀表板", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚀 SEGM 投資模型</h1>", unsafe_allow_html=True)

# 側邊欄：初始資金 & 模型設定
st.sidebar.header("⚙️ 模型設定")
start_capital = st.sidebar.number_input("初始投入資金 (USD)", value=100000, step=10000)
start_date = st.sidebar.date_input("回測起始日期", value=datetime.today())
agg_mode = st.sidebar.checkbox("開啟 2.5x 激進模式", value=True)

target_goal = 100_000_000  # 1 億美元目標
fee_rate = 0.0005  # 手續費 0.05%

max_lev = 2.5 if agg_mode else 1.0

# ------------------------
# 2️⃣ 取得歷史數據
# ------------------------
@st.cache_data(ttl=3600)
def get_data():
    tickers = {"BTC": "BTC-USD", "QQQ": "QQQ", "GOLD": "GLD", "VIX": "^VIX", "UUP": "UUP"}
    data = pd.DataFrame()
    for name, sym in tickers.items():
        try:
            t = yf.download(sym, period="3y", interval="1d", progress=False)['Close']
            if isinstance(t, pd.DataFrame): t = t.iloc[:,0]
            data[name] = t
        except:
            continue
    return data.dropna()

df_full = get_data()
if df_full.empty:
    st.error("數據抓取失敗")
    st.stop()

# ------------------------
# 3️⃣ 計算指標
# ------------------------
df_full['BTC_MA100'] = df_full['BTC'].rolling(100).mean()
mom_df = df_full[['BTC','QQQ','GOLD']].pct_change(60)
returns = df_full[['BTC','QQQ','GOLD','UUP']].pct_change()

# 只保留從回測起始日開始的資料顯示
df = df_full[df_full.index >= pd.to_datetime(start_date)]

# ------------------------
# 4️⃣ 回測
# ------------------------
m1, m2, d_lev = "QQQ", "BTC", 1.0  # 今日策略預設
equity = [float(start_capital)]

for i in range(100, len(df_full)):
    row_vix = df_full.iloc[i]['VIX']
    row_btc = df_full.iloc[i]['BTC']
    row_ma  = df_full.iloc[i]['BTC_MA100']

    # 動量排名
    current_mom = mom_df.iloc[i].sort_values(ascending=False)
    m1_tmp = current_mom.index[0]
    m2_tmp = current_mom.index[1]

    # BTC 均線避險
    if row_btc < row_ma:
        if m1_tmp == "BTC":
            m1_tmp = "GOLD" if current_mom['GOLD'] > current_mom['QQQ'] else "QQQ"
        if m2_tmp == "BTC":
            m2_tmp = "UUP"

    # VIX 調整槓桿
    lev = max_lev
    if row_vix > 40: lev = 0.5
    elif row_vix > 30: lev = 0.6
    elif row_vix > 20: lev = 0.8

    # 計算當日回報，扣手續費
    day_ret = (returns.iloc[i][m1_tmp]*0.6 + returns.iloc[i][m2_tmp]*0.4)*lev
    day_ret -= fee_rate  # 扣掉手續費
    equity.append(equity[-1]*(1 + day_ret))

    # 如果是回測起始日，存今日策略
    if df_full.index[i] == pd.to_datetime(start_date):
        m1, m2, d_lev = m1_tmp, m2_tmp, lev

equity_series = pd.Series(equity[-len(df):], index=df.index)

# ------------------------
# 5️⃣ 介面顯示
# ------------------------
st.write(f"📊 **初始本金：${start_capital:,} USD** | 🕒 **最後更新：{df_full.index[-1].strftime('%Y-%m-%d')}**")

# 今日策略
st.divider()
st.subheader("🎯 今日操作指令")
c1,c2,c3 = st.columns(3)
c1.metric("建議槓桿", f"{d_lev}x")
c2.info(f"🥇 第一名 (60%): **{m1}**")
c3.info(f"🥈 第二名 (40%): **{m2}**")

# 數據統計
st.divider()
k1,k2,k3 = st.columns(3)
total_return = (equity_series.iloc[-1]/start_capital)-1
annual_return = (1+total_return)**(252/len(equity_series))-1
mdd = (equity_series/equity_series.cummax()-1).min()

k1.metric("目前帳戶價值", f"${int(equity_series.iloc[-1]):,} USD")
k2.metric("年化報酬率", f"{round(annual_return*100,2)}%")
k3.metric("歷史最大回撤", f"{round(mdd*100,2)}%", delta_color="inverse")

# 複利成長曲線
st.subheader("📈 複利成長曲線 (可縮放)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values,
                         mode='lines', line=dict(color='#00ff88', width=2.5),
                         fill='tozeroy', fillcolor='rgba(0,255,136,0.05)'))
fig.update_layout(template="plotly_dark", hovermode="x unified",
                  dragmode="pan", height=600,
                  xaxis=dict(showgrid=False, rangeslider=dict(visible=True)),
                  yaxis=dict(showgrid=True, fixedrange=False))
st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

# 終極目標
st.divider()
st.subheader("🏁 SEGM 終極財富目標")
current_val = equity_series.iloc[-1]
progress = min(current_val/target_goal, 1.0)
st.progress(progress)
st.write(f"目前已達成 **1 億美金** 目標的 **{round(progress*100,6)}%**")
st.info("💡 模型核心：風控避黑天鵝 + 槓桿階級跨躍 + 複利累積")
st.warning("⚠️ 本網站僅展示策略，不構成投資建議，請自行決策。")
