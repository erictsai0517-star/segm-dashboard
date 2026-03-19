import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="SEGM 投資儀表板", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚀 SEGM 投資模型</h1>", unsafe_allow_html=True)

# 側邊欄
st.sidebar.header("⚙️ 模型設定")
start_capital = st.sidebar.number_input("初始投入資金 (USD)", value=100000, step=10000)
target_goal = 100000000 # 一億美金目標

@st.cache_data(ttl=3600)
def get_clean_data():
    tickers = {"BTC": "BTC-USD", "QQQ": "QQQ", "GOLD": "GLD", "VIX": "^VIX", "UUP": "UUP"}
    data = pd.DataFrame()
    for name, sym in tickers.items():
        try:
            t = yf.download(sym, period="3y", interval="1d", progress=False)['Close']
            if isinstance(t, pd.DataFrame): t = t.iloc[:, 0]
            data[name] = t
        except: continue
    return data.dropna()

df = get_clean_data()
if df.empty: st.stop()

# 策略邏輯
df['BTC_MA100'] = df['BTC'].rolling(100).mean()
mom_df = df[['BTC', 'QQQ', 'GOLD']].pct_change(60)
returns = df[['BTC', 'QQQ', 'GOLD', 'UUP']].pct_change()

equity = [float(start_capital)]
start_date = df.index[100].strftime('%Y-%m-%d')

for i in range(100, len(df)):
    d_vix = df.iloc[i]['VIX']
    d_lev = 2.5
    if d_vix > 40: d_lev = 0.5
    elif d_vix > 30: d_lev = 0.6
    elif d_vix > 20: d_lev = 0.8
    
    d_mom = mom_df.iloc[i].sort_values(ascending=False)
    m1, m2 = d_mom.index, d_mom.index
    
    if df.iloc[i]['BTC'] < df.iloc[i]['BTC_MA100']:
        if m1 == "BTC": m1 = "GOLD" if d_mom['GOLD'] > d_mom['QQQ'] else "QQQ"
        if m2 == "BTC": m2 = "UUP"

    day_ret = (returns.iloc[i][m1] * 0.6 + returns.iloc[i][m2] * 0.4) * d_lev
    equity.append(equity[-1] * (1 + day_ret))

equity_series = pd.Series(equity, index=df.index[99:])

# =========================
# 介面排列
# =========================
st.write(f"📅 **數據回測起始日：{start_date}** (包含 100 天指標預熱期)")

# 今日指令
st.divider()
st.subheader("🎯 今日操作指令")
c1, c2, c3 = st.columns(3)
c1.metric("建議槓桿", f"{d_lev}x")
c2.info(f"🥇 重倉 (60%): **{m1}**")
c3.info(f"🥈 配倉 (40%): **{m2}**")

# 圖表
st.divider()
st.subheader("📈 複利成長曲線")
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values, mode='lines', line=dict(color='#00ff88', width=2.5)))
fig.update_layout(template="plotly_dark", hovermode="x unified", dragmode="pan", height=500)
st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

# 數據統計
k1, k2, k3 = st.columns(3)
total_return = (equity_series.iloc[-1] / start_capital) - 1
annual_return = (1 + total_return) ** (252 / len(equity_series)) - 1
mdd = (equity_series / equity_series.cummax() - 1).min()

k1.metric("目前帳戶價值", f"${int(equity_series.iloc[-1]):,} USD")
k2.metric("年化報酬率", f"{round(annual_return*100, 2)}%")
k3.metric("歷史最大回撤", f"{round(mdd*100, 2)}%", delta_color="inverse")

# =========================
# 終極目標看板 (置底)
# =========================
st.divider()
st.subheader("🏁 SEGM 終極財富目標")
current_val = equity_series.iloc[-1]
progress = min(current_val / target_goal, 1.0)
st.progress(progress)
st.write(f"目前已達成 **1 億美金** 目標的 **{round(progress*100, 4)}%**")
st.info("💡 **模型核心目的：** 透過風控避開黑天鵝，以 2.5x 槓桿實現跨世代的資產階級跳躍。")
