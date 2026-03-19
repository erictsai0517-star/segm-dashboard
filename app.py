import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# 1. 設置
st.set_page_config(page_title="SEGM 投資儀表板", layout="wide")
st.title("🚀 SEGM 投資模型")

# 側邊欄控制
st.sidebar.header("⚙️ 模型設定")
start_capital = st.sidebar.number_input("初始投入資金 (USD)", value=100000, step=10000)

# =========================
# 數據抓取
# =========================
@st.cache_data(ttl=3600)
def get_clean_data():
    tickers = {"BTC": "BTC-USD", "QQQ": "QQQ", "GOLD": "GLD", "VIX": "^VIX", "UUP": "UUP"}
    data = pd.DataFrame()
    for name, sym in tickers.items():
        try:
            t = yf.download(sym, period="2y", interval="1d", progress=False)['Close']
            if isinstance(t, pd.DataFrame): t = t.iloc[:, 0]
            data[name] = t
        except: continue
    return data.dropna()

df = get_clean_data()
if df.empty: st.stop()

# =========================
# 策略邏輯
# =========================
df['BTC_MA100'] = df['BTC'].rolling(100).mean()
mom_df = df[['BTC', 'QQQ', 'GOLD']].pct_change(60)

# 模擬回測
returns = df[['BTC', 'QQQ', 'GOLD', 'UUP']].pct_change()
equity = [float(start_capital)]

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
    equity.append(equity[-1] * (1 + day_ret))

equity_series = pd.Series(equity, index=df.index[99:])

# 計算指標
total_return = (equity_series.iloc[-1] / start_capital) - 1
annual_return = (1 + total_return) ** (252 / len(equity_series)) - 1
mdd = (equity_series / equity_series.cummax() - 1).min()

# =========================
# 介面顯示
# =========================
st.subheader("📈 實戰淨值曲線 (TradingView 風格)")

# 使用 Plotly 建立專業圖表
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values, 
                         mode='lines', name='SEGM Equity',
                         line=dict(color='#00ff88', width=2),
                         fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)'))

fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    xaxis=dict(showgrid=False, title="日期"),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="帳戶價值 (USD)"),
    height=500,
    margin=dict(l=0, r=0, t=0, b=0)
)
st.plotly_chart(fig, use_container_width=True)

# 核心數據顯示
st.divider()
k1, k2, k3 = st.columns(3)
k1.metric("累計資產", f"${int(equity_series.iloc[-1]):,} USD")
k2.metric("預估年化報酬", f"{round(annual_return*100, 2)}%")
k3.metric("歷史最大回撤", f"{round(mdd*100, 2)}%", delta_color="inverse")

# 今日指令面板 (置底方便路人看)
st.success(f"🎯 今日指令：第一名 {m1} (60%) / 第二名 {m2} (40%) / 槓桿 {d_lev}x")
