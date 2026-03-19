import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# 1. 基礎設定
st.set_page_config(page_title="SEGM 投資儀表板", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚀 SEGM 投資模型</h1>", unsafe_allow_html=True)

# 側邊欄
st.sidebar.header("⚙️ 模型設定")
start_capital = st.sidebar.number_input("初始投入資金 (USD)", value=100000, step=10000)

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

# 策略運算邏輯
df['BTC_MA100'] = df['BTC'].rolling(100).mean()
mom_df = df[['BTC', 'QQQ', 'GOLD']].pct_change(60)
returns = df[['BTC', 'QQQ', 'GOLD', 'UUP']].pct_change()

equity = [float(start_capital)]
trade_history = []

for i in range(100, len(df)):
    d_vix = df.iloc[i]['VIX']
    d_lev = 2.5
    if d_vix > 40: d_lev = 0.5
    elif d_vix > 30: d_lev = 0.6
    elif d_vix > 20: d_lev = 0.8
    
    d_mom = mom_df.iloc[i].sort_values(ascending=False)
    m1, m2 = d_mom.index[0], d_mom.index[1]
    
    # BTC 避險過濾
    if df.iloc[i]['BTC'] < df.iloc[i]['BTC_MA100']:
        if m1 == "BTC": m1 = "GOLD" if d_mom['GOLD'] > d_mom['QQQ'] else "QQQ"
        if m2 == "BTC": m2 = "UUP"

    day_ret = (returns.iloc[i][m1] * 0.6 + returns.iloc[i][m2] * 0.4) * d_lev
    equity.append(equity[-1] * (1 + day_ret))

equity_series = pd.Series(equity, index=df.index[99:])

# =========================
# UI 調整：今日指令移至最上方
# =========================
st.divider()
st.subheader("🎯 今日操作指令")
c1, c2, c3 = st.columns(3)
c1.metric("建議槓桿", f"{d_lev}x")
c2.info(f"🥇 重倉 (60%): **{m1}**")
c3.info(f"🥈 配倉 (40%): **{m2}**")

# 指標統計
st.divider()
k1, k2, k3 = st.columns(3)
total_return = (equity_series.iloc[-1] / start_capital) - 1
annual_return = (1 + total_return) ** (252 / len(equity_series)) - 1
mdd = (equity_series / equity_series.cummax() - 1).min()

k1.metric("累計資產", f"${int(equity_series.iloc[-1]):,} USD")
k2.metric("年化報酬", f"{round(annual_return*100, 2)}%")
k3.metric("最大回撤", f"{round(mdd*100, 2)}%", delta_color="inverse")

# =========================
# 圖表優化：自由縮放版
# =========================
st.subheader("📈 歷史淨值曲線 (可雙手/滾輪縮放)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values, 
                         mode='lines', name='淨值',
                         line=dict(color='#00ff88', width=2.5),
                         fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.05)'))

fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    dragmode="pan",  # 預設為平移，方便手機操作
    height=600,
    xaxis=dict(showgrid=False, rangeslider=dict(visible=True)), # 加入下方的時間拉條
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', fixedrange=False), # 縱軸允許縮放
    margin=dict(l=10, r=10, t=10, b=10)
)

# 啟用滾輪縮放
st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

st.caption("💡 提示：使用滑鼠滾輪或兩指在圖表上縮放；點擊下方滑動條可快速切換時間區段。")
