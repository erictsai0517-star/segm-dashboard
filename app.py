import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# =========================
# 基礎設定
# =========================
st.set_page_config(page_title="SEGM 投資儀表板", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚀 SEGM 投資模型</h1>", unsafe_allow_html=True)

# 側邊欄設定
st.sidebar.header("⚙️ 模型設定")
start_capital = st.sidebar.number_input("初始投入資金 (USD)", value=100000, step=10000)
start_date_str = st.sidebar.text_input("回測起始日期 (YYYY-MM-DD)", value="2020-01-01")
trade_fee = st.sidebar.number_input("單筆交易手續費 (%)", value=0.1, step=0.1) / 100
agg_mode = st.sidebar.checkbox("開啟 2.5x 激進模式", value=True)

target_goal = 100000000  # 一億美金目標

# =========================
# 抓數據
# =========================
@st.cache_data(ttl=3600)
def get_clean_data():
    tickers = {"BTC": "BTC-USD", "QQQ": "QQQ", "GOLD": "GLD", "VIX": "^VIX", "UUP": "UUP"}
    data = pd.DataFrame()
    for name, sym in tickers.items():
        try:
            t = yf.download(sym, period="3y", interval="1d", progress=False)['Close']
            if isinstance(t, pd.DataFrame):
                t = t.iloc[:, 0]
            data[name] = t
        except:
            continue
    return data.dropna()

df = get_clean_data()
if df.empty:
    st.error("數據抓取失敗，請重新整理頁面")
    st.stop()

# =========================
# 可選起始日期
# =========================
try:
    start_date = pd.to_datetime(start_date_str)
    df = df[df.index >= start_date]
except:
    st.warning("起始日期格式錯誤，使用全部資料")
    
if len(df) < 100:
    st.error("資料不足，無法回測")
    st.stop()

# =========================
# 計算指標
# =========================
df['BTC_MA100'] = df['BTC'].rolling(100).mean()
mom_df = df[['BTC','QQQ','GOLD']].pct_change(60)
returns = df[['BTC','QQQ','GOLD','UUP']].pct_change()
equity = [float(start_capital)]
last_idx = len(df)

# 設定最大槓桿
max_lev = 2.5 if agg_mode else 1.0

# =========================
# 回測循環
# =========================
for i in range(100, last_idx):
    row = df.iloc[i]
    
    # VIX槓桿邏輯
    d_lev = max_lev
    if row['VIX'] > 40: d_lev = max_lev*0.2
    elif row['VIX'] > 30: d_lev = max_lev*0.24
    elif row['VIX'] > 20: d_lev = max_lev*0.32
    
    # 動量排名
    current_mom = mom_df.iloc[i].sort_values(ascending=False)
    assets = current_mom.index.tolist()
    
    # BTC 均線避險
    m1 = assets[0] if len(assets) > 0 else "N/A"
    m2 = assets[1] if len(assets) > 1 else "N/A"
    
    if row['BTC'] < row['BTC_MA100']:
        if m1 == "BTC":
            m1 = "GOLD" if current_mom.get('GOLD',0) > current_mom.get('QQQ',0) else "QQQ"
        if m2 == "BTC":
            m2 = "UUP"
    
    # 當日回報計算
    day_ret = 0
    for asset, w in zip([m1,m2],[0.6,0.4]):
        if asset in returns.columns:
            day_ret += returns.iloc[i][asset]*w
    
    day_ret = day_ret*d_lev
    
    # 扣除交易手續費
    day_ret -= 2*trade_fee  # 假設兩筆資產都要手續費
    
    equity.append(equity[-1]*(1+day_ret))

equity_series = pd.Series(equity, index=df.index[99:])

# =========================
# 安全顯示今日操作
# =========================
safe_d_lev = d_lev if 'd_lev' in locals() else 1.0
safe_m1 = m1 if 'm1' in locals() else "N/A"
safe_m2 = m2 if 'm2' in locals() else "N/A"

st.write(f"📊 **初始本金：${start_capital:,} USD** | 🕒 **最後更新：{df.index[-1].strftime('%Y-%m-%d')}**")

st.divider()
st.subheader("🎯 今日操作指令")
c1, c2, c3 = st.columns(3)
c1.metric("建議槓桿", f"{safe_d_lev:.2f}x")
c2.info(f"🥇 第一名 (60%): **{safe_m1}**")
c3.info(f"🥈 第二名 (40%): **{safe_m2}**")

# =========================
# 數據統計
# =========================
st.divider()
k1, k2, k3 = st.columns(3)
total_return = (equity_series.iloc[-1]/start_capital)-1
annual_return = (1+total_return)**(252/len(equity_series))-1
mdd = (equity_series/equity_series.cummax()-1).min()

k1.metric("目前帳戶價值", f"${int(equity_series.iloc[-1]):,} USD")
k2.metric("年化報酬率", f"{round(annual_return*100,2)}%")
k3.metric("歷史最大回撤", f"{round(mdd*100,2)}%", delta_color="inverse")

# =========================
# 成長曲線
# =========================
st.subheader("📈 複利成長曲線")
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values,
                         mode='lines', line=dict(color='#00ff88', width=2.5),
                         fill='tozeroy', fillcolor='rgba(0,255,136,0.05)'))
fig.update_layout(template="plotly_dark", hovermode="x unified",
                  dragmode="pan", height=600,
                  xaxis=dict(showgrid=False,rangeslider=dict(visible=True)),
                  yaxis=dict(showgrid=True,fixedrange=False))
st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

# =========================
# 終極目標
# =========================
st.divider()
st.subheader("🏁 SEGM 終極財富目標")
current_val = equity_series.iloc[-1]
progress = min(current_val/target_goal,1.0)
st.progress(progress)
st.write(f"目前已達成 **1 億美金** 目標的 **{round(progress*100,6)}%**")
st.info("💡 **模型核心目的：** 用風控避開黑天鵝，以槓桿實現階級跨躍，回撤是複利的門票，風控是活命的保險，只為最後的數字服務。")
st.warning("⚠️ 本網站僅為策略展示，不構成任何投資建議。投資有風險，請自行決策。")
