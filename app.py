 import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# =========================
# 基礎設定
# =========================
st.set_page_config(page_title="SEGM 今日起始模擬", layout="wide")
st.title("🚀 SEGM 今日起始資金模擬")

# 側邊欄設定
st.sidebar.header("⚙️ 模型設定")
start_capital = st.sidebar.number_input("初始投入資金 (USD)", value=100000, step=10000)
start_date = pd.to_datetime(st.sidebar.date_input("模擬開始日期", value=pd.to_datetime("2026-03-19")))

max_lev = st.sidebar.slider("最高槓桿倍數", min_value=0.5, max_value=5.0, value=2.5, step=0.1)

# =========================
# 取得數據
# =========================
tickers = {"BTC": "BTC-USD", "QQQ": "QQQ", "GOLD": "GLD", "VIX": "^VIX", "UUP": "UUP"}
data = pd.DataFrame()

for name, sym in tickers.items():
    t = yf.download(sym, start=start_date, period="1y", interval="1d", progress=False)['Close']
    if isinstance(t, pd.DataFrame):
        t = t.iloc[:,0]
    data[name] = t

df = data.dropna()
if df.empty:
    st.error("❌ 從指定日期抓不到資料")
    st.stop()

# 計算指標
df['BTC_MA100'] = df['BTC'].rolling(100, min_periods=1).mean()
mom_df = df[['BTC','QQQ','GOLD']].pct_change(60)
returns = df[['BTC','QQQ','GOLD','UUP']].pct_change()

# =========================
# 回測核心
# =========================
equity = [float(start_capital)]
m1, m2, d_lev = None, None, max_lev

for i in range(1, len(df)):
    try:
        today_row = df.iloc[i]
        today_mom = mom_df.iloc[i].sort_values(ascending=False)
        m1 = today_mom.index[0]
        m2 = today_mom.index[1]

        # BTC 避險
        if today_row['BTC'] < today_row['BTC_MA100']:
            if m1 == "BTC":
                m1 = "GOLD" if today_mom.get('GOLD',0) > today_mom.get('QQQ',0) else "QQQ"
            if m2 == "BTC":
                m2 = "UUP"

        # VIX 調整槓桿
        d_lev = max_lev
        vix = today_row['VIX']
        if vix > 40: d_lev *= 0.2
        elif vix > 30: d_lev *= 0.24
        elif vix > 20: d_lev *= 0.32

        # 計算日回報
        day_ret = (returns.iloc[i][m1]*0.6 + returns.iloc[i][m2]*0.4) * d_lev
        equity.append(equity[-1]*(1+day_ret))

    except Exception:
        equity.append(equity[-1])  # 發生錯誤就資產保持不變

equity_series = pd.Series(equity, index=df.index)

# =========================
# 顯示結果
# =========================
st.subheader("📈 模擬資產成長曲線")
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values, mode='lines', fill='tozeroy', fillcolor='rgba(0,255,136,0.05)', line=dict(color='#00ff88', width=2.5)))
fig.update_layout(template="plotly_dark", height=600, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("💰 當前模擬資產")
st.write(f"從 {start_date.date()} 開始，本金 {start_capital:,} USD")
st.write(f"最新資產價值: **{int(equity_series.iloc[-1]):,} USD**")k2.metric("年化報酬率", f"{round(annual_return*100,2)}%")
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
