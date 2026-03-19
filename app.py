
# =========================
# 1️⃣ 基礎設定
# =========================
st.set_page_config(page_title="SEGM 投資儀表板", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚀 SEGM 投資模型</h1>", unsafe_allow_html=True)

# 側邊欄設定
st.sidebar.header("⚙️ 模型設定")
start_capital = st.sidebar.number_input("初始投入資金 (USD)", value=100000, step=10000)
agg_mode = st.sidebar.checkbox("開啟 2.5x 激進模式", value=True)
start_date = st.sidebar.date_input("回測起始日", value=pd.to_datetime("2023-01-01"))

max_lev = 2.5 if agg_mode else 1.0  # 槓桿上限
target_goal = 100_000_000  # 一億美金

# 手續費選項
fee_pct = st.sidebar.number_input("交易手續費 (%) 每次換倉", value=0.05, step=0.01)/100

# =========================
# 2️⃣ 抓取數據
# =========================
@st.cache_data(ttl=3600)
def get_data(start_date):
    tickers = {"BTC": "BTC-USD", "QQQ": "QQQ", "GOLD": "GLD", "VIX": "^VIX", "UUP": "UUP"}
    data = pd.DataFrame()
    for name, sym in tickers.items():
        try:
            t = yf.download(sym, start=start_date - pd.Timedelta(days=200), interval="1d", progress=False)['Close']
            if isinstance(t, pd.DataFrame):
                t = t.iloc[:,0]
            data[name] = t
        except: continue
    return data.dropna()

df = get_data(start_date)
if df.empty:
    st.error("❌ 資料抓取失敗")
    st.stop()

# =========================
# 3️⃣ 計算指標
# =========================
df['BTC_MA100'] = df['BTC'].rolling(100).mean()
mom_df = df[['BTC', 'QQQ', 'GOLD']].pct_change(60)
returns = df[['BTC', 'QQQ', 'GOLD', 'UUP']].pct_change()

# 只從使用者選的日期開始回測
df = df[df.index >= pd.to_datetime(start_date)]
mom_df = mom_df.loc[df.index]
returns = returns.loc[df.index]

# =========================
# 4️⃣ 回測模擬
# =========================
equity = [float(start_capital)]
last_idx = len(df)

for i in range(100, last_idx):
    row_vix = df.iloc[i]['VIX']
    row_btc = df.iloc[i]['BTC']
    row_ma  = df.iloc[i]['BTC_MA100']

    # VIX 降槓桿
    d_lev = max_lev
    if row_vix > 40: d_lev = 0.5
    elif row_vix > 30: d_lev = 0.6
    elif row_vix > 20: d_lev = 0.8

    # 動量排名
    current_mom = mom_df.iloc[i].sort_values(ascending=False)
    m1 = current_mom.index[0]
    m2 = current_mom.index[1]

    # BTC 均線避險
    if row_btc < row_ma:
        if m1 == "BTC":
            m1 = "GOLD" if current_mom['GOLD'] > current_mom['QQQ'] else "QQQ"
        if m2 == "BTC":
            m2 = "UUP"

    # 計算回報
    day_ret = (returns.iloc[i][m1]*0.6 + returns.iloc[i][m2]*0.4) * d_lev

    # 模擬手續費：假設每天換倉會扣一次手續費
    day_ret -= fee_pct * 2  # 一次買一次賣
    equity.append(equity[-1]*(1+day_ret))

equity_series = pd.Series(equity, index=df.index[99:])

# =========================
# 5️⃣ 介面顯示
# =========================
st.write(f"📊 **初始本金：${start_capital:,} USD** | 🕒 最後更新：{df.index[-1].strftime('%Y-%m-%d')}")

# 今日指令
st.divider()
st.subheader("🎯 今日操作指令")
c1, c2, c3 = st.columns(3)
c1.metric("建議槓桿", f"{d_lev}x")
c2.info(f"🥇 第一名 (60%): **{m1}**")
c3.info(f"🥈 第二名 (40%): **{m2}**")

# 統計
st.divider()
k1, k2, k3 = st.columns(3)
total_return = (equity_series.iloc[-1]/start_capital)-1
annual_return = (1 + total_return)**(252/len(equity_series)) -1
mdd = (equity_series/equity_series.cummax()-1).min()
k1.metric("目前帳戶價值", f"${int(equity_series.iloc[-1]):,} USD")
k2.metric("年化報酬率", f"{round(annual_return*100,2)}%")
k3.metric("歷史最大回撤", f"{round(mdd*100,2)}%", delta_color="inverse")

# 複利曲線
st.subheader("📈 複利成長曲線")
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values,
                         mode='lines', line=dict(color='#00ff88', width=2.5),
                         fill='tozeroy', fillcolor='rgba(0,255,136,0.05)'))
fig.update_layout(template="plotly_dark", hovermode="x unified", dragmode="pan",
                  height=600, xaxis=dict(rangeslider=dict(visible=True)))
st.plotly_chart(fig, use_container_width=True)

# 終極目標
st.divider()
st.subheader("🏁 終極財富目標")
current_val = equity_series.iloc[-1]
progress = min(current_val / target_goal, 1.0)
st.progress(progress)
st.write(f"目前已達成 1 億美金目標的 **{round(progress*100,6)}%**")
st.info("💡 模型核心：風控避開黑天鵝，槓桿加速複利，回撤是複利的門票。")
st.warning("⚠️ 僅策略展示，不構成投資建議。投資有風險，請自行決策。")    x=equity_series.index,
    y=equity_series.values,
    mode='lines',
    line=dict(color='#00ff88', width=2.5),
    fill='tozeroy',
    fillcolor='rgba(0, 255, 136, 0.05)'
))
fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    dragmode="pan",
    height=600,
    xaxis=dict(showgrid=False, rangeslider=dict(visible=True)),
    yaxis=dict(showgrid=True, fixedrange=False)
)
st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})


