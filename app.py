import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SEGM 投資儀表板", layout="wide")
st.title("🚀 SEGM 投資儀表板（凱利動態槓桿版）")

# ==================== 標的設定 ====================
TICKER_MAP = {
    "QQQ":  "QQQ",
    "BTC":  "BTC-USD",
    "TQQQ": "TQQQ",
    "SQQQ": "SQQQ",
    "SH":   "SH",
    "TLT":  "TLT",
    "GLD":  "GLD",
    "USO":  "USO",
    "SPY":  "SPY",
    "VIX":  "^VIX",
}
TRADABLE = ["QQQ", "TQQQ", "BTC", "TLT", "GLD", "USO"]

# ==================== 側邊欄 ====================
st.sidebar.header("⚙️ 策略設定")
momentum_period  = st.sidebar.number_input("動能週期（天）",    10, 200, 25)
btc_ma_period    = st.sidebar.number_input("BTC 均線週期（天）", 10, 200, 50)
cash_annual_rate = st.sidebar.number_input("現金年化利率（%）",  0.0, 10.0, 2.0)
cash_daily_rate  = (1 + cash_annual_rate / 100) ** (1/252) - 1

st.sidebar.markdown("---")
st.sidebar.markdown("**⚡ 凱利槓桿設定**")
kelly_window  = st.sidebar.number_input("凱利計算回望窗口（天）", 20, 120, 60)
kelly_fraction = st.sidebar.slider("凱利分數（1.0=全凱利，0.5=半凱利）", 0.1, 1.0, 0.5, 0.05)
kelly_max     = st.sidebar.slider("槓桿上限", 1.0, 4.0, 3.0, 0.1)
kelly_min     = st.sidebar.slider("槓桿下限", 0.1, 1.0, 0.2, 0.1)
st.sidebar.caption("凱利公式根據近期勝率和賠率動態計算最佳槓桿，數學上能最大化長期終值")

st.sidebar.markdown("---")
st.sidebar.markdown("**📉 VIX 緊急降槓**")
st.sidebar.caption("只在極端恐慌時才介入，平常交給凱利決定")
vix_panic     = st.sidebar.slider("VIX 緊急上限（超過此值強制降槓）", 30, 60, 45, 1)
vix_panic_lev = st.sidebar.slider("VIX 觸發後槓桿上限", 0.1, 1.0, 0.3, 0.1)

st.sidebar.markdown("---")
start_date = st.sidebar.date_input("回測起始日期", datetime.date(2020, 1, 1))
end_date   = st.sidebar.date_input("回測結束日期", datetime.date.today())

if start_date >= end_date:
    st.sidebar.error("起始日期必須早於結束日期")
    st.stop()

# ==================== 資料下載 ====================
@st.cache_data(ttl=3600)
def load_data(ticker_map: dict) -> pd.DataFrame:
    series_dict = {}
    failed = []
    for name, ticker in ticker_map.items():
        try:
            raw = yf.download(ticker, period="max", progress=False, auto_adjust=True)
            if raw.empty:
                failed.append(name)
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.get_level_values(0):
                    series_dict[name] = raw["Close"].squeeze()
                else:
                    failed.append(f"{name}(無Close)")
            else:
                if "Close" in raw.columns:
                    series_dict[name] = raw["Close"]
                else:
                    failed.append(f"{name}(無Close)")
        except Exception as e:
            failed.append(f"{name}({str(e)[:40]})")
    if failed:
        st.sidebar.warning(f"下載失敗：{', '.join(failed)}")
    return pd.DataFrame(series_dict).ffill().bfill()


def kelly_leverage(ret_series: pd.Series, fraction: float,
                   lev_min: float, lev_max: float) -> float:
    """
    凱利公式：f = μ / σ²
    其中 μ = 近期平均日報酬，σ² = 近期日報酬變異數
    乘上 fraction（半凱利 = 0.5）後 clip 到 [lev_min, lev_max]

    數學原理：最大化 E[log(終值)] = 最大化長期複利終值
    超過凱利值的槓桿會讓終值下降，低於凱利值的槓桿不夠積極
    """
    r = ret_series.dropna()
    if len(r) < 5:
        return fraction  # 資料不足時用保守預設

    mu    = r.mean()
    sigma2 = r.var()

    if sigma2 <= 0 or mu <= 0:
        return lev_min  # 負期望報酬不該加槓桿

    kelly = mu / sigma2  # 每日凱利槓桿
    kelly = kelly * fraction  # 用分數凱利降低波動
    return float(np.clip(kelly, lev_min, lev_max))


def calc_metrics(ret: pd.Series, rf_daily: float = 0.0) -> dict:
    r = ret.dropna()
    if len(r) < 2:
        return {k: 0 for k in ["total","cagr","vol","sharpe","mdd","winrate"]}
    n       = len(r)
    total   = (1 + r).prod() - 1
    cagr    = (1 + total) ** (252/n) - 1
    vol     = r.std() * np.sqrt(252)
    sharpe  = (cagr - rf_daily*252) / vol if vol > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = (cum / cum.cummax() - 1).min()
    winrate = (r > 0).mean()
    return {"total": total, "cagr": cagr, "vol": vol,
            "sharpe": sharpe, "mdd": mdd, "winrate": winrate}


# ==================== 主程式 ====================
try:
    with st.spinner("📡 載入市場資料中..."):
        df_all = load_data(TICKER_MAP)

    missing = [c for c in ["VIX", "BTC", "QQQ", "SPY"] if c not in df_all.columns]
    if missing:
        st.error(f"❌ 缺少關鍵欄位：{missing}")
        st.stop()

    df_all["BTC_MA"] = df_all["BTC"].rolling(btc_ma_period).mean()
    momentum = df_all[TRADABLE].pct_change(momentum_period)

    bt = df_all.loc[str(start_date): str(end_date)].copy()
    bt = bt.dropna(subset=["VIX", "BTC", "BTC_MA"])

    if len(bt) < 5:
        st.error(f"❌ 資料不足（{len(bt)}天），請調整日期範圍。")
        st.stop()

    # ==================== 今日訊號 ====================
    today      = bt.index[-1]
    vix_now    = bt.loc[today, "VIX"]
    btc_now    = bt.loc[today, "BTC"]
    btc_ma_now = bt.loc[today, "BTC_MA"]

    mom_today = momentum.loc[today, TRADABLE].dropna().sort_values(ascending=False)
    p1 = mom_today.index[0] if len(mom_today) > 0 else "CASH"
    p2 = mom_today.index[1] if len(mom_today) > 1 else "CASH"
    m1_val = mom_today.iloc[0] if len(mom_today) > 0 else -1
    m2_val = mom_today.iloc[1] if len(mom_today) > 1 else -1

    if m1_val < 0: p1 = "CASH"
    if m2_val < 0: p2 = "CASH"

    if btc_now < btc_ma_now:
        qqq_m = momentum.loc[today, "QQQ"] if "QQQ" in momentum.columns else -1
        if p1 == "BTC": p1 = "QQQ" if qqq_m >= 0 else "CASH"
        if p2 == "BTC": p2 = "QQQ" if qqq_m >= 0 else "CASH"

    # 今日凱利槓桿（用過去 kelly_window 天的策略報酬估算）
    # 回測前先用 QQQ 近期報酬代理估算
    recent_qqq = df_all["QQQ"].pct_change().iloc[-kelly_window:]
    lev_now = kelly_leverage(recent_qqq, kelly_fraction, kelly_min, kelly_max)
    if vix_now > vix_panic:
        lev_now = min(lev_now, vix_panic_lev)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 今日策略訊號")
        st.write(f"📅 訊號日期：{today.strftime('%Y-%m-%d')}")
        st.write(f"🥇 第一名（60%）：**{p1}**　動能：{m1_val:.2%}" if p1 != "CASH" else "🥇 第一名（60%）：**現金防禦**")
        st.write(f"🥈 第二名（40%）：**{p2}**　動能：{m2_val:.2%}" if p2 != "CASH" else "🥈 第二名（40%）：**現金防禦**")
        st.markdown(f"### 🔥 凱利建議槓桿：{lev_now:.2f}x")
        st.caption(f"基於近 {kelly_window} 天報酬計算，{kelly_fraction} 分數凱利")

    with col2:
        st.subheader("⚠️ 風控狀態")
        st.write(f"VIX：**{vix_now:.2f}**")
        st.write(f"BTC：**{btc_now:,.0f}** / {btc_ma_period}日均線：**{btc_ma_now:,.0f}**")
        if vix_now > vix_panic: st.error(f"🚨 VIX>{vix_panic} 緊急降槓至 {vix_panic_lev}x")
        elif vix_now > 30:      st.warning("⚠️ 市場波動加劇，凱利會自動縮槓")
        else:                   st.success("✅ 市場狀態正常")

    st.markdown("**📋 當日動能排名**")
    rank_df = pd.DataFrame({
        "標的": mom_today.index,
        "動能": [f"{v:.2%}" for v in mom_today.values],
        "狀態": ["✅ 正動能" if v > 0 else "❌ 負動能" for v in mom_today.values],
    })
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    # ==================== 回測引擎 ====================
    st.markdown("---")
    st.subheader(f"📈 策略回測（{start_date} ～ {end_date}）vs SPY")

    asset_ret    = df_all[TRADABLE].pct_change()
    spy_ret      = df_all["SPY"].pct_change()
    strat_ret_ts = pd.Series(dtype=float)  # 滾動累積策略報酬，給凱利公式用

    rows  = []
    dates = bt.index[:-1]

    for date in dates:
        if date not in momentum.index:
            continue
        loc = df_all.index.get_loc(date)
        if loc + 1 >= len(df_all):
            continue
        next_day = df_all.index[loc + 1]

        # ── 選股 ──
        mom_t = momentum.loc[date, TRADABLE].dropna().sort_values(ascending=False)
        if len(mom_t) < 2:
            continue

        pk1 = mom_t.index[0]; pk2 = mom_t.index[1]
        if mom_t.iloc[0] < 0: pk1 = "CASH"
        if mom_t.iloc[1] < 0: pk2 = "CASH"

        btc_t    = bt.loc[date, "BTC"]
        btc_ma_t = bt.loc[date, "BTC_MA"]
        if btc_t < btc_ma_t:
            qqq_m = momentum.loc[date, "QQQ"]
            if pk1 == "BTC": pk1 = "QQQ" if qqq_m >= 0 else "CASH"
            if pk2 == "BTC": pk2 = "QQQ" if qqq_m >= 0 else "CASH"

        # ── 凱利槓桿（用過去 kelly_window 天的策略實際報酬） ──
        if len(strat_ret_ts) >= kelly_window:
            window_ret = strat_ret_ts.iloc[-kelly_window:]
        elif len(strat_ret_ts) >= 5:
            window_ret = strat_ret_ts  # 資料不足 window 時用全部
        else:
            # 最初幾天用持倉標的歷史報酬代理
            proxy = asset_ret[pk1 if pk1 != "CASH" else "QQQ"]
            window_ret = proxy.loc[:date].iloc[-kelly_window:]

        lev_t = kelly_leverage(window_ret, kelly_fraction, kelly_min, kelly_max)

        # VIX 緊急上限
        vix_t = bt.loc[date, "VIX"]
        if vix_t > vix_panic:
            lev_t = min(lev_t, vix_panic_lev)

        # ── 次日報酬 ──
        def next_ret(pick):
            if pick == "CASH":
                return cash_daily_rate
            if pick not in asset_ret.columns or next_day not in asset_ret.index:
                return 0.0
            v = asset_ret.loc[next_day, pick]
            return 0.0 if pd.isna(v) else float(v)

        r1 = next_ret(pk1)
        r2 = next_ret(pk2)

        risky   = (0.6 * r1 if pk1 != "CASH" else 0) + (0.4 * r2 if pk2 != "CASH" else 0)
        cash_r  = (0.6 * cash_daily_rate if pk1 == "CASH" else 0) + \
                  (0.4 * cash_daily_rate if pk2 == "CASH" else 0)
        day_ret = risky * lev_t + cash_r

        # 把這天的報酬加進滾動序列，讓下一天凱利可以用
        strat_ret_ts = pd.concat([strat_ret_ts, pd.Series([day_ret], index=[date])])

        rows.append({"Date": date, "ret": day_ret, "pk1": pk1, "pk2": pk2, "lev": lev_t})

    if not rows:
        st.error("❌ 回測區間內沒有有效資料")
        st.stop()

    res = pd.DataFrame(rows).set_index("Date")
    res["cum_strat"] = (1 + res["ret"]).cumprod() - 1
    res["cum_spy"]   = (1 + spy_ret.reindex(res.index).fillna(0)).cumprod() - 1

    # ── 圖表 ──
    chart = (res[["cum_strat", "cum_spy"]] * 100).rename(
        columns={"cum_strat": "SEGM策略", "cum_spy": "SPY基準"}
    )
    st.line_chart(chart, use_container_width=True)

    # ── 槓桿歷史圖 ──
    with st.expander("📊 槓桿歷史（凱利動態槓桿走勢）"):
        st.line_chart(res["lev"].rename("凱利槓桿"), use_container_width=True)
        st.caption(f"平均槓桿：{res['lev'].mean():.2f}x　最高：{res['lev'].max():.2f}x　最低：{res['lev'].min():.2f}x")

    # ── 績效摘要 ──
    m_s = calc_metrics(res["ret"],                            cash_daily_rate)
    m_b = calc_metrics(spy_ret.reindex(res.index).fillna(0), cash_daily_rate)

    st.subheader("📊 績效摘要")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("年化報酬（CAGR）", f"{m_s['cagr']:.2%}",    f"SPY: {m_b['cagr']:.2%}")
    c2.metric("夏普比率",         f"{m_s['sharpe']:.2f}",  f"SPY: {m_b['sharpe']:.2f}")
    c3.metric("最大回撤（MDD）",  f"{m_s['mdd']:.2%}",     f"SPY: {m_b['mdd']:.2%}")
    c4.metric("勝率",             f"{m_s['winrate']:.1%}", f"SPY: {m_b['winrate']:.1%}")

    d1, d2 = st.columns(2)
    d1.metric("SEGM 總累積報酬", f"{m_s['total']*100:.2f}%",
              f"{(m_s['total'] - m_b['total'])*100:.2f}% vs SPY")
    d2.metric("SPY 總累積報酬",  f"{m_b['total']*100:.2f}%")

    # ── 最近 30 天 ──
    st.subheader("🔄 最近 30 天輪動紀錄")
    recent = res[["pk1","pk2","lev","ret"]].tail(30).copy()
    recent["ret"] = recent["ret"].map(lambda x: f"{x*100:+.2f}%")
    recent["lev"] = recent["lev"].map(lambda x: f"{x:.2f}x")
    recent.index  = recent.index.strftime("%Y-%m-%d")
    recent.columns = ["第一名 (60%)", "第二名 (40%)", "凱利槓桿", "當日報酬"]
    st.dataframe(recent[::-1], use_container_width=True)

except Exception as e:
    import traceback
    st.error(f"⚠️ 執行錯誤：{e}")
    with st.expander("詳細錯誤訊息"):
        st.code(traceback.format_exc())
