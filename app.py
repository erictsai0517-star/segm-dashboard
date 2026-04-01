import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SEGM 投資儀表板", layout="wide")
st.title("🚀 SEGM 投資儀表板（四版本對比）")

TICKER_MAP = {
    "QQQ": "QQQ", "BTC": "BTC-USD", "TQQQ": "TQQQ",
    "TLT": "TLT", "GLD": "GLD",     "USO":  "USO",
    "SPY": "SPY", "VIX": "^VIX",
}
TRADABLE = ["QQQ", "BTC", "TLT", "GLD", "USO"]

FEES = {"QQQ": 0.0005, "TQQQ": 0.0015, "BTC": 0.001,
        "TLT": 0.0005, "GLD":  0.0005,  "USO": 0.001, "CASH": 0.0}

# ── 四個預設版本 ──
PRESETS = {
    "🚀 激進（最大化終值）": {
        "momentum_period": 25, "btc_ma_period": 50,
        "kelly_fraction": 1.0, "kelly_max": 4.0, "kelly_min": 0.8,
        "kelly_window": 20,    "vix_base": 20,
        "bs_vix_spike": 30,    "bs_lev_cap": 0.3, "bs_cooldown": 2,
        "desc": "全凱利，高槓桿，只有黑天鵝才降槓。終值最大，MDD 最高。"
    },
    "⚖️ 平衡（收益與回撤兼顧）": {
        "momentum_period": 25, "btc_ma_period": 50,
        "kelly_fraction": 0.75, "kelly_max": 3.0, "kelly_min": 0.5,
        "kelly_window": 40,    "vix_base": 20,
        "bs_vix_spike": 25,    "bs_lev_cap": 0.3, "bs_cooldown": 3,
        "desc": "0.75凱利，中槓桿，黑天鵝觸發更靈敏。收益與回撤的甜蜜點。"
    },
    "🛡️ 保守（MDD 優先）": {
        "momentum_period": 25, "btc_ma_period": 50,
        "kelly_fraction": 0.5, "kelly_max": 2.0, "kelly_min": 0.3,
        "kelly_window": 60,    "vix_base": 18,
        "bs_vix_spike": 20,    "bs_lev_cap": 0.2, "bs_cooldown": 5,
        "desc": "半凱利，低槓桿，VIX更早介入。MDD 最低，終值犧牲最多。"
    },
    "🎯 自訂": {
        "momentum_period": 25, "btc_ma_period": 50,
        "kelly_fraction": 0.95, "kelly_max": 4.0, "kelly_min": 0.8,
        "kelly_window": 20,    "vix_base": 20,
        "bs_vix_spike": 30,    "bs_lev_cap": 0.3, "bs_cooldown": 2,
        "desc": "自行調整所有參數。"
    },
}

# ==================== 側邊欄 ====================
st.sidebar.header("⚙️ 策略設定")

preset_name = st.sidebar.selectbox("版本選擇", list(PRESETS.keys()))
p = PRESETS[preset_name]
st.sidebar.caption(p["desc"])
st.sidebar.markdown("---")

is_custom = (preset_name == "🎯 自訂")

momentum_period  = st.sidebar.number_input("動能週期（天）",     10, 200, p["momentum_period"], disabled=not is_custom)
btc_ma_period    = st.sidebar.number_input("BTC 均線週期（天）", 10, 200, p["btc_ma_period"],    disabled=not is_custom)
cash_annual_rate = st.sidebar.number_input("現金年化利率（%）",  0.0, 10.0, 2.0)
cash_daily_rate  = (1 + cash_annual_rate / 100) ** (1/252) - 1

st.sidebar.markdown("**⚡ 凱利槓桿**")
kelly_fraction = st.sidebar.slider("凱利分數",   0.1, 1.0, float(p["kelly_fraction"]), 0.05, disabled=not is_custom)
kelly_max      = st.sidebar.slider("槓桿上限",   1.0, 5.0, float(p["kelly_max"]),      0.1,  disabled=not is_custom)
kelly_min      = st.sidebar.slider("槓桿下限",   0.1, 1.5, float(p["kelly_min"]),      0.1,  disabled=not is_custom)
kelly_window   = st.sidebar.number_input("凱利回望窗口（天）", 10, 120, p["kelly_window"], disabled=not is_custom)

st.sidebar.markdown("**📉 VIX 動態上限**")
vix_base = st.sidebar.slider("基準 VIX", 10, 30, p["vix_base"], 1, disabled=not is_custom)
st.sidebar.caption(f"動態上限 = 槓桿上限 × ({vix_base} / 當日VIX)")

st.sidebar.markdown("**🦢 黑天鵝防護**")
bs_vix_spike = st.sidebar.slider("VIX 單日暴漲觸發（%）", 15, 60, p["bs_vix_spike"], 5,   disabled=not is_custom)
bs_lev_cap   = st.sidebar.slider("觸發後槓桿上限",        0.1, 1.0, float(p["bs_lev_cap"]), 0.1, disabled=not is_custom)
bs_cooldown  = st.sidebar.slider("冷卻天數",               1,   5,  p["bs_cooldown"],  1,   disabled=not is_custom)

st.sidebar.markdown("---")
start_date = st.sidebar.date_input("回測起始日期", datetime.date(2018, 1, 1))
end_date   = st.sidebar.date_input("回測結束日期", datetime.date.today())

if start_date >= end_date:
    st.sidebar.error("起始日期必須早於結束日期")
    st.stop()

# ==================== 函數 ====================
@st.cache_data(ttl=3600)
def load_data(ticker_map):
    series_dict = {}
    failed = []
    for name, ticker in ticker_map.items():
        try:
            raw = yf.download(ticker, period="max", progress=False, auto_adjust=True)
            if raw.empty:
                failed.append(name); continue
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


def kelly_leverage(ret_series, fraction, lev_min, lev_max, vix_val=20.0, vix_base=20.0):
    r = ret_series.dropna()
    if len(r) < 5: return fraction
    mu = r.mean(); sigma2 = r.var()
    if sigma2 <= 0 or mu <= 0: return lev_min
    dynamic_max = max(lev_max * (vix_base / max(vix_val, 1.0)), lev_min)
    return float(np.clip((mu / sigma2) * fraction, lev_min, dynamic_max))


def calc_metrics(ret, rf_daily=0.0):
    r = ret.dropna()
    if len(r) < 2:
        return {k: 0 for k in ["total","cagr","vol","sharpe","mdd","winrate"]}
    n = len(r)
    total   = (1 + r).prod() - 1
    cagr    = (1 + total) ** (252/n) - 1
    vol     = r.std() * np.sqrt(252)
    sharpe  = (cagr - rf_daily*252) / vol if vol > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = (cum / cum.cummax() - 1).min()
    winrate = (r > 0).mean()
    return {"total": total, "cagr": cagr, "vol": vol,
            "sharpe": sharpe, "mdd": mdd, "winrate": winrate}


def run_backtest(df_all, bt, momentum, momentum_period,
                 kelly_fraction, kelly_max, kelly_min, kelly_window,
                 vix_base, bs_vix_spike, bs_lev_cap, bs_cooldown,
                 cash_daily_rate):
    """回測引擎，回傳 rows list"""
    asset_ret     = df_all[TRADABLE].pct_change()
    vix_daily_chg = df_all["VIX"].pct_change()
    spy_ma60      = df_all["SPY"].rolling(60).mean()
    strat_ret_ts  = pd.Series(dtype=float)

    rows       = []
    prev_pk1   = None
    prev_pk2   = None
    bs_cd_left = 0

    for date in bt.index[:-1]:
        if date not in momentum.index: continue
        loc = df_all.index.get_loc(date)
        if loc + 1 >= len(df_all): continue
        next_day = df_all.index[loc + 1]

        mom_t = momentum.loc[date, TRADABLE].dropna().sort_values(ascending=False)
        if len(mom_t) < 2: continue

        pk1 = mom_t.index[0]; pk2 = mom_t.index[1]
        m1_t = mom_t.iloc[0]; m2_t = mom_t.iloc[1]
        if m1_t < 0: pk1 = "CASH"
        if m2_t < 0: pk2 = "CASH"

        btc_t    = bt.loc[date, "BTC"]
        btc_ma_t = bt.loc[date, "BTC_MA"]
        if btc_t < btc_ma_t:
            qqq_m = momentum.loc[date, "QQQ"] if "QQQ" in momentum.columns else -1
            if pk1 == "BTC": pk1 = "QQQ" if qqq_m >= 0 else "CASH"
            if pk2 == "BTC": pk2 = "QQQ" if qqq_m >= 0 else "CASH"

        # 動態權重
        if pk1 != "CASH" and pk2 != "CASH":
            w1 = min(0.8, 0.6 + abs(m1_t - m2_t) * 2)
        else:
            w1 = 0.6
        w2 = 1 - w1

        # 市場狀態
        vix_t        = bt.loc[date, "VIX"]
        qqq_mom_t    = momentum.loc[date, "QQQ"] if "QQQ" in momentum.columns else 0
        spy_above_ma = df_all.loc[date, "SPY"] > spy_ma60.loc[date] if date in spy_ma60.index else True
        is_bull = (qqq_mom_t > 0) and spy_above_ma
        is_bear = (qqq_mom_t < 0) and (vix_t > vix_base)

        if is_bull:
            dyn_max, dyn_min = kelly_max, kelly_min
        elif is_bear:
            dyn_max, dyn_min = kelly_max * 0.5, kelly_min * 0.5
        else:
            dyn_max, dyn_min = kelly_max * 0.75, kelly_min

        # 凱利槓桿
        if len(strat_ret_ts) >= kelly_window:
            wr = strat_ret_ts.iloc[-kelly_window:]
        elif len(strat_ret_ts) >= 5:
            wr = strat_ret_ts
        else:
            proxy = asset_ret[pk1 if pk1 != "CASH" else "QQQ"]
            wr    = proxy.loc[:date].iloc[-kelly_window:]

        lev_t = kelly_leverage(wr, kelly_fraction, dyn_min, dyn_max,
                               vix_val=vix_t, vix_base=vix_base)

        # 黑天鵝
        vix_chg_t = vix_daily_chg.loc[date] if date in vix_daily_chg.index else 0
        bs_hit    = (not pd.isna(vix_chg_t)) and (vix_chg_t * 100 > bs_vix_spike)
        if bs_hit: bs_cd_left = bs_cooldown
        if bs_cd_left > 0:
            lev_t = min(lev_t, bs_lev_cap)
            bs_cd_left -= 1

        # 手續費
        fee = 0.0
        if prev_pk1 is not None:
            if pk1 != prev_pk1:
                if prev_pk1 != "CASH": fee += FEES.get(prev_pk1, 0.001) * w1
                if pk1      != "CASH": fee += FEES.get(pk1,      0.001) * w1
            if pk2 != prev_pk2:
                if prev_pk2 != "CASH": fee += FEES.get(prev_pk2, 0.001) * w2
                if pk2      != "CASH": fee += FEES.get(pk2,      0.001) * w2
        prev_pk1 = pk1; prev_pk2 = pk2

        # 次日報酬
        def next_ret(pick):
            if pick == "CASH": return cash_daily_rate
            if pick not in asset_ret.columns or next_day not in asset_ret.index: return 0.0
            v = asset_ret.loc[next_day, pick]
            return 0.0 if pd.isna(v) else float(v)

        r1 = next_ret(pk1); r2 = next_ret(pk2)
        risky  = (w1*r1 if pk1 != "CASH" else 0) + (w2*r2 if pk2 != "CASH" else 0)
        cash_r = (w1*cash_daily_rate if pk1 == "CASH" else 0) + \
                 (w2*cash_daily_rate if pk2 == "CASH" else 0)
        day_ret = risky * lev_t + cash_r - fee

        strat_ret_ts = pd.concat([strat_ret_ts, pd.Series([day_ret], index=[date])])
        regime = ("🐂" if is_bull else "🐻" if is_bear else "↔") + ("🦢" if bs_hit else "")
        rows.append({"Date": date, "ret": day_ret, "pk1": pk1, "pk2": pk2,
                     "w1": w1, "lev": lev_t, "regime": regime, "fee": fee})
    return rows


# ==================== 主程式 ====================
try:
    with st.spinner("📡 載入市場資料中..."):
        df_all = load_data(TICKER_MAP)

    missing = [c for c in ["VIX","BTC","QQQ","SPY"] if c not in df_all.columns]
    if missing:
        st.error(f"❌ 缺少關鍵欄位：{missing}"); st.stop()

    df_all["BTC_MA"] = df_all["BTC"].rolling(btc_ma_period).mean()
    momentum = df_all[TRADABLE].pct_change(momentum_period)
    spy_ret  = df_all["SPY"].pct_change()

    bt = df_all.loc[str(start_date): str(end_date)].copy()
    bt = bt.dropna(subset=["VIX","BTC","BTC_MA"])
    if len(bt) < 5:
        st.error(f"❌ 資料不足（{len(bt)}天）"); st.stop()

    # ── 今日訊號 ──
    today      = bt.index[-1]
    vix_now    = bt.loc[today, "VIX"]
    btc_now    = bt.loc[today, "BTC"]
    btc_ma_now = bt.loc[today, "BTC_MA"]

    mom_today = momentum.loc[today, TRADABLE].dropna().sort_values(ascending=False)
    p1 = mom_today.index[0] if len(mom_today) > 0 else "CASH"
    p2 = mom_today.index[1] if len(mom_today) > 1 else "CASH"
    m1_val = mom_today.iloc[0] if len(mom_today) > 0 else -1.0
    m2_val = mom_today.iloc[1] if len(mom_today) > 1 else -1.0
    if m1_val < 0: p1 = "CASH"
    if m2_val < 0: p2 = "CASH"
    if btc_now < btc_ma_now:
        qqq_m = momentum.loc[today, "QQQ"] if "QQQ" in momentum.columns else -1
        if p1 == "BTC": p1 = "QQQ" if qqq_m >= 0 else "CASH"
        if p2 == "BTC": p2 = "QQQ" if qqq_m >= 0 else "CASH"

    w1_now = min(0.8, 0.6 + abs(m1_val - m2_val) * 2) if p1 != "CASH" and p2 != "CASH" else 0.6
    w2_now = 1 - w1_now
    lev_now = kelly_leverage(df_all["QQQ"].pct_change().iloc[-kelly_window:],
                             kelly_fraction, kelly_min, kelly_max,
                             vix_val=vix_now, vix_base=vix_base)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 今日策略訊號")
        st.write(f"📅 {today.strftime('%Y-%m-%d')}　版本：**{preset_name}**")
        st.write(f"🥇 **{p1}** ({w1_now:.0%})　動能 {m1_val:.2%}" if p1 != "CASH" else f"🥇 **現金** ({w1_now:.0%})")
        st.write(f"🥈 **{p2}** ({w2_now:.0%})　動能 {m2_val:.2%}" if p2 != "CASH" else f"🥈 **現金** ({w2_now:.0%})")
        st.markdown(f"### 🔥 凱利槓桿：{lev_now:.2f}x")
    with col2:
        st.subheader("⚠️ 風控狀態")
        st.write(f"VIX：**{vix_now:.2f}**　BTC：**{btc_now:,.0f}** / 均線 **{btc_ma_now:,.0f}**")
        vix_chg_now = df_all["VIX"].pct_change().iloc[-1] * 100
        dyn_max_now = kelly_max * (vix_base / max(vix_now, 1.0))
        if vix_chg_now > bs_vix_spike:
            st.error(f"🦢 黑天鵝！VIX 單日 +{vix_chg_now:.1f}%，明日上限 {bs_lev_cap}x")
        elif vix_now > vix_base:
            st.warning(f"⚠️ VIX {vix_now:.1f} > 基準 {vix_base}，動態上限 {dyn_max_now:.2f}x")
        else:
            st.success("✅ 市場狀態正常")

    st.markdown("**📋 當日動能排名**")
    st.dataframe(pd.DataFrame({
        "標的": mom_today.index,
        "動能": [f"{v:.2%}" for v in mom_today.values],
        "狀態": ["✅" if v > 0 else "❌" for v in mom_today.values],
    }), use_container_width=True, hide_index=True)

    # ── 回測（當前版本） ──
    st.markdown("---")
    st.subheader(f"📈 回測（{start_date} ～ {end_date}）")

    rows = run_backtest(df_all, bt, momentum, momentum_period,
                        kelly_fraction, kelly_max, kelly_min, kelly_window,
                        vix_base, bs_vix_spike, bs_lev_cap, bs_cooldown,
                        cash_daily_rate)

    if not rows:
        st.error("❌ 無有效資料"); st.stop()

    res = pd.DataFrame(rows).set_index("Date")
    res["cum_strat"] = (1 + res["ret"]).cumprod() - 1
    res["cum_spy"]   = (1 + spy_ret.reindex(res.index).fillna(0)).cumprod() - 1

    st.line_chart(
        (res[["cum_strat","cum_spy"]] * 100).rename(
            columns={"cum_strat": f"{preset_name}", "cum_spy": "SPY基準"}),
        use_container_width=True
    )

    trade_days    = (res["fee"] > 0).sum()
    total_fee_pct = res["fee"].sum() * 100
    st.caption(f"💸 總手續費：**{total_fee_pct:.2f}%**　換倉：**{trade_days}** 天")

    # ── 四版本對比表 ──
    st.markdown("---")
    st.subheader("📊 四版本績效對比")
    compare_rows = []
    for name, cfg in PRESETS.items():
        if name == "🎯 自訂": continue
        r_list = run_backtest(
            df_all, bt, momentum, cfg["momentum_period"],
            cfg["kelly_fraction"], cfg["kelly_max"], cfg["kelly_min"], cfg["kelly_window"],
            cfg["vix_base"], cfg["bs_vix_spike"], cfg["bs_lev_cap"], cfg["bs_cooldown"],
            cash_daily_rate
        )
        if not r_list: continue
        r_df = pd.DataFrame(r_list).set_index("Date")
        m    = calc_metrics(r_df["ret"], cash_daily_rate)
        compare_rows.append({
            "版本": name,
            "總累積報酬": f"{m['total']*100:.1f}%",
            "年化報酬":   f"{m['cagr']*100:.1f}%",
            "最大回撤":   f"{m['mdd']*100:.1f}%",
            "夏普比率":   f"{m['sharpe']:.2f}",
            "勝率":       f"{m['winrate']*100:.1f}%",
        })
    # 加上自訂版本當前參數結果
    m_cur = calc_metrics(res["ret"], cash_daily_rate)
    m_spy = calc_metrics(spy_ret.reindex(res.index).fillna(0), cash_daily_rate)
    compare_rows.append({
        "版本": f"🎯 自訂（{preset_name}）" if preset_name == "🎯 自訂" else f"▶ {preset_name}（當前）",
        "總累積報酬": f"{m_cur['total']*100:.1f}%",
        "年化報酬":   f"{m_cur['cagr']*100:.1f}%",
        "最大回撤":   f"{m_cur['mdd']*100:.1f}%",
        "夏普比率":   f"{m_cur['sharpe']:.2f}",
        "勝率":       f"{m_cur['winrate']*100:.1f}%",
    })
    compare_rows.append({
        "版本": "📌 SPY 基準",
        "總累積報酬": f"{m_spy['total']*100:.1f}%",
        "年化報酬":   f"{m_spy['cagr']*100:.1f}%",
        "最大回撤":   f"{m_spy['mdd']*100:.1f}%",
        "夏普比率":   f"{m_spy['sharpe']:.2f}",
        "勝率":       f"{m_spy['winrate']*100:.1f}%",
    })
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

    # ── 詳細績效 ──
    st.subheader(f"📋 {preset_name} 詳細績效")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("年化報酬 CAGR", f"{m_cur['cagr']:.2%}",    f"SPY: {m_spy['cagr']:.2%}")
    c2.metric("夏普比率",       f"{m_cur['sharpe']:.2f}",  f"SPY: {m_spy['sharpe']:.2f}")
    c3.metric("最大回撤 MDD",  f"{m_cur['mdd']:.2%}",     f"SPY: {m_spy['mdd']:.2%}")
    c4.metric("勝率",           f"{m_cur['winrate']:.1%}", f"SPY: {m_spy['winrate']:.1%}")
    d1, d2 = st.columns(2)
    d1.metric("SEGM 總累積報酬", f"{m_cur['total']*100:.2f}%",
              f"{(m_cur['total']-m_spy['total'])*100:.2f}% vs SPY")
    d2.metric("SPY 總累積報酬",  f"{m_spy['total']*100:.2f}%")

    with st.expander("📊 凱利槓桿歷史"):
        st.line_chart(res["lev"].rename("槓桿"), use_container_width=True)
        st.caption(f"平均 {res['lev'].mean():.2f}x　最高 {res['lev'].max():.2f}x　最低 {res['lev'].min():.2f}x")

    st.subheader("🔄 最近 30 天")
    recent = res[["pk1","pk2","w1","lev","regime","fee","ret"]].tail(30).copy()
    recent["ret"] = recent["ret"].map(lambda x: f"{x*100:+.2f}%")
    recent["lev"] = recent["lev"].map(lambda x: f"{x:.2f}x")
    recent["w1"]  = recent["w1"].map(lambda x: f"{x:.0%}/{1-x:.0%}")
    recent["fee"] = recent["fee"].map(lambda x: f"{x*100:.3f}%" if x > 0 else "-")
    recent.index  = recent.index.strftime("%Y-%m-%d")
    recent.columns = ["第一名","第二名","權重","槓桿","狀態","手續費","當日報酬"]
    st.dataframe(recent[::-1], use_container_width=True)

except Exception as e:
    import traceback
    st.error(f"⚠️ 執行錯誤：{e}")
    with st.expander("詳細錯誤"):
        st.code(traceback.format_exc())
