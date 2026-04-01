import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SEGM 投資儀表板 v3", layout="wide")
st.title("🚀 SEGM 投資儀表板 v3（Ensemble + 止損 + 黑天鵝防護）")

# ==================== 標的設定 ====================
TICKER_MAP = {
    # 多頭輪動池
    "QQQ":  "QQQ",
    "SPY":  "SPY",
    "TQQQ": "TQQQ",
    "BTC":  "BTC-USD",
    "IWM":  "IWM",
    "EEM":  "EEM",
    "VNQ":  "VNQ",
    # 避險 / 原物料
    "TLT":  "TLT",
    "SHY":  "SHY",
    "GLD":  "GLD",
    "GDX":  "GDX",
    "USO":  "USO",
    "UNG":  "UNG",
    # 空頭（不進輪動，只在黑天鵝時備用）
    "SQQQ": "SQQQ",
    "SH":   "SH",
    # 指標
    "VIX":  "^VIX",
}

# 參與動能輪動的標的（排除空頭/VIX）
TRADABLE = ["QQQ", "SPY", "TQQQ", "BTC", "IWM", "EEM", "VNQ",
            "TLT", "SHY", "GLD", "GDX", "USO", "UNG"]

# Ensemble 多週期動能（財務學有根據的週期）
MOMENTUM_PERIODS = [20, 40, 60, 120]  # 1個月、2個月、3個月、6個月

# ==================== 側邊欄 ====================
st.sidebar.header("⚙️ 策略設定")

st.sidebar.markdown("**📐 Ensemble 動能週期**")
st.sidebar.caption("同時計算以下週期的排名，取平均 → 無過擬合")
for p in MOMENTUM_PERIODS:
    st.sidebar.markdown(f"　・{p} 天（{'1個月' if p==20 else '2個月' if p==40 else '3個月' if p==60 else '6個月'}）")

btc_ma_period    = st.sidebar.number_input("BTC 均線週期（天）", 10, 200, 50)
cash_annual_rate = st.sidebar.number_input("現金年化利率（%）",  0.0, 10.0, 4.5)
cash_daily_rate  = (1 + cash_annual_rate / 100) ** (1/252) - 1

st.sidebar.markdown("---")
st.sidebar.markdown("**⚡ 槓桿設定**")
lev_normal = st.sidebar.slider("正常槓桿 VIX < 20",  1.0, 3.0, 2.0, 0.1)
lev_mild   = st.sidebar.slider("輕度警戒 VIX 20~30", 0.5, 2.0, 0.8, 0.1)
lev_high   = st.sidebar.slider("高度警戒 VIX 30~40", 0.2, 1.5, 0.6, 0.1)
lev_panic  = st.sidebar.slider("極度恐慌 VIX > 40",  0.1, 1.0, 0.4, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**🛡️ 止損設定**")
sl_half  = st.sidebar.slider("半倉止損線（從高點回撤 %）", 5,  25, 12, 1)
sl_full  = st.sidebar.slider("全倉止損線（從高點回撤 %）", 10, 40, 20, 1)
sl_resume_vix = st.sidebar.slider("止損後重新進場 VIX 閾值", 15, 35, 25, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("**🦢 黑天鵝防護**")
bs_vix_spike  = st.sidebar.slider("VIX 單日暴漲觸發（%）",       20, 60, 30, 5)
bs_tqqq_drop  = st.sidebar.slider("TQQQ 單日暴跌觸發（%）",      -30, -5, -15, 1)
bs_lev_after  = st.sidebar.slider("黑天鵝觸發後槓桿",             0.1, 1.0, 0.5, 0.1)
bs_cooldown   = st.sidebar.slider("黑天鵝後冷卻天數（強制低槓）", 1, 10, 3, 1)

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
            failed.append(f"{name}({str(e)[:30]})")

    if failed:
        st.sidebar.warning(f"下載失敗：{', '.join(failed)}")

    df = pd.DataFrame(series_dict).ffill().bfill()
    return df


def get_leverage(vix_val: float) -> float:
    if vix_val > 40:   return lev_panic
    elif vix_val > 30: return lev_high
    elif vix_val > 20: return lev_mild
    else:              return lev_normal


def calc_metrics(ret: pd.Series, rf_daily: float = 0.0) -> dict:
    r = ret.dropna()
    if len(r) < 2:
        return {k: 0 for k in ["total","cagr","vol","sharpe","mdd","winrate","calmar"]}
    n     = len(r)
    total = (1 + r).prod() - 1
    cagr  = (1 + total) ** (252/n) - 1
    vol   = r.std() * np.sqrt(252)
    sharpe  = (cagr - rf_daily*252) / vol if vol > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = (cum / cum.cummax() - 1).min()
    calmar  = cagr / abs(mdd) if mdd != 0 else 0
    winrate = (r > 0).mean()
    return {"total": total, "cagr": cagr, "vol": vol,
            "sharpe": sharpe, "mdd": mdd, "calmar": calmar, "winrate": winrate}


# ==================== 主程式 ====================
try:
    with st.spinner("📡 載入市場資料中..."):
        df_all = load_data(TICKER_MAP)

    missing = [c for c in ["VIX", "BTC", "QQQ", "SPY"] if c not in df_all.columns]
    if missing:
        st.error(f"❌ 缺少關鍵欄位：{missing}")
        st.stop()

    # ── 技術指標（在完整歷史上算）──
    df_all["BTC_MA"]   = df_all["BTC"].rolling(btc_ma_period).mean()
    df_all["VIX_1D_CHG"] = df_all["VIX"].pct_change()  # VIX 單日變動率（黑天鵝偵測）

    # ── Ensemble 動能：每個週期算排名，取平均排名 ──
    rank_dfs = []
    for period in MOMENTUM_PERIODS:
        mom  = df_all[TRADABLE].pct_change(period)
        # 每天對所有標的做排名（1=最高動能）
        rank = mom.rank(axis=1, ascending=False)
        rank_dfs.append(rank)

    # 平均排名（數字越小越好）
    ensemble_rank = pd.concat(rank_dfs).groupby(level=0).mean()

    # 切片
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
    lev_now    = get_leverage(vix_now)

    rank_today = ensemble_rank.loc[today].dropna().sort_values()  # 小=好
    avail = [a for a in rank_today.index if a in df_all.columns]
    rank_today = rank_today[avail]

    p1 = rank_today.index[0] if len(rank_today) > 0 else "CASH"
    p2 = rank_today.index[1] if len(rank_today) > 1 else "CASH"

    # 若所有週期動能都是負的才轉現金（用最短週期20天判斷）
    mom20 = df_all[TRADABLE].pct_change(20)
    if p1 in mom20.columns and mom20.loc[today, p1] < 0: p1 = "CASH"
    if p2 in mom20.columns and mom20.loc[today, p2] < 0: p2 = "CASH"

    # BTC 均線過濾
    if btc_now < btc_ma_now:
        qqq_m = mom20.loc[today, "QQQ"] if "QQQ" in mom20.columns else -1
        if p1 == "BTC": p1 = "QQQ" if qqq_m >= 0 else "CASH"
        if p2 == "BTC": p2 = "QQQ" if qqq_m >= 0 else "CASH"

    # 黑天鵝今日檢查
    vix_spike_today  = bt.loc[today, "VIX_1D_CHG"] * 100 if "VIX_1D_CHG" in bt.columns else 0
    tqqq_drop_today  = df_all["TQQQ"].pct_change().loc[today] * 100 if "TQQQ" in df_all.columns else 0
    bs_triggered     = (vix_spike_today > bs_vix_spike) or (tqqq_drop_today < bs_tqqq_drop)
    if bs_triggered:
        lev_now = bs_lev_after

    # ── 版面：今日訊號 ──
    tab1, tab2 = st.tabs(["📊 今日訊號", "📈 回測分析"])

    with tab1:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("策略訊號")
            st.write(f"📅 {today.strftime('%Y-%m-%d')}")
            st.write(f"🥇 **{p1}**（60%）" + (f" 　平均排名 {rank_today.get(p1,'-'):.1f}" if p1 != "CASH" else " 　現金防禦"))
            st.write(f"🥈 **{p2}**（40%）" + (f" 　平均排名 {rank_today.get(p2,'-'):.1f}" if p2 != "CASH" else " 　現金防禦"))
            lev_color = "🟢" if lev_now >= 1.5 else "🟡" if lev_now >= 0.8 else "🔴"
            st.markdown(f"### {lev_color} 建議槓桿：{lev_now}x")
            if bs_triggered:
                st.error(f"🦢 黑天鵝觸發！VIX暴漲{vix_spike_today:.1f}% / TQQQ單日{tqqq_drop_today:.1f}%")

        with c2:
            st.subheader("風控狀態")
            vix_col = "🔴" if vix_now > 30 else "🟡" if vix_now > 20 else "🟢"
            st.write(f"VIX：{vix_col} **{vix_now:.2f}**（單日變動 {vix_spike_today:+.1f}%）")
            st.write(f"BTC：**{btc_now:,.0f}** / {btc_ma_period}日均線：**{btc_ma_now:,.0f}**")
            btc_status = "均線上方 ✅" if btc_now >= btc_ma_now else "均線下方 ⚠️"
            st.write(f"BTC趨勢：{btc_status}")
            if vix_now > 30:   st.error("🚨 市場極度恐慌")
            elif vix_now > 20: st.warning("⚠️ 市場波動加劇")
            else:              st.success("✅ 市場狀態正常")

        with c3:
            st.subheader("Ensemble 動能排名")
            show_assets = [a for a in rank_today.index[:8] if a in df_all.columns]
            rank_display = []
            for i, asset in enumerate(show_assets):
                # 用20天動能判斷正負
                m20_val = mom20.loc[today, asset] if asset in mom20.columns else 0
                rank_display.append({
                    "排名": i+1,
                    "標的": asset,
                    "平均排名分": f"{rank_today[asset]:.1f}",
                    "20日動能": f"{m20_val:.2%}",
                    "狀態": "✅" if m20_val > 0 else "❌"
                })
            st.dataframe(pd.DataFrame(rank_display), use_container_width=True, hide_index=True)

    # ==================== 回測引擎 ====================
    with tab2:
        st.subheader(f"策略回測（{start_date} ～ {end_date}）")
        st.caption("Ensemble動能 + 動態止損 + 黑天鵝防護 vs SPY")

        asset_ret = df_all[TRADABLE].pct_change()
        spy_ret   = df_all["SPY"].pct_change()

        rows = []
        peak_value   = 1.0   # 追蹤淨值高點（止損用）
        cum_value    = 1.0
        sl_state     = "normal"   # normal / half / full_cash
        bs_cooldown_left = 0      # 黑天鵝冷卻剩餘天數

        dates = bt.index[:-1]

        for date in dates:
            if date not in ensemble_rank.index:
                continue

            loc = df_all.index.get_loc(date)
            if loc + 1 >= len(df_all):
                continue
            next_day = df_all.index[loc + 1]

            # ── 黑天鵝偵測 ──
            vix_chg  = df_all.loc[date, "VIX_1D_CHG"] if "VIX_1D_CHG" in df_all.columns else 0
            tqqq_chg = asset_ret.loc[date, "TQQQ"] if "TQQQ" in asset_ret.columns and date in asset_ret.index else 0
            tqqq_chg = tqqq_chg if not pd.isna(tqqq_chg) else 0

            bs_hit = (vix_chg * 100 > bs_vix_spike) or (tqqq_chg * 100 < bs_tqqq_drop)
            if bs_hit:
                bs_cooldown_left = bs_cooldown

            # ── 止損狀態機 ──
            drawdown_now = cum_value / peak_value - 1

            if sl_state == "full_cash":
                # 全倉現金，等VIX降下來才恢復
                vix_t = bt.loc[date, "VIX"] if date in bt.index else 99
                if vix_t < sl_resume_vix:
                    sl_state = "normal"
            elif drawdown_now < -(sl_full / 100):
                sl_state = "full_cash"
            elif drawdown_now < -(sl_half / 100):
                sl_state = "half"
            else:
                sl_state = "normal"

            # ── 選股訊號 ──
            rank_t = ensemble_rank.loc[date].dropna().sort_values()
            avail_t = [a for a in rank_t.index if a in asset_ret.columns]
            rank_t  = rank_t[avail_t]

            pk1 = rank_t.index[0] if len(rank_t) > 0 else "CASH"
            pk2 = rank_t.index[1] if len(rank_t) > 1 else "CASH"

            if date in mom20.index:
                if pk1 in mom20.columns and mom20.loc[date, pk1] < 0: pk1 = "CASH"
                if pk2 in mom20.columns and mom20.loc[date, pk2] < 0: pk2 = "CASH"

            btc_t    = bt.loc[date, "BTC"]    if date in bt.index else 0
            btc_ma_t = bt.loc[date, "BTC_MA"] if date in bt.index else 0
            if btc_t < btc_ma_t:
                qqq_m_t = mom20.loc[date, "QQQ"] if date in mom20.index and "QQQ" in mom20.columns else -1
                if pk1 == "BTC": pk1 = "QQQ" if qqq_m_t >= 0 else "CASH"
                if pk2 == "BTC": pk2 = "QQQ" if qqq_m_t >= 0 else "CASH"

            # ── 止損覆蓋選股結果 ──
            if sl_state == "full_cash":
                pk1 = pk2 = "CASH"
            elif sl_state == "half":
                # 60%持倉 → 30%，另外30%現金；40%持倉 → 20%，另外20%現金
                # 用權重調整實現（等比縮減持倉）
                pass  # 在報酬計算時處理

            # ── 槓桿 ──
            vix_t  = bt.loc[date, "VIX"] if date in bt.index else 20
            lev_bt = get_leverage(vix_t)
            if bs_cooldown_left > 0:
                lev_bt = min(lev_bt, bs_lev_after)
                bs_cooldown_left -= 1

            # 半倉止損：槓桿再砍半
            if sl_state == "half":
                lev_bt = lev_bt * 0.5

            # ── 次日報酬 ──
            def get_ret(pick):
                if pick == "CASH":
                    return cash_daily_rate
                if pick not in asset_ret.columns or next_day not in asset_ret.index:
                    return 0.0
                v = asset_ret.loc[next_day, pick]
                return 0.0 if pd.isna(v) else float(v)

            r1 = get_ret(pk1)
            r2 = get_ret(pk2)

            risky = (0.6 * r1 if pk1 != "CASH" else 0) + (0.4 * r2 if pk2 != "CASH" else 0)
            cash  = (0.6 * cash_daily_rate if pk1 == "CASH" else 0) + \
                    (0.4 * cash_daily_rate if pk2 == "CASH" else 0)
            day_ret = risky * lev_bt + cash

            # 更新淨值與高點
            cum_value *= (1 + day_ret)
            peak_value = max(peak_value, cum_value)

            rows.append({
                "Date":    date,
                "ret":     day_ret,
                "pk1":     pk1,
                "pk2":     pk2,
                "lev":     lev_bt,
                "sl":      sl_state,
                "bs_hit":  bs_hit,
            })

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

        # ── 績效摘要 ──
        m_s = calc_metrics(res["ret"],                               cash_daily_rate)
        m_b = calc_metrics(spy_ret.reindex(res.index).fillna(0),    cash_daily_rate)

        st.subheader("📊 績效摘要")
        cols = st.columns(5)
        metrics = [
            ("年化報酬 CAGR",   f"{m_s['cagr']:.2%}",    f"SPY {m_b['cagr']:.2%}"),
            ("夏普比率",        f"{m_s['sharpe']:.2f}",  f"SPY {m_b['sharpe']:.2f}"),
            ("最大回撤 MDD",    f"{m_s['mdd']:.2%}",     f"SPY {m_b['mdd']:.2%}"),
            ("Calmar 比率",     f"{m_s['calmar']:.2f}",  f"SPY {m_b['calmar']:.2f}"),
            ("勝率",            f"{m_s['winrate']:.1%}", f"SPY {m_b['winrate']:.1%}"),
        ]
        for col, (label, val, delta) in zip(cols, metrics):
            col.metric(label, val, delta)

        d1, d2, d3 = st.columns(3)
        d1.metric("SEGM 總累積報酬", f"{m_s['total']*100:.1f}%",
                  f"{(m_s['total']-m_b['total'])*100:.1f}% vs SPY")
        d2.metric("SPY 總累積報酬",  f"{m_b['total']*100:.1f}%")
        d3.metric("年化波動率",      f"{m_s['vol']:.2%}", f"SPY {m_b['vol']:.2%}")

        # ── 止損 & 黑天鵝觸發記錄 ──
        with st.expander("🛡️ 止損觸發紀錄"):
            sl_events = res[res["sl"] != "normal"][["pk1","pk2","lev","sl","ret"]]
            if sl_events.empty:
                st.success("回測期間未觸發止損 ✅")
            else:
                sl_events.index = sl_events.index.strftime("%Y-%m-%d")
                sl_events.columns = ["標的1","標的2","槓桿","止損狀態","當日報酬"]
                sl_events["當日報酬"] = sl_events["當日報酬"].map(lambda x: f"{x*100:+.2f}%")
                st.dataframe(sl_events, use_container_width=True)

        with st.expander("🦢 黑天鵝觸發紀錄"):
            bs_events = res[res["bs_hit"] == True][["pk1","pk2","lev","ret"]]
            if bs_events.empty:
                st.success("回測期間未觸發黑天鵝防護 ✅")
            else:
                bs_events.index = bs_events.index.strftime("%Y-%m-%d")
                bs_events.columns = ["標的1","標的2","緊急槓桿","當日報酬"]
                bs_events["當日報酬"] = bs_events["當日報酬"].map(lambda x: f"{x*100:+.2f}%")
                st.dataframe(bs_events, use_container_width=True)

        # ── 最近30天輪動 ──
        with st.expander("🔄 最近 30 天輪動紀錄"):
            recent = res[["pk1","pk2","lev","sl","ret"]].tail(30).copy()
            recent["ret"] = recent["ret"].map(lambda x: f"{x*100:+.2f}%")
            recent["lev"] = recent["lev"].map(lambda x: f"{x}x")
            recent.index  = recent.index.strftime("%Y-%m-%d")
            recent.columns = ["第一名(60%)", "第二名(40%)", "槓桿", "止損狀態", "當日報酬"]
            st.dataframe(recent[::-1], use_container_width=True)

except Exception as e:
    import traceback
    st.error(f"⚠️ 執行錯誤：{e}")
    with st.expander("詳細錯誤訊息"):
        st.code(traceback.format_exc())
