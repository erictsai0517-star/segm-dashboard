import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SEGM Pro 儀表板", layout="wide")
st.title("🚀 SEGM Pro｜散戶極致版 v2")

# ══════════════════════════════════════════
# 常數設定（甜蜜點，不開放調整）
# ══════════════════════════════════════════
TICKER_MAP = {
    "QQQ": "QQQ", "BTC": "BTC-USD",
    "TLT": "TLT", "GLD": "GLD", "USO": "USO",
    "SHY": "SHY",   # 短期美債，震盪時停泊
    "SPY": "SPY", "VIX": "^VIX",
}
TRADABLE    = ["QQQ", "BTC", "TLT", "GLD", "USO"]
SAFE_ASSET  = "SHY"   # 震盪退場時持有這個

FEES = {
    "QQQ": 0.0005, "BTC": 0.001,
    "TLT": 0.0005, "GLD": 0.0005,
    "USO": 0.001,  "SHY": 0.0002, "CASH": 0.0,
}

SHARPE_MOM_WINDOW    = 60    # Sharpe動能回望（天）
CORR_WINDOW          = 60    # 相關性計算回望（天）
CORR_THRESHOLD       = 0.3   # 相關係數上限（降至0.3）
ADX_WINDOW           = 14    # ADX 標準窗口
ADX_THRESHOLD        = 20    # ADX < 20 = 震盪，退場
KELLY_MAX_DAILY_CHG  = 0.3   # 凱利每日最大變動
VOL_TARGET           = 0.20  # 波動率目標（年化20%）
MARGIN_ANNUAL_RATE   = 0.06  # 融資年利率

# ══════════════════════════════════════════
# 三個預設版本
# ══════════════════════════════════════════
PRESETS = {
    "🚀 激進": {
        "kelly_fraction": 0.9, "kelly_max": 3.5, "kelly_min": 0.8,
        "kelly_window": 30,    "vix_base": 20,
        "desc": "最大化終值，MDD 較高。長期持有且心臟夠強。"
    },
    "⚖️ 平衡": {
        "kelly_fraction": 0.65, "kelly_max": 2.5, "kelly_min": 0.5,
        "kelly_window": 40,     "vix_base": 20,
        "desc": "夏普比率最優，散戶極致推薦版。"
    },
    "🛡️ 保守": {
        "kelly_fraction": 0.4, "kelly_max": 1.8, "kelly_min": 0.3,
        "kelly_window": 60,    "vix_base": 18,
        "desc": "MDD 優先，適合資金量大或風險承受低的人。"
    },
    "🎯 自訂": {
        "kelly_fraction": 0.65, "kelly_max": 2.5, "kelly_min": 0.5,
        "kelly_window": 40,     "vix_base": 20,
        "desc": "自行調整所有參數。"
    },
}

# ══════════════════════════════════════════
# 側邊欄
# ══════════════════════════════════════════
st.sidebar.header("⚙️ 策略設定")
preset_name = st.sidebar.selectbox("版本選擇", list(PRESETS.keys()))
p = PRESETS[preset_name]
st.sidebar.caption(p["desc"])
st.sidebar.markdown("---")

is_custom = (preset_name == "🎯 自訂")

momentum_period  = st.sidebar.number_input("動能週期（天）",     10, 200, 25,   disabled=not is_custom)
btc_ma_period    = st.sidebar.number_input("BTC 均線週期（天）", 10, 200, 50,   disabled=not is_custom)
cash_annual_rate = st.sidebar.number_input("現金年化利率（%）",  0.0, 10.0, 2.0)
cash_daily_rate  = (1 + cash_annual_rate / 100) ** (1/252) - 1

st.sidebar.markdown("**⚡ 凱利槓桿**")
kelly_fraction = st.sidebar.slider("凱利分數",  0.1, 1.0, float(p["kelly_fraction"]), 0.05, disabled=not is_custom)
kelly_max      = st.sidebar.slider("槓桿上限",  1.0, 5.0, float(p["kelly_max"]),      0.1,  disabled=not is_custom)
kelly_min      = st.sidebar.slider("槓桿下限",  0.1, 1.5, float(p["kelly_min"]),      0.1,  disabled=not is_custom)
kelly_window   = st.sidebar.number_input("凱利回望窗口（天）", 10, 120, p["kelly_window"], disabled=not is_custom)

st.sidebar.markdown("**📉 VIX 動態上限**")
vix_base = st.sidebar.slider("基準 VIX", 10, 30, p["vix_base"], 1, disabled=not is_custom)
st.sidebar.caption(f"動態上限 = 槓桿上限 × ({vix_base} / 當日VIX)")

st.sidebar.markdown("---")
start_date = st.sidebar.date_input("回測起始日期", datetime.date(2018, 1, 1))
end_date   = st.sidebar.date_input("回測結束日期", datetime.date.today())
if start_date >= end_date:
    st.sidebar.error("起始日期必須早於結束日期"); st.stop()

# ══════════════════════════════════════════
# 資料下載
# ══════════════════════════════════════════
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


# ══════════════════════════════════════════
# 技術指標計算
# ══════════════════════════════════════════
def sharpe_momentum(df_prices, window):
    """報酬 ÷ 波動，過濾高波動低品質的動能"""
    ret      = df_prices.pct_change()
    total    = df_prices.pct_change(window)
    vol      = ret.rolling(window).std() * np.sqrt(window)
    return total / vol.replace(0, np.nan)


def calc_adx(df_prices, asset, window=ADX_WINDOW):
    """
    ADX（平均趨向指數）：衡量趨勢強度，不管方向
    ADX > 20 = 有趨勢（做多）
    ADX < 20 = 震盪（退場持 SHY）
    """
    high  = df_prices[asset] * 1.005   # 用收盤價近似 High/Low
    low   = df_prices[asset] * 0.995
    close = df_prices[asset]

    plus_dm  = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    tr       = (high - low).abs()

    atr       = tr.ewm(span=window, adjust=False).mean()
    plus_di   = 100 * (plus_dm.ewm(span=window, adjust=False).mean() / atr)
    minus_di  = 100 * (minus_dm.ewm(span=window, adjust=False).mean() / atr)
    dx        = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx       = dx.ewm(span=window, adjust=False).mean()
    return adx


def get_corr(df_prices, date, window, assets):
    """計算指定日期往前 window 天的相關矩陣"""
    ret = df_prices[assets].pct_change()
    loc = df_prices.index.get_loc(date)
    if loc < window: return None
    return ret.iloc[loc - window: loc].corr()


def kelly_lev(ret_series, fraction, lev_min, lev_max,
              vix_val=20.0, vix_base=20.0, prev_lev=None):
    """凱利公式 + VIX動態上限 + 每日平滑"""
    r = ret_series.dropna()
    if len(r) < 5:
        return prev_lev if prev_lev is not None else lev_min
    mu = r.mean(); sigma2 = r.var()
    if sigma2 <= 0 or mu <= 0:
        raw = lev_min
    else:
        dyn_max = max(lev_max * (vix_base / max(vix_val, 1.0)), lev_min)
        raw = float(np.clip((mu / sigma2) * fraction, lev_min, dyn_max))
    if prev_lev is not None:
        raw = float(np.clip(raw,
                            prev_lev - KELLY_MAX_DAILY_CHG,
                            prev_lev + KELLY_MAX_DAILY_CHG))
    return raw


def vol_scale(lev, recent_ret, target=VOL_TARGET):
    """若近期波動超過目標，等比例縮槓"""
    r = recent_ret.dropna()
    if len(r) < 5: return lev
    rv = r.std() * np.sqrt(252)
    if rv <= 0: return lev
    return lev * min(1.0, target / rv)


def calc_metrics(ret, rf_daily=0.0):
    r = ret.dropna()
    if len(r) < 2:
        return {k: 0 for k in ["total","cagr","vol","sharpe","mdd","winrate","calmar"]}
    n       = len(r)
    total   = (1 + r).prod() - 1
    cagr    = (1 + total) ** (252/n) - 1
    vol     = r.std() * np.sqrt(252)
    sharpe  = (cagr - rf_daily*252) / vol if vol > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = (cum / cum.cummax() - 1).min()
    calmar  = cagr / abs(mdd) if mdd != 0 else 0
    winrate = (r > 0).mean()
    return {"total": total, "cagr": cagr, "vol": vol,
            "sharpe": sharpe, "mdd": mdd, "winrate": winrate, "calmar": calmar}


# ══════════════════════════════════════════
# 回測引擎（精簡版）
# ══════════════════════════════════════════
def run_backtest(df_all, bt, sharpe_mom, adx_qqq,
                 btc_ma_period, kelly_fraction, kelly_max, kelly_min,
                 kelly_window, vix_base, cash_daily_rate):

    asset_ret    = df_all[TRADABLE + [SAFE_ASSET]].pct_change()
    margin_daily = (1 + MARGIN_ANNUAL_RATE) ** (1/252) - 1
    strat_ret_ts = pd.Series(dtype=float)

    rows     = []
    prev_pk1 = None
    prev_pk2 = None
    prev_lev = None

    for date in bt.index[:-1]:
        if date not in sharpe_mom.index: continue
        loc = df_all.index.get_loc(date)
        if loc + 1 >= len(df_all): continue
        next_day = df_all.index[loc + 1]

        vix_t = bt.loc[date, "VIX"]

        # ── ADX 趨勢強度判斷 ──
        adx_val   = adx_qqq.loc[date] if date in adx_qqq.index else 25.0
        is_trend  = (not pd.isna(adx_val)) and (adx_val >= ADX_THRESHOLD)

        if not is_trend:
            # 震盪：退場持 SHY
            pk1, pk2 = SAFE_ASSET, SAFE_ASSET
            w1,  w2  = 0.6, 0.4
            regime   = "↔️震盪"
        else:
            # ── Sharpe 動能選股 ──
            sm_t = sharpe_mom.loc[date, TRADABLE].dropna().sort_values(ascending=False)
            if len(sm_t) < 2: continue

            pk1 = sm_t.index[0]; pk2 = sm_t.index[1]
            s1  = sm_t.iloc[0];  s2  = sm_t.iloc[1]
            if s1 < 0: pk1 = "CASH"
            if s2 < 0: pk2 = "CASH"

            # BTC 均線過濾
            btc_t    = bt.loc[date, "BTC"]
            btc_ma_t = df_all["BTC"].rolling(btc_ma_period).mean().loc[date]
            if btc_t < btc_ma_t:
                qqq_sm = sharpe_mom.loc[date, "QQQ"] if "QQQ" in sharpe_mom.columns else -1
                if pk1 == "BTC": pk1 = "QQQ" if qqq_sm >= 0 else "CASH"
                if pk2 == "BTC": pk2 = "QQQ" if qqq_sm >= 0 else "CASH"

            # ── 相關性過濾（門檻 0.3）──
            if pk1 not in ["CASH"] and pk2 not in ["CASH"]:
                corr = get_corr(df_all, date, CORR_WINDOW, TRADABLE)
                if corr is not None and pk1 in corr.index and pk2 in corr.index:
                    if corr.loc[pk1, pk2] > CORR_THRESHOLD:
                        others    = [a for a in TRADABLE if a != pk1
                                     and a in sm_t.index and sm_t.loc[a] > 0]
                        if others:
                            pk2 = corr.loc[pk1, others].idxmin()

            w1, w2  = 0.6, 0.4
            is_bull = sharpe_mom.loc[date, "QQQ"] > 0 if "QQQ" in sharpe_mom.columns else True
            regime  = "🐂牛市" if is_bull else "🐻熊市"

        # ── 凱利槓桿（震盪時槓桿設為下限，避免 SHY 也被放大）──
        if not is_trend:
            lev_t    = kelly_min
            prev_lev = lev_t
        else:
            if len(strat_ret_ts) >= kelly_window:
                wr = strat_ret_ts.iloc[-kelly_window:]
            elif len(strat_ret_ts) >= 5:
                wr = strat_ret_ts
            else:
                proxy = asset_ret["QQQ"]
                wr    = proxy.loc[:date].iloc[-kelly_window:]

            lev_t    = kelly_lev(wr, kelly_fraction, kelly_min, kelly_max,
                                 vix_val=vix_t, vix_base=vix_base,
                                 prev_lev=prev_lev)
            vol_wr   = strat_ret_ts.iloc[-20:] if len(strat_ret_ts) >= 20 else wr
            lev_t    = vol_scale(lev_t, vol_wr)
            prev_lev = lev_t

        # ── 手續費 ──
        fee = 0.0
        if prev_pk1 is not None:
            if pk1 != prev_pk1:
                if prev_pk1 not in ["CASH"]: fee += FEES.get(prev_pk1, 0.001) * w1
                if pk1      not in ["CASH"]: fee += FEES.get(pk1,      0.001) * w1
            if pk2 != prev_pk2:
                if prev_pk2 not in ["CASH"]: fee += FEES.get(prev_pk2, 0.001) * w2
                if pk2      not in ["CASH"]: fee += FEES.get(pk2,      0.001) * w2
        prev_pk1 = pk1; prev_pk2 = pk2

        # ── 融資成本 ──
        risky_w     = (w1 if pk1 not in ["CASH",SAFE_ASSET] else 0) + \
                      (w2 if pk2 not in ["CASH",SAFE_ASSET] else 0)
        margin_cost = max(0, lev_t - 1) * risky_w * margin_daily

        # ── 次日報酬 ──
        def next_ret(pick):
            if pick == "CASH": return cash_daily_rate
            if pick not in asset_ret.columns or next_day not in asset_ret.index: return 0.0
            v = asset_ret.loc[next_day, pick]
            return 0.0 if pd.isna(v) else float(v)

        r1 = next_ret(pk1); r2 = next_ret(pk2)

        # SHY / CASH 不乘槓桿
        def apply_lev(pick, r, lev):
            if pick in ["CASH", SAFE_ASSET]: return r
            return r * lev

        day_ret = (w1 * apply_lev(pk1, r1, lev_t) +
                   w2 * apply_lev(pk2, r2, lev_t) - fee - margin_cost)

        strat_ret_ts = pd.concat([strat_ret_ts, pd.Series([day_ret], index=[date])])
        rows.append({"Date": date, "ret": day_ret, "pk1": pk1, "pk2": pk2,
                     "lev": lev_t, "adx": round(adx_val, 1) if not pd.isna(adx_val) else 0,
                     "regime": regime, "fee": fee, "margin": margin_cost})
    return rows


# ══════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════
try:
    with st.spinner("📡 載入市場資料中..."):
        df_all = load_data(TICKER_MAP)

    missing = [c for c in ["VIX","BTC","QQQ","SPY","SHY"] if c not in df_all.columns]
    if missing:
        st.error(f"❌ 缺少關鍵欄位：{missing}"); st.stop()

    df_all["BTC_MA"] = df_all["BTC"].rolling(btc_ma_period).mean()

    # 預計算指標（全段歷史）
    sharpe_mom = sharpe_momentum(df_all[TRADABLE + ["QQQ"]]
                                 if "QQQ" not in TRADABLE
                                 else df_all[TRADABLE], SHARPE_MOM_WINDOW)
    adx_qqq    = calc_adx(df_all, "QQQ", ADX_WINDOW)
    spy_ret    = df_all["SPY"].pct_change()

    bt = df_all.loc[str(start_date): str(end_date)].copy()
    bt = bt.dropna(subset=["VIX","BTC","BTC_MA"])
    if len(bt) < 5:
        st.error(f"❌ 資料不足（{len(bt)}天）"); st.stop()

    # ── 今日訊號 ──
    today      = bt.index[-1]
    vix_now    = bt.loc[today, "VIX"]
    btc_now    = bt.loc[today, "BTC"]
    btc_ma_now = bt.loc[today, "BTC_MA"]
    adx_now    = adx_qqq.loc[today] if today in adx_qqq.index else 0
    is_trend   = (not pd.isna(adx_now)) and (adx_now >= ADX_THRESHOLD)

    sm_today = sharpe_mom.loc[today, TRADABLE].dropna().sort_values(ascending=False)

    if not is_trend:
        p1, p2   = SAFE_ASSET, SAFE_ASSET
        s1_val   = s2_val = 0.0
        regime_now = f"↔️ 震盪（ADX={adx_now:.1f}<{ADX_THRESHOLD}）→ 持 SHY"
    else:
        p1 = sm_today.index[0] if len(sm_today) > 0 else "CASH"
        p2 = sm_today.index[1] if len(sm_today) > 1 else "CASH"
        s1_val = sm_today.iloc[0] if len(sm_today) > 0 else -1.0
        s2_val = sm_today.iloc[1] if len(sm_today) > 1 else -1.0
        if s1_val < 0: p1 = "CASH"
        if s2_val < 0: p2 = "CASH"
        if btc_now < btc_ma_now:
            qqq_sm = sm_today.get("QQQ", -1)
            if p1 == "BTC": p1 = "QQQ" if qqq_sm >= 0 else "CASH"
            if p2 == "BTC": p2 = "QQQ" if qqq_sm >= 0 else "CASH"

        # 相關性過濾今日
        corr_note = ""
        corr_t = get_corr(df_all, today, CORR_WINDOW, TRADABLE)
        if corr_t is not None and p1 not in ["CASH"] and p2 not in ["CASH"]:
            if p1 in corr_t.index and p2 in corr_t.index:
                c12 = corr_t.loc[p1, p2]
                if c12 > CORR_THRESHOLD:
                    others = [a for a in TRADABLE if a != p1
                              and sm_today.get(a, -1) > 0]
                    if others:
                        new_p2 = corr_t.loc[p1, others].idxmin()
                        corr_note = f" ⚡相關{c12:.2f}>{CORR_THRESHOLD}，{p2}→{new_p2}"
                        p2 = new_p2

        is_bull    = sm_today.get("QQQ", -1) > 0
        regime_now = f"{'🐂 牛市' if is_bull else '🐻 熊市'}（ADX={adx_now:.1f}）"

    lev_now = kelly_lev(df_all["QQQ"].pct_change().iloc[-kelly_window:],
                        kelly_fraction, kelly_min, kelly_max,
                        vix_val=vix_now, vix_base=vix_base)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 今日策略訊號")
        st.write(f"📅 {today.strftime('%Y-%m-%d')}　**{preset_name}**")
        if is_trend:
            st.write(f"🥇 **{p1}** (60%)　Sharpe動能 {s1_val:.3f}")
            st.write(f"🥈 **{p2}** (40%)　Sharpe動能 {s2_val:.3f}" +
                     (corr_note if 'corr_note' in dir() else ""))
            st.markdown(f"### 🔥 凱利槓桿：{lev_now:.2f}x")
        else:
            st.warning(f"⏸️ 趨勢不明，全部持 **SHY**（短期美債）")
            st.markdown("### 💤 槓桿：1x（SHY 不放大）")
    with col2:
        st.subheader("⚠️ 風控狀態")
        st.write(f"VIX：**{vix_now:.2f}**　BTC：**{btc_now:,.0f}** / 均線 **{btc_ma_now:,.0f}**")
        st.write(f"市場狀態：**{regime_now}**")
        dyn_max = kelly_max * (vix_base / max(vix_now, 1.0))
        if not is_trend:
            st.info("⏸️ ADX 低於閾值，趨勢引擎暫停")
        elif vix_now > vix_base:
            st.warning(f"⚠️ VIX {vix_now:.1f} > 基準 {vix_base}，動態上限 {dyn_max:.2f}x")
        else:
            st.success("✅ 市場狀態正常，趨勢引擎運行中")

    st.markdown("**📋 今日 Sharpe 動能排名**")
    st.dataframe(pd.DataFrame({
        "標的":       sm_today.index,
        "Sharpe動能": [f"{v:.3f}" for v in sm_today.values],
        "狀態":       ["✅" if v > 0 else "❌" for v in sm_today.values],
    }), use_container_width=True, hide_index=True)

    # ── 回測 ──
    st.markdown("---")
    st.subheader(f"📈 回測（{start_date} ～ {end_date}）vs SPY")

    rows = run_backtest(df_all, bt, sharpe_mom, adx_qqq,
                        btc_ma_period, kelly_fraction, kelly_max, kelly_min,
                        kelly_window, vix_base, cash_daily_rate)

    if not rows:
        st.error("❌ 無有效資料"); st.stop()

    res = pd.DataFrame(rows).set_index("Date")
    res["cum_strat"] = (1 + res["ret"]).cumprod() - 1
    res["cum_spy"]   = (1 + spy_ret.reindex(res.index).fillna(0)).cumprod() - 1

    st.line_chart(
        (res[["cum_strat","cum_spy"]] * 100).rename(
            columns={"cum_strat": f"SEGM Pro {preset_name}", "cum_spy": "SPY 基準"}),
        use_container_width=True
    )

    trade_days   = (res["fee"] > 0).sum()
    total_fee    = res["fee"].sum() * 100
    total_margin = res["margin"].sum() * 100
    trend_days   = (res["regime"] != "↔️震盪").sum()
    sideline_days= (res["regime"] == "↔️震盪").sum()
    st.caption(
        f"💸 手續費：**{total_fee:.2f}%**（{trade_days}天換倉）　"
        f"📊 融資成本：**{total_margin:.2f}%**　"
        f"合計摩擦：**{total_fee+total_margin:.2f}%**　"
        f"｜趨勢運行：**{trend_days}天**　空倉SHY：**{sideline_days}天**"
    )

    # ── 四版本對比 ──
    st.markdown("---")
    st.subheader("📊 三版本績效對比")

    m_cur = calc_metrics(res["ret"], cash_daily_rate)
    m_spy = calc_metrics(spy_ret.reindex(res.index).fillna(0), cash_daily_rate)

    compare_rows = []
    for name, cfg in PRESETS.items():
        if name == "🎯 自訂": continue
        r_list = run_backtest(
            df_all, bt, sharpe_mom, adx_qqq,
            btc_ma_period, cfg["kelly_fraction"], cfg["kelly_max"], cfg["kelly_min"],
            cfg["kelly_window"], cfg["vix_base"], cash_daily_rate
        )
        if not r_list: continue
        r_df = pd.DataFrame(r_list).set_index("Date")
        m    = calc_metrics(r_df["ret"], cash_daily_rate)
        compare_rows.append({
            "版本":       name,
            "總累積報酬": f"{m['total']*100:.1f}%",
            "年化報酬":   f"{m['cagr']*100:.1f}%",
            "最大回撤":   f"{m['mdd']*100:.1f}%",
            "夏普比率":   f"{m['sharpe']:.2f}",
            "Calmar":     f"{m['calmar']:.2f}",
            "勝率":       f"{m['winrate']*100:.1f}%",
        })

    if preset_name == "🎯 自訂":
        compare_rows.append({
            "版本": "🎯 自訂（當前）",
            "總累積報酬": f"{m_cur['total']*100:.1f}%",
            "年化報酬":   f"{m_cur['cagr']*100:.1f}%",
            "最大回撤":   f"{m_cur['mdd']*100:.1f}%",
            "夏普比率":   f"{m_cur['sharpe']:.2f}",
            "Calmar":     f"{m_cur['calmar']:.2f}",
            "勝率":       f"{m_cur['winrate']*100:.1f}%",
        })

    compare_rows.append({
        "版本": "📌 SPY 基準",
        "總累積報酬": f"{m_spy['total']*100:.1f}%",
        "年化報酬":   f"{m_spy['cagr']*100:.1f}%",
        "最大回撤":   f"{m_spy['mdd']*100:.1f}%",
        "夏普比率":   f"{m_spy['sharpe']:.2f}",
        "Calmar":     f"{m_spy['calmar']:.2f}",
        "勝率":       f"{m_spy['winrate']*100:.1f}%",
    })
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

    # ── 詳細績效 ──
    st.subheader(f"📋 {preset_name} 詳細績效")
    c
