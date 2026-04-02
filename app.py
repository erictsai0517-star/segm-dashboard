import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SEGM Pro 儀表板", layout="wide")
st.title("🚀 SEGM Pro｜散戶極致版")

# ══════════════════════════════════════════
# 標的與費率
# ══════════════════════════════════════════
TICKER_MAP = {
    "QQQ": "QQQ", "BTC": "BTC-USD",
    "TLT": "TLT", "GLD": "GLD", "USO": "USO",
    "SPY": "SPY", "VIX": "^VIX",
}
TRADABLE = ["QQQ", "BTC", "TLT", "GLD", "USO"]

FEES = {
    "QQQ": 0.0005, "BTC": 0.001,
    "TLT": 0.0005, "GLD": 0.0005, "USO": 0.001, "CASH": 0.0,
}

# 融資成本：借貸利率（槓桿>1的部分每天要付利息）
# SOFR 約 5.3%，加上券商點差約 1%，保守估計 6%
MARGIN_ANNUAL_RATE = 0.06
MARGIN_DAILY_RATE  = (1 + MARGIN_ANNUAL_RATE) ** (1/252) - 1

# Sharpe動能與相關性的回望窗口（甜蜜點：足夠穩定又不過時）
SHARPE_MOM_WINDOW = 60   # 夏普動能回望
CORR_WINDOW       = 60   # 相關性過濾回望
CORR_THRESHOLD    = 0.7  # 相關係數上限

# 凱利平滑：每天槓桿最多變動 ±0.3x（避免突然爆槓）
KELLY_MAX_DAILY_CHANGE = 0.3

# 波動率目標（年化，超過就等比例縮槓）
VOL_TARGET = 0.20  # 20%

# ══════════════════════════════════════════
# 四個預設版本（所有參數都是甜蜜點）
# ══════════════════════════════════════════
PRESETS = {
    "🚀 激進": {
        "momentum_period": 25, "btc_ma_period": 50,
        "kelly_fraction": 0.9, "kelly_max": 3.5, "kelly_min": 0.8,
        "kelly_window": 30,    "vix_base": 20,
        "bs_vix_spike": 25,    "bs_lev_cap": 0.3, "bs_cooldown": 2,
        "desc": "最大化終值，MDD 較高，適合長期持有且心臟夠強的人。"
    },
    "⚖️ 平衡": {
        "momentum_period": 25, "btc_ma_period": 50,
        "kelly_fraction": 0.65, "kelly_max": 2.5, "kelly_min": 0.5,
        "kelly_window": 40,     "vix_base": 20,
        "bs_vix_spike": 20,     "bs_lev_cap": 0.25, "bs_cooldown": 3,
        "desc": "收益與回撤的甜蜜點，夏普比率最優。散戶極致推薦版。"
    },
    "🛡️ 保守": {
        "momentum_period": 25, "btc_ma_period": 50,
        "kelly_fraction": 0.4, "kelly_max": 1.8, "kelly_min": 0.3,
        "kelly_window": 60,    "vix_base": 18,
        "bs_vix_spike": 15,    "bs_lev_cap": 0.2, "bs_cooldown": 5,
        "desc": "MDD 優先，終值犧牲換穩定，適合資金量大或風險承受低的人。"
    },
    "🎯 自訂": {
        "momentum_period": 25, "btc_ma_period": 50,
        "kelly_fraction": 0.65, "kelly_max": 2.5, "kelly_min": 0.5,
        "kelly_window": 40,     "vix_base": 20,
        "bs_vix_spike": 20,     "bs_lev_cap": 0.25, "bs_cooldown": 3,
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
bs_vix_spike = st.sidebar.slider("VIX 單日暴漲觸發（%）", 10, 60, p["bs_vix_spike"], 5,   disabled=not is_custom)
bs_lev_cap   = st.sidebar.slider("觸發後槓桿上限",        0.1, 1.0, float(p["bs_lev_cap"]), 0.05, disabled=not is_custom)
bs_cooldown  = st.sidebar.slider("冷卻天數",               1,   7,  p["bs_cooldown"],  1,   disabled=not is_custom)

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
# 核心函數
# ══════════════════════════════════════════
def sharpe_momentum(df_prices, window):
    """
    Sharpe動能 = 過去 window 天報酬率 ÷ 過去 window 天日報酬標準差
    報酬高但波動也高的標的（如BTC）會被壓分
    報酬穩定的標的（如GLD）會被加分
    """
    ret   = df_prices.pct_change()
    total = df_prices.pct_change(window)
    vol   = ret.rolling(window).std() * np.sqrt(window)  # 同期間化的波動
    sharpe_mom = total / vol.replace(0, np.nan)
    return sharpe_mom


def get_corr_matrix(df_prices, date, window, assets):
    """計算 date 當天往前 window 天的相關係數矩陣"""
    ret  = df_prices[assets].pct_change()
    loc  = df_prices.index.get_loc(date)
    if loc < window: return None
    window_ret = ret.iloc[loc - window: loc]
    return window_ret.corr()


def kelly_leverage(ret_series, fraction, lev_min, lev_max,
                   vix_val=20.0, vix_base=20.0, prev_lev=None):
    """
    凱利公式 f* = μ/σ²
    加入：
    1. VIX 動態上限
    2. 每日變動平滑（避免槓桿暴跳）
    """
    r = ret_series.dropna()
    if len(r) < 5: return lev_min if prev_lev is None else prev_lev
    mu = r.mean(); sigma2 = r.var()
    if sigma2 <= 0 or mu <= 0:
        raw_lev = lev_min
    else:
        dynamic_max = max(lev_max * (vix_base / max(vix_val, 1.0)), lev_min)
        raw_lev = float(np.clip((mu / sigma2) * fraction, lev_min, dynamic_max))

    # 平滑：每天最多變動 ±KELLY_MAX_DAILY_CHANGE
    if prev_lev is not None:
        raw_lev = float(np.clip(raw_lev,
                                prev_lev - KELLY_MAX_DAILY_CHANGE,
                                prev_lev + KELLY_MAX_DAILY_CHANGE))
    return raw_lev


def vol_target_scale(lev, recent_ret, vol_target=VOL_TARGET):
    """
    波動率目標：如果最近報酬的年化波動超過 vol_target
    等比例縮小槓桿，讓預期波動回到 vol_target
    """
    r = recent_ret.dropna()
    if len(r) < 5: return lev
    realized_vol = r.std() * np.sqrt(252)
    if realized_vol <= 0: return lev
    scale = min(1.0, vol_target / realized_vol)  # 超標才縮，不放大
    return lev * scale


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
# 回測引擎
# ══════════════════════════════════════════
def run_backtest(df_all, bt, sharpe_mom, momentum_period,
                 kelly_fraction, kelly_max, kelly_min, kelly_window,
                 vix_base, bs_vix_spike, bs_lev_cap, bs_cooldown,
                 cash_daily_rate):

    asset_ret     = df_all[TRADABLE].pct_change()
    vix_daily_chg = df_all["VIX"].pct_change()
    spy_ma60      = df_all["SPY"].rolling(60).mean()
    strat_ret_ts  = pd.Series(dtype=float)

    rows       = []
    prev_pk1   = None
    prev_pk2   = None
    prev_lev   = None
    bs_cd_left = 0

    for date in bt.index[:-1]:
        if date not in sharpe_mom.index: continue
        loc = df_all.index.get_loc(date)
        if loc + 1 >= len(df_all): continue
        next_day = df_all.index[loc + 1]

        # ── 1. Sharpe 動能排名（取代純報酬排名）──
        sm_t = sharpe_mom.loc[date, TRADABLE].dropna().sort_values(ascending=False)
        if len(sm_t) < 2: continue

        pk1 = sm_t.index[0]; pk2 = sm_t.index[1]
        s1_t = sm_t.iloc[0]; s2_t = sm_t.iloc[1]

        # Sharpe動能為負 → 轉現金
        if s1_t < 0: pk1 = "CASH"
        if s2_t < 0: pk2 = "CASH"

        # BTC 均線過濾
        btc_t    = bt.loc[date, "BTC"]
        btc_ma_t = bt.loc[date, "BTC_MA"]
        if btc_t < btc_ma_t:
            qqq_sm = sharpe_mom.loc[date, "QQQ"] if "QQQ" in sharpe_mom.columns else -1
            if pk1 == "BTC": pk1 = "QQQ" if qqq_sm >= 0 else "CASH"
            if pk2 == "BTC": pk2 = "QQQ" if qqq_sm >= 0 else "CASH"

        # ── 2. 相關性過濾 ──
        # 如果兩個選出的標的相關係數 > 0.7，第二名換成相關性最低的其他標的
        if pk1 not in ["CASH"] and pk2 not in ["CASH"]:
            corr_mat = get_corr_matrix(df_all, date, CORR_WINDOW, TRADABLE)
            if corr_mat is not None and pk1 in corr_mat.index and pk2 in corr_mat.index:
                corr_12 = corr_mat.loc[pk1, pk2]
                if corr_12 > CORR_THRESHOLD:
                    # 找與 pk1 相關性最低的其他標的
                    others = [a for a in TRADABLE if a != pk1]
                    corr_with_pk1 = corr_mat.loc[pk1, others].dropna()
                    # 只考慮 Sharpe 動能為正的標的
                    pos_sm = [a for a in others if
                              a in sm_t.index and sm_t.loc[a] > 0]
                    if pos_sm:
                        corr_filtered = corr_with_pk1[pos_sm]
                        if not corr_filtered.empty:
                            pk2 = corr_filtered.idxmin()  # 相關性最低的換上來

        # ── 動態權重（Sharpe差距決定集中度）──
        if pk1 != "CASH" and pk2 != "CASH":
            gap = abs(s1_t - s2_t)
            w1  = min(0.8, 0.6 + gap * 0.5)  # Sharpe差距放大因子0.5（比純報酬更保守）
        else:
            w1 = 0.6
        w2 = 1 - w1

        # ── 市場狀態 ──
        vix_t        = bt.loc[date, "VIX"]
        qqq_sm_t     = sharpe_mom.loc[date, "QQQ"] if "QQQ" in sharpe_mom.columns else 0
        spy_above_ma = df_all.loc[date, "SPY"] > spy_ma60.loc[date] if date in spy_ma60.index else True
        is_bull = (qqq_sm_t > 0) and spy_above_ma
        is_bear = (qqq_sm_t < 0) and (vix_t > vix_base)

        if is_bull:
            dyn_max, dyn_min = kelly_max, kelly_min
        elif is_bear:
            dyn_max, dyn_min = kelly_max * 0.5, kelly_min * 0.5
        else:
            dyn_max, dyn_min = kelly_max * 0.75, kelly_min

        # ── 凱利槓桿（含平滑）──
        if len(strat_ret_ts) >= kelly_window:
            wr = strat_ret_ts.iloc[-kelly_window:]
        elif len(strat_ret_ts) >= 5:
            wr = strat_ret_ts
        else:
            proxy = asset_ret[pk1 if pk1 != "CASH" else "QQQ"]
            wr    = proxy.loc[:date].iloc[-kelly_window:]

        lev_t = kelly_leverage(wr, kelly_fraction, dyn_min, dyn_max,
                               vix_val=vix_t, vix_base=vix_base,
                               prev_lev=prev_lev)

        # ── 波動率目標縮放 ──
        vol_window_ret = strat_ret_ts.iloc[-20:] if len(strat_ret_ts) >= 20 else wr
        lev_t = vol_target_scale(lev_t, vol_window_ret)
        prev_lev = lev_t

        # ── 黑天鵝 ──
        vix_chg_t = vix_daily_chg.loc[date] if date in vix_daily_chg.index else 0
        bs_hit    = (not pd.isna(vix_chg_t)) and (vix_chg_t * 100 > bs_vix_spike)
        if bs_hit: bs_cd_left = bs_cooldown
        if bs_cd_left > 0:
            lev_t      = min(lev_t, bs_lev_cap)
            bs_cd_left -= 1

        # ── 手續費 ──
        fee = 0.0
        if prev_pk1 is not None:
            if pk1 != prev_pk1:
                if prev_pk1 != "CASH": fee += FEES.get(prev_pk1, 0.001) * w1
                if pk1      != "CASH": fee += FEES.get(pk1,      0.001) * w1
            if pk2 != prev_pk2:
                if prev_pk2 != "CASH": fee += FEES.get(prev_pk2, 0.001) * w2
                if pk2      != "CASH": fee += FEES.get(pk2,      0.001) * w2
        prev_pk1 = pk1; prev_pk2 = pk2

        # ── 融資成本（槓桿>1的部分每天付利息）──
        risky_weight = (w1 if pk1 != "CASH" else 0) + (w2 if pk2 != "CASH" else 0)
        borrowed     = max(0, lev_t - 1) * risky_weight  # 借了多少倍
        margin_cost  = borrowed * MARGIN_DAILY_RATE

        # ── 次日報酬 ──
        def next_ret(pick):
            if pick == "CASH": return cash_daily_rate
            if pick not in asset_ret.columns or next_day not in asset_ret.index: return 0.0
            v = asset_ret.loc[next_day, pick]
            return 0.0 if pd.isna(v) else float(v)

        r1 = next_ret(pk1); r2 = next_ret(pk2)
        risky  = (w1*r1 if pk1 != "CASH" else 0) + (w2*r2 if pk2 != "CASH" else 0)
        cash_r = (w1*cash_daily_rate if pk1 == "CASH" else 0) + \
                 (w2*cash_daily_rate if pk2 == "CASH" else 0)
        day_ret = risky * lev_t + cash_r - fee - margin_cost

        strat_ret_ts = pd.concat([strat_ret_ts, pd.Series([day_ret], index=[date])])
        regime = ("🐂" if is_bull else "🐻" if is_bear else "↔") + ("🦢" if bs_hit else "")
        rows.append({"Date": date, "ret": day_ret, "pk1": pk1, "pk2": pk2,
                     "w1": w1, "lev": lev_t, "regime": regime,
                     "fee": fee, "margin": margin_cost})
    return rows


# ══════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════
try:
    with st.spinner("📡 載入市場資料中..."):
        df_all = load_data(TICKER_MAP)

    missing = [c for c in ["VIX","BTC","QQQ","SPY"] if c not in df_all.columns]
    if missing:
        st.error(f"❌ 缺少關鍵欄位：{missing}"); st.stop()

    df_all["BTC_MA"] = df_all["BTC"].rolling(btc_ma_period).mean()

    # Sharpe動能（在完整歷史算，避免邊緣效應）
    sharpe_mom = sharpe_momentum(df_all[TRADABLE + ["QQQ"]]
                                 if "QQQ" not in TRADABLE
                                 else df_all[TRADABLE], SHARPE_MOM_WINDOW)

    spy_ret = df_all["SPY"].pct_change()

    bt = df_all.loc[str(start_date): str(end_date)].copy()
    bt = bt.dropna(subset=["VIX","BTC","BTC_MA"])
    if len(bt) < 5:
        st.error(f"❌ 資料不足（{len(bt)}天）"); st.stop()

    # ── 今日訊號 ──
    today      = bt.index[-1]
    vix_now    = bt.loc[today, "VIX"]
    btc_now    = bt.loc[today, "BTC"]
    btc_ma_now = bt.loc[today, "BTC_MA"]

    sm_today = sharpe_mom.loc[today, TRADABLE].dropna().sort_values(ascending=False)
    p1 = sm_today.index[0] if len(sm_today) > 0 else "CASH"
    p2 = sm_today.index[1] if len(sm_today) > 1 else "CASH"
    s1_val = sm_today.iloc[0] if len(sm_today) > 0 else -1.0
    s2_val = sm_today.iloc[1] if len(sm_today) > 1 else -1.0
    if s1_val < 0: p1 = "CASH"
    if s2_val < 0: p2 = "CASH"
    if btc_now < btc_ma_now:
        if p1 == "BTC": p1 = "QQQ" if (sm_today.get("QQQ", -1) >= 0) else "CASH"
        if p2 == "BTC": p2 = "QQQ" if (sm_today.get("QQQ", -1) >= 0) else "CASH"

    # 今日相關性過濾
    corr_today = get_corr_matrix(df_all, today, CORR_WINDOW, TRADABLE)
    corr_note = ""
    if corr_today is not None and p1 not in ["CASH"] and p2 not in ["CASH"]:
        if p1 in corr_today.index and p2 in corr_today.index:
            c12 = corr_today.loc[p1, p2]
            if c12 > CORR_THRESHOLD:
                others   = [a for a in TRADABLE if a != p1 and sm_today.get(a, -1) > 0]
                if others:
                    new_p2 = corr_today.loc[p1, others].idxmin()
                    corr_note = f"（相關性{c12:.2f}>{CORR_THRESHOLD}，{p2}→{new_p2}）"
                    p2 = new_p2

    w1_now = min(0.8, 0.6 + abs(s1_val - s2_val) * 0.5) if p1 != "CASH" and p2 != "CASH" else 0.6
    w2_now = 1 - w1_now
    lev_now = kelly_leverage(df_all["QQQ"].pct_change().iloc[-kelly_window:],
                             kelly_fraction, kelly_min, kelly_max,
                             vix_val=vix_now, vix_base=vix_base)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 今日策略訊號")
        st.write(f"📅 {today.strftime('%Y-%m-%d')}　版本：**{preset_name}**")
        st.write(f"🥇 **{p1}** ({w1_now:.0%})　Sharpe動能 {s1_val:.2f}" if p1 != "CASH" else f"🥇 **現金** ({w1_now:.0%})")
        st.write(f"🥈 **{p2}** ({w2_now:.0%})　Sharpe動能 {s2_val:.2f}" + corr_note if p2 != "CASH" else f"🥈 **現金** ({w2_now:.0%})")
        st.markdown(f"### 🔥 凱利槓桿：{lev_now:.2f}x")
        st.caption(f"融資成本：年化 {MARGIN_ANNUAL_RATE*100:.0f}%，借 {max(0,lev_now-1):.2f}x = 每年約付 {max(0,lev_now-1)*MARGIN_ANNUAL_RATE*100:.1f}%")
    with col2:
        st.subheader("⚠️ 風控狀態")
        st.write(f"VIX：**{vix_now:.2f}**　BTC：**{btc_now:,.0f}** / 均線 **{btc_ma_now:,.0f}**")
        dyn_max_now  = kelly_max * (vix_base / max(vix_now, 1.0))
        vix_chg_now  = df_all["VIX"].pct_change().iloc[-1] * 100
        if vix_chg_now > bs_vix_spike:
            st.error(f"🦢 黑天鵝！VIX 單日 +{vix_chg_now:.1f}%，明日上限 {bs_lev_cap}x")
        elif vix_now > vix_base:
            st.warning(f"⚠️ VIX {vix_now:.1f} > 基準 {vix_base}，動態上限 {dyn_max_now:.2f}x")
        else:
            st.success("✅ 市場狀態正常")

    st.markdown("**📋 今日 Sharpe 動能排名**")
    st.dataframe(pd.DataFrame({
        "標的":       sm_today.index,
        "Sharpe動能": [f"{v:.3f}" for v in sm_today.values],
        "狀態":       ["✅" if v > 0 else "❌" for v in sm_today.values],
    }), use_container_width=True, hide_index=True)

    # ── 回測 ──
    st.markdown("---")
    st.subheader(f"📈 回測（{start_date} ～ {end_date}）vs SPY")

    rows = run_backtest(df_all, bt, sharpe_mom, momentum_period,
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
            columns={"cum_strat": f"SEGM Pro {preset_name}", "cum_spy": "SPY 基準"}),
        use_container_width=True
    )

    # 成本明細
    trade_days     = (res["fee"] > 0).sum()
    total_fee      = res["fee"].sum() * 100
    total_margin   = res["margin"].sum() * 100
    st.caption(
        f"💸 手續費：**{total_fee:.2f}%**（{trade_days}天換倉）　"
        f"📊 融資成本：**{total_margin:.2f}%**　"
        f"合計摩擦成本：**{total_fee+total_margin:.2f}%**"
    )

    m_cur = calc_metrics(res["ret"], cash_daily_rate)
    m_spy = calc_metrics(spy_ret.reindex(res.index).fillna(0), cash_daily_rate)

    # ── 四版本對比 ──
    st.markdown("---")
    st.subheader("📊 四版本績效對比（含融資成本）")
    compare_rows = []
    for name, cfg in PRESETS.items():
        if name == "🎯 自訂": continue
        r_list = run_backtest(
            df_all, bt, sharpe_mom, cfg["momentum_period"],
            cfg["kelly_fraction"], cfg["kelly_max"], cfg["kelly_min"], cfg["kelly_window"],
            cfg["vix_base"], cfg["bs_vix_spike"], cfg["bs_lev_cap"], cfg["bs_cooldown"],
            cash_daily_rate
        )
        if not r_list: continue
        r_df = pd.DataFrame(r_list).set_index("Date")
        m    = calc_metrics(r_df["ret"], cash_daily_rate)
        compare_rows.append({
            "版本":       name,
            "總累積報酬": f"{m['total']*100:.1f}%",
            "年化報酬":
