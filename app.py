import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SEGM Pro 儀表板", layout="wide")
st.title("🚀 SEGM Pro｜三策略版")

# ══════════════════════════════════════════
# 常數
# ══════════════════════════════════════════
TICKER_MAP = {
    "QQQ":  "QQQ",     "BTC":  "BTC-USD",
    "TLT":  "TLT",     "IEF":  "IEF",
    "GLD":  "GLD",     "SGOL": "SGOL",
    "USO":  "USO",     "SHY":  "SHY",
    "SPY":  "SPY",     "VIX":  "^VIX",
}

FEES = {
    "QQQ":  0.0005, "BTC":  0.001,
    "TLT":  0.0005, "IEF":  0.0005,
    "GLD":  0.0005, "SGOL": 0.0005,
    "USO":  0.001,  "SHY":  0.0002, "CASH": 0.0,
}

SHARPE_MOM_WINDOW   = 60
CORR_WINDOW         = 60
CORR_THRESHOLD      = 0.3
ADX_WINDOW          = 14
ADX_THRESHOLD       = 20
KELLY_MAX_DAILY_CHG = 0.3
VOL_TARGET          = 0.20
MARGIN_ANNUAL_RATE  = 0.06
SAFE_ASSET          = "SHY"

# ══════════════════════════════════════════
# 三個策略定義
# ══════════════════════════════════════════
STRATEGIES = {
    "🚀 極盡型": {
        "desc":         "最大化長期終值。高槓桿、完整標的池，接受高波動。",
        "tradable":     ["QQQ", "BTC", "IEF", "GLD", "USO"],
        "kelly_f":      0.9,
        "kelly_max":    3.5,
        "kelly_min":    0.8,
        "kelly_window": 30,
        "vix_base":     22,
        "corr_th":      0.3,
        "adx_th":       20,
    },
    "⚖️ 平衡型": {
        "desc":         "長期終值優於 SPY 且 MDD < 25%。QQQ+BTC+IEF，中槓桿。",
        "tradable":     ["QQQ", "BTC", "IEF"],
        "kelly_f":      0.6,
        "kelly_max":    2.5,
        "kelly_min":    0.5,
        "kelly_window": 40,
        "vix_base":     20,
        "corr_th":      0.3,
        "adx_th":       20,
    },
    "🛡️ 防守型": {
        "desc":         "長期穩定跑贏 SPY，MDD < 15%。QQQ+IEF+SGOL，低槓桿、嚴格退場。",
        "tradable":     ["QQQ", "IEF", "SGOL"],
        "kelly_f":      0.4,
        "kelly_max":    1.8,
        "kelly_min":    0.3,
        "kelly_window": 60,
        "vix_base":     18,
        "corr_th":      0.5,
        "adx_th":       18,
    },
}

# ══════════════════════════════════════════
# 側邊欄
# ══════════════════════════════════════════
st.sidebar.header("⚙️ 設定")
strat_name = st.sidebar.selectbox("策略選擇", list(STRATEGIES.keys()))
s = STRATEGIES[strat_name]
st.sidebar.caption(s["desc"])
st.sidebar.markdown("---")

btc_ma_period    = st.sidebar.number_input("BTC 均線週期（天）", 10, 200, 50)
cash_annual_rate = st.sidebar.number_input("現金年化利率（%）",  0.0, 10.0, 2.0)
cash_daily_rate  = (1 + cash_annual_rate / 100) ** (1/252) - 1

st.sidebar.markdown("---")
start_date = st.sidebar.date_input("回測起始日期", datetime.date(2018, 1, 1))
end_date   = st.sidebar.date_input("回測結束日期", datetime.date.today())
if start_date >= end_date:
    st.sidebar.error("起始日期必須早於結束日期"); st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**📋 各策略標的池**")
for name, cfg in STRATEGIES.items():
    st.sidebar.caption(f"{name}：{' · '.join(cfg['tradable'])}")

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
# 技術指標
# ══════════════════════════════════════════
def sharpe_momentum(df_prices, assets, window):
    ret   = df_prices[assets].pct_change()
    total = df_prices[assets].pct_change(window)
    vol   = ret.rolling(window).std() * np.sqrt(window)
    return total / vol.replace(0, np.nan)


def calc_adx(series, window=ADX_WINDOW):
    high  = series * 1.005
    low   = series * 0.995
    plus_dm  = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    tr       = (high - low).abs()
    atr      = tr.ewm(span=window, adjust=False).mean()
    plus_di  = 100 * (plus_dm.ewm(span=window, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=window, adjust=False).mean() / atr)
    dx       = (100 * (plus_di - minus_di).abs() /
                (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    return dx.ewm(span=window, adjust=False).mean()


def get_corr(df_prices, date, window, assets):
    ret = df_prices[assets].pct_change()
    loc = df_prices.index.get_loc(date)
    if loc < window: return None
    return ret.iloc[loc - window: loc].corr()


def kelly_lev(ret_series, fraction, lev_min, lev_max,
              vix_val=20.0, vix_base=20.0, prev_lev=None):
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
    r = recent_ret.dropna()
    if len(r) < 5: return lev
    rv = r.std() * np.sqrt(252)
    return lev * min(1.0, target / rv) if rv > 0 else lev


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
def run_backtest(df_all, bt, cfg, btc_ma_period, cash_daily_rate):
    tradable    = cfg["tradable"]
    kelly_f     = cfg["kelly_f"]
    kelly_max   = cfg["kelly_max"]
    kelly_min   = cfg["kelly_min"]
    kelly_window= cfg["kelly_window"]
    vix_base    = cfg["vix_base"]
    corr_th     = cfg["corr_th"]
    adx_th      = cfg["adx_th"]

    # 預計算 Sharpe 動能
    all_assets = list(set(tradable + ["QQQ"]))
    sm = sharpe_momentum(df_all, all_assets, SHARPE_MOM_WINDOW)

    asset_ret    = df_all[tradable + [SAFE_ASSET]].pct_change()
    margin_daily = (1 + MARGIN_ANNUAL_RATE) ** (1/252) - 1
    strat_ret_ts = pd.Series(dtype=float)

    rows     = []
    prev_pk1 = None
    prev_pk2 = None
    prev_lev = None

    for date in bt.index[:-1]:
        if date not in sm.index: continue
        loc = df_all.index.get_loc(date)
        if loc + 1 >= len(df_all): continue
        next_day = df_all.index[loc + 1]

        vix_t    = bt.loc[date, "VIX"]
        ma50_t   = df_all.loc[date, "QQQ_MA50"]
        ma200_t  = df_all.loc[date, "QQQ_MA200"]
        adx_t    = df_all.loc[date, "ADX_QQQ"]
        golden_x = (not pd.isna(ma50_t)) and (not pd.isna(ma200_t)) and (ma50_t > ma200_t)
        is_trend = (not pd.isna(adx_t)) and (adx_t >= adx_th) and golden_x

        # ── ADX 動態部位縮放 ──
        # ADX 20~40 線性插值，決定做多比例
        # ADX < 20 = 0%做多，ADX >= 40 = 100%做多
        if not is_trend:
            adx_scale = 0.0
        else:
            adx_scale = float(np.clip((adx_t - adx_th) / (40 - adx_th), 0.0, 1.0))

        if adx_scale == 0.0:
            pk1 = pk2 = SAFE_ASSET
            w1  = w2  = 0.5
            regime = "🐻熊市退場" if (not pd.isna(adx_t) and adx_t >= adx_th and not golden_x) \
                     else "↔️震盪退場"
        else:
            # Sharpe 動能選股
            sm_t = sm.loc[date, tradable].dropna().sort_values(ascending=False)
            if len(sm_t) < 2:
                pk1 = pk2 = SAFE_ASSET; w1 = w2 = 0.5
                regime = "↔️震盪退場"
            else:
                pk1 = sm_t.index[0]; pk2 = sm_t.index[1]
                s1  = sm_t.iloc[0];  s2  = sm_t.iloc[1]
                if s1 < 0: pk1 = SAFE_ASSET
                if s2 < 0: pk2 = SAFE_ASSET

                # BTC 均線過濾
                if "BTC" in tradable:
                    btc_t    = bt.loc[date, "BTC"]
                    btc_ma_t = df_all["BTC"].rolling(btc_ma_period).mean().loc[date]
                    if btc_t < btc_ma_t:
                        qqq_sm = sm.loc[date, "QQQ"] if "QQQ" in sm.columns else -1
                        if pk1 == "BTC": pk1 = "QQQ" if qqq_sm >= 0 else SAFE_ASSET
                        if pk2 == "BTC": pk2 = "QQQ" if qqq_sm >= 0 else SAFE_ASSET

                # 相關性過濾
                if pk1 not in [SAFE_ASSET] and pk2 not in [SAFE_ASSET]:
                    corr = get_corr(df_all, date, CORR_WINDOW, tradable)
                    if corr is not None and pk1 in corr.index and pk2 in corr.index:
                        if corr.loc[pk1, pk2] > corr_th:
                            others = [a for a in tradable if a != pk1
                                      and a in sm_t.index and sm_t.loc[a] > 0]
                            if others:
                                pk2 = corr.loc[pk1, others].idxmin()

                w1 = 0.6; w2 = 0.4
                is_bull = sm.loc[date, "QQQ"] > 0 if "QQQ" in sm.columns else True
                regime  = "🐂牛市" if is_bull else "🐻熊市做多"

        # 凱利槓桿
        if pk1 == SAFE_ASSET and pk2 == SAFE_ASSET:
            lev_t    = kelly_min
            prev_lev = lev_t
        else:
            if len(strat_ret_ts) >= kelly_window:
                wr = strat_ret_ts.iloc[-kelly_window:]
            elif len(strat_ret_ts) >= 5:
                wr = strat_ret_ts
            else:
                proxy = df_all["QQQ"].pct_change()
                wr    = proxy.loc[:date].iloc[-kelly_window:]

            lev_t  = kelly_lev(wr, kelly_f, kelly_min, kelly_max,
                               vix_val=vix_t, vix_base=vix_base, prev_lev=prev_lev)
            vol_wr = strat_ret_ts.iloc[-20:] if len(strat_ret_ts) >= 20 else wr
            lev_t  = vol_scale(lev_t, vol_wr)
            prev_lev = lev_t

        # 手續費
        fee = 0.0
        if prev_pk1 is not None:
            if pk1 != prev_pk1:
                if prev_pk1 not in [SAFE_ASSET]: fee += FEES.get(prev_pk1, 0.001) * w1
                if pk1      not in [SAFE_ASSET]: fee += FEES.get(pk1,      0.001) * w1
            if pk2 != prev_pk2:
                if prev_pk2 not in [SAFE_ASSET]: fee += FEES.get(prev_pk2, 0.001) * w2
                if pk2      not in [SAFE_ASSET]: fee += FEES.get(pk2,      0.001) * w2
        prev_pk1 = pk1; prev_pk2 = pk2

        # 融資成本（只對做多部位收）
        risky_w     = adx_scale * ((w1 if pk1 not in [SAFE_ASSET] else 0) +
                                   (w2 if pk2 not in [SAFE_ASSET] else 0))
        margin_cost = max(0, lev_t - 1) * risky_w * margin_daily

        # 次日報酬
        def next_ret(pick):
            if pick == "CASH": return cash_daily_rate
            if pick not in asset_ret.columns or next_day not in asset_ret.index: return 0.0
            v = asset_ret.loc[next_day, pick]
            return 0.0 if pd.isna(v) else float(v)

        r1 = next_ret(pk1); r2 = next_ret(pk2)

        def apply_lev(pick, r, lev):
            return r if pick == SAFE_ASSET else r * lev

        # adx_scale 決定做多比例，其餘放 SHY
        risky_ret = adx_scale * (w1 * apply_lev(pk1, r1, lev_t) +
                                  w2 * apply_lev(pk2, r2, lev_t))
        safe_ret  = (1 - adx_scale) * next_ret(SAFE_ASSET)
        day_ret   = risky_ret + safe_ret - fee - margin_cost

        strat_ret_ts = pd.concat([strat_ret_ts, pd.Series([day_ret], index=[date])])
        rows.append({"Date": date, "ret": day_ret, "pk1": pk1, "pk2": pk2,
                     "lev": lev_t, "scale": round(adx_scale*100), "regime": regime,
                     "fee": fee, "margin": margin_cost})
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

    df_all["BTC_MA"]    = df_all["BTC"].rolling(btc_ma_period).mean()
    df_all["QQQ_MA50"]  = df_all["QQQ"].rolling(50).mean()
    df_all["QQQ_MA200"] = df_all["QQQ"].rolling(200).mean()
    df_all["ADX_QQQ"]   = calc_adx(df_all["QQQ"], ADX_WINDOW)

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
    ma50_now   = df_all.loc[today, "QQQ_MA50"]
    ma200_now  = df_all.loc[today, "QQQ_MA200"]
    adx_now    = df_all.loc[today, "ADX_QQQ"]
    golden_x   = (not pd.isna(ma50_now)) and (not pd.isna(ma200_now)) and (ma50_now > ma200_now)

    cfg_now  = STRATEGIES[strat_name]
    tradable = cfg_now["tradable"]
    sm_today = sharpe_momentum(df_all, list(set(tradable + ["QQQ"])),
                               SHARPE_MOM_WINDOW).loc[today, tradable].dropna().sort_values(ascending=False)

    is_trend_now = (not pd.isna(adx_now)) and (adx_now >= cfg_now["adx_th"]) and golden_x

    if not is_trend_now:
        p1 = p2 = SAFE_ASSET
        cross_state = "🔴 死亡交叉" if golden_x == False and not pd.isna(adx_now) and adx_now >= cfg_now["adx_th"] \
                      else "↔️ 震盪"
        regime_label = f"{cross_state}，持 SHY"
    else:
        p1 = sm_today.index[0] if len(sm_today) > 0 else SAFE_ASSET
        p2 = sm_today.index[1] if len(sm_today) > 1 else SAFE_ASSET
        s1 = sm_today.iloc[0]  if len(sm_today) > 0 else -1.0
        s2 = sm_today.iloc[1]  if len(sm_today) > 1 else -1.0
        if s1 < 0: p1 = SAFE_ASSET
        if s2 < 0: p2 = SAFE_ASSET
        if "BTC" in tradable and btc_now < btc_ma_now:
            sm_qqq = sm_today.get("QQQ", -1)
            if p1 == "BTC": p1 = "QQQ" if sm_qqq >= 0 else SAFE_ASSET
            if p2 == "BTC": p2 = "QQQ" if sm_qqq >= 0 else SAFE_ASSET
        regime_label = "🟢 黃金交叉，趨勢做多"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 今日策略訊號")
        st.write(f"📅 {today.strftime('%Y-%m-%d')}　**{strat_name}**")
        st.write(f"標的池：{' · '.join(tradable)}")
        if is_trend_now:
            st.write(f"🥇 **{p1}** (60%)")
            st.write(f"🥈 **{p2}** (40%)")
            lev_disp = kelly_lev(df_all["QQQ"].pct_change().iloc[-cfg_now["kelly_window"]:],
                                 cfg_now["kelly_f"], cfg_now["kelly_min"], cfg_now["kelly_max"],
                                 vix_val=vix_now, vix_base=cfg_now["vix_base"])
            st.markdown(f"### 🔥 凱利槓桿：{lev_disp:.2f}x")
        else:
            st.warning(f"⏸️ 退場中，持 **SHY**（短期美債）")
            st.markdown("### 💤 槓桿：1x")
    with col2:
        st.subheader("⚠️ 風控狀態")
        st.write(f"VIX：**{vix_now:.2f}**")
        st.write(f"BTC：**{btc_now:,.0f}** / {btc_ma_period}日均線：**{btc_ma_now:,.0f}**")
        cross_txt = f"🟢 黃金交叉（MA50={ma50_now:.1f} > MA200={ma200_now:.1f}）" if golden_x \
                    else f"🔴 死亡交叉（MA50={ma50_now:.1f} < MA200={ma200_now:.1f}）"
        st.write(f"QQQ 均線：{cross_txt}")
        st.write(f"ADX：**{adx_now:.1f}**　市場：**{regime_label}**")
        if golden_x and not pd.isna(adx_now) and adx_now >= cfg_now["adx_th"]:
            st.success("✅ 趨勢向上，引擎全開")
        elif not golden_x:
            st.error("🐻 死亡交叉，熊市退場")
        else:
            st.warning(f"↔️ ADX={adx_now:.1f} 低於閾值，震盪退場")

    st.markdown("**📋 今日 Sharpe 動能排名**")
    st.dataframe(pd.DataFrame({
        "標的":       sm_today.index,
        "Sharpe動能": [f"{v:.3f}" for v in sm_today.values],
        "狀態":       ["✅" if v > 0 else "❌" for v in sm_today.values],
    }), use_container_width=True, hide_index=True)

    # ── 回測（當前策略）──
    st.markdown("---")
    st.subheader(f"📈 回測（{start_date} ～ {end_date}）vs SPY")

    rows = run_backtest(df_all, bt, cfg_now, btc_ma_period, cash_daily_rate)
    if not rows:
        st.error("❌ 無有效資料"); st.stop()

    res = pd.DataFrame(rows).set_index("Date")
    res["cum_strat"] = (1 + res["ret"]).cumprod() - 1
    res["cum_spy"]   = (1 + spy_ret.reindex(res.index).fillna(0)).cumprod() - 1

    st.line_chart(
        (res[["cum_strat","cum_spy"]] * 100).rename(
            columns={"cum_strat": strat_name, "cum_spy": "SPY 基準"}),
        use_container_width=True
    )

    trade_days    = (res["fee"] > 0).sum()
    total_fee     = res["fee"].sum() * 100
    total_margin  = res["margin"].sum() * 100
    trend_days    = res["regime"].str.contains("牛市").sum()
    bear_days     = res["regime"].str.contains("熊市退場").sum()
    sideline_days = res["regime"].str.contains("震盪退場").sum()
    st.caption(
        f"💸 手續費：**{total_fee:.2f}%**（{trade_days}天換倉）　"
        f"📊 融資成本：**{total_margin:.2f}%**　合計：**{total_fee+total_margin:.2f}%**　｜　"
        f"趨勢做多：**{trend_days}天**　熊市退場：**{bear_days}天**　震盪退場：**{sideline_days}天**"
    )

    # ── 三策略對比 ──
    st.markdown("---")
    st.subheader("📊 三策略績效對比")

    m_cur = calc_metrics(res["ret"], cash_daily_rate)
    m_spy = calc_metrics(spy_ret.reindex(res.index).fillna(0), cash_daily_rate)

    compare_rows = []
    for name, cfg in STRATEGIES.items():
        r_list = run_backtest(df_all, bt, cfg, btc_ma_period, cash_daily_rate)
        if not r_list: continue
        r_df = pd.DataFrame(r_list).set_index("Date")
        m    = calc_metrics(r_df["ret"], cash_daily_rate)
        compare_rows.append({
            "策略":       name,
            "標的池":     " · ".join(cfg["tradable"]),
            "總累積報酬": f"{m['total']*100:.1f}%",
            "年化報酬":   f"{m['cagr']*100:.1f}%",
            "最大回撤":   f"{m['mdd']*100:.1f}%",
            "夏普比率":   f"{m['sharpe']:.2f}",
            "Calmar":     f"{m['calmar']:.2f}",
            "勝率":       f"{m['winrate']*100:.1f}%",
        })

    compare_rows.append({
        "策略": "📌 SPY 基準",
        "標的池": "SPY",
        "總累積報酬": f"{m_spy['total']*100:.1f}%",
        "年化報酬":   f"{m_spy['cagr']*100:.1f}%",
        "最大回撤":   f"{m_spy['mdd']*100:.1f}%",
        "夏普比率":   f"{m_spy['sharpe']:.2f}",
        "Calmar":     f"{m_spy['calmar']:.2f}",
        "勝率":       f"{m_spy['winrate']*100:.1f}%",
    })
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

    # ── 詳細績效 ──
    st.subheader(f"📋 {strat_name} 詳細績效")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("年化報酬 CAGR", f"{m_cur['cagr']:.2%}",    f"SPY: {m_spy['cagr']:.2%}")
    c2.metric("夏普比率",       f"{m_cur['sharpe']:.2f}",  f"SPY: {m_spy['sharpe']:.2f}")
    c3.metric("最大回撤 MDD",  f"{m_cur['mdd']:.2%}",     f"SPY: {m_spy['mdd']:.2%}")
    c4.metric("Calmar 比率",   f"{m_cur['calmar']:.2f}",  f"SPY: {m_spy['calmar']:.2f}")
    c5.metric("勝率",           f"{m_cur['winrate']:.1%}", f"SPY: {m_spy['winrate']:.1%}"
    d1, d2 = st.columns(2)
    d1.metric("總累積報酬", f"{m_cur['total']*100:.2f}%",
              f"{(m_cur['total']-m_spy['total'])*100:.2f}% vs SPY")
    d2.metric("SPY 總累積報酬", f"{m_spy['total']*100:.2f}%")

    with st.expander("📊 凱利槓桿 & ADX 歷史"):
        st.line_chart(res["lev"].rename("槓桿"), use_container_width=True)
        adx_bt = df_all["ADX_QQQ"].reindex(res.index)
        st.line_chart(adx_bt.rename("ADX"), use_container_width=True)

    st.subheader("🔄 最近 30 天輪動紀錄")
    recent = res[["pk1","pk2","lev","scale","regime","fee","ret"]].tail(30).copy()
    recent["ret"]   = recent["ret"].map(lambda x: f"{x*100:+.2f}%")
    recent["lev"]   = recent["lev"].map(lambda x: f"{x:.2f}x")
    recent["scale"] = recent["scale"].map(lambda x: f"{x}%")
    recent["fee"]   = recent["fee"].map(lambda x: f"{x*100:.3f}%" if x > 0 else "-")
    recent.index    = recent.index.strftime("%Y-%m-%d")
    recent.columns  = ["第一名","第二名","槓桿","做多比例","市場狀態","手續費","當日報酬"]
    st.dataframe(recent[::-1], use_container_width=True)

except Exception as e:
    import traceback
    st.error(f"⚠️ 執行錯誤：{e}")
    with st.expander("詳細錯誤"):
        st.code(traceback.format_exc())
