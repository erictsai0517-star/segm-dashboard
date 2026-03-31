import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="SEGM 投資儀表板", layout="wide")
st.title("🚀 SEGM 投資儀表板（終極修正版）")

# ==================== 標的設定 ====================
# 【修正1】空頭ETF (SQQQ/SH) 不參與動能排名，單獨處理
TICKER_MAP = {
    "QQQ":  "QQQ",
    "BTC":  "BTC-USD",
    "TQQQ": "TQQQ",
    "SQQQ": "SQQQ",   # 空頭，不進動能排名
    "SH":   "SH",     # 空頭，不進動能排名
    "TLT":  "TLT",
    "GLD":  "GLD",
    "USO":  "USO",
    "SPY":  "SPY",    # 基準
    "VIX":  "^VIX",   # 風控指標
}

# 只有這些標的參與動能排名（排除空頭、VIX、SPY）
TRADABLE = ["QQQ", "TQQQ", "BTC", "TLT", "GLD", "USO"]

# ==================== 側邊欄 ====================
st.sidebar.header("⚙️ 策略與回測設定")
momentum_period  = st.sidebar.number_input("動能週期（天）",      10, 200, 60)
btc_ma_period    = st.sidebar.number_input("BTC 均線週期（天）",   10, 200, 100)
cash_annual_rate = st.sidebar.number_input("現金年化利率（%）",    0.0, 10.0, 2.0)
cash_daily_rate  = (1 + cash_annual_rate / 100) ** (1/252) - 1

st.sidebar.markdown("---")
st.sidebar.markdown("**槓桿閾值設定**")
lev_normal = st.sidebar.slider("正常槓桿 VIX < 20",   1.0, 3.0, 2.5, 0.1)
lev_mild   = st.sidebar.slider("輕度警戒 VIX 20~30",  0.5, 2.0, 0.8, 0.1)
lev_high   = st.sidebar.slider("高度警戒 VIX 30~40",  0.2, 1.5, 0.6, 0.1)
lev_panic  = st.sidebar.slider("極度恐慌 VIX > 40",   0.1, 1.0, 0.5, 0.1)

st.sidebar.markdown("---")
# 【修正2】預設起始日改為有資料的日期
start_date = st.sidebar.date_input("回測起始日期", datetime.date(2020, 1, 1))
end_date   = st.sidebar.date_input("回測結束日期", datetime.date.today())

if start_date >= end_date:
    st.sidebar.error("起始日期必須早於結束日期")
    st.stop()

# ==================== 資料下載 ====================
@st.cache_data(ttl=3600)
def load_data(ticker_map: dict) -> pd.DataFrame:
    """
    【修正3】正確處理 yfinance MultiIndex
    yfinance >= 0.2 下載單一 ticker 時，columns 為 MultiIndex:
      level 0 = 欄位名稱 (Open/High/Low/Close/Volume)
      level 1 = Ticker 代號
    """
    series_dict = {}
    failed = []

    for name, ticker in ticker_map.items():
        try:
            raw = yf.download(ticker, period="max", progress=False, auto_adjust=True)
            if raw.empty:
                failed.append(name)
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                # 正確做法：用 level 0 找 'Close'，再 droplevel
                if "Close" in raw.columns.get_level_values(0):
                    close = raw["Close"]
                    # close 可能是 DataFrame（單ticker時是只有一欄的DF）
                    series_dict[name] = close.squeeze()  # 強制轉成 Series
                else:
                    failed.append(f"{name}(無Close欄)")
            else:
                if "Close" in raw.columns:
                    series_dict[name] = raw["Close"]
                else:
                    failed.append(f"{name}(無Close欄)")

        except Exception as e:
            failed.append(f"{name}({str(e)[:40]})")

    if failed:
        st.sidebar.warning(f"下載失敗：{', '.join(failed)}")

    df = pd.DataFrame(series_dict)
    df = df.ffill().bfill()  # 填補週末/假日空值
    return df


def get_leverage(vix_val: float) -> float:
    if vix_val > 40:   return lev_panic
    elif vix_val > 30: return lev_high
    elif vix_val > 20: return lev_mild
    else:              return lev_normal


def calc_metrics(ret: pd.Series, rf_daily: float = 0.0) -> dict:
    """計算績效指標"""
    r = ret.dropna()
    if len(r) < 2:
        return {"total": 0, "cagr": 0, "vol": 0, "sharpe": 0, "mdd": 0, "winrate": 0}
    n = len(r)
    total    = (1 + r).prod() - 1
    cagr     = (1 + total) ** (252/n) - 1
    vol      = r.std() * np.sqrt(252)
    sharpe   = (cagr - rf_daily*252) / vol if vol > 0 else 0
    cum      = (1 + r).cumprod()
    mdd      = (cum / cum.cummax() - 1).min()
    winrate  = (r > 0).mean()
    return {"total": total, "cagr": cagr, "vol": vol,
            "sharpe": sharpe, "mdd": mdd, "winrate": winrate}


# ==================== 主程式 ====================
try:
    with st.spinner("📡 載入市場資料中..."):
        df_all = load_data(TICKER_MAP)

    # 檢查核心欄位
    missing = [c for c in ["VIX", "BTC", "QQQ", "SPY"] if c not in df_all.columns]
    if missing:
        st.error(f"❌ 缺少關鍵欄位：{missing}，請確認網路並清除快取後重試。")
        st.stop()

    # 技術指標（在完整歷史上算，避免回測區間開頭的 NaN 太多）
    df_all["BTC_MA"] = df_all["BTC"].rolling(btc_ma_period).mean()
    momentum = df_all[TRADABLE].pct_change(momentum_period)

    # 切片到回測區間
    bt = df_all.loc[str(start_date): str(end_date)].copy()
    bt = bt.dropna(subset=["VIX", "BTC", "BTC_MA"])

    if len(bt) < 5:
        st.error(f"❌ 日期區間 {start_date}~{end_date} 的有效資料不足（只有 {len(bt)} 天），請調整起始日期。")
        st.stop()

    # ==================== 今日訊號 ====================
    today       = bt.index[-1]
    vix_now     = bt.loc[today, "VIX"]
    btc_now     = bt.loc[today, "BTC"]
    btc_ma_now  = bt.loc[today, "BTC_MA"]
    lev_now     = get_leverage(vix_now)

    mom_today = momentum.loc[today, TRADABLE].dropna().sort_values(ascending=False)
    p1 = mom_today.index[0] if len(mom_today) > 0 else "CASH"
    p2 = mom_today.index[1] if len(mom_today) > 1 else "CASH"
    m1_val = mom_today.iloc[0] if len(mom_today) > 0 else -1
    m2_val = mom_today.iloc[1] if len(mom_today) > 1 else -1

    if m1_val < 0: p1 = "CASH"
    if m2_val < 0: p2 = "CASH"

    # BTC 均線過濾
    if btc_now < btc_ma_now:
        for side in ["p1", "p2"]:
            if locals()[side] == "BTC":
                qqq_m = momentum.loc[today, "QQQ"] if "QQQ" in momentum.columns else -1
                locals()[side]  # 讀一次（避免 linting 警告）
                if side == "p1": p1 = "QQQ" if qqq_m >= 0 else "CASH"
                else:            p2 = "QQQ" if qqq_m >= 0 else "CASH"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 今日策略訊號")
        st.write(f"📅 訊號日期：{today.strftime('%Y-%m-%d')}")
        st.write(f"🥇 第一名（60%）：**{p1}**　動能：{m1_val:.2%}" if p1 != "CASH" else "🥇 第一名（60%）：**現金防禦**")
        st.write(f"🥈 第二名（40%）：**{p2}**　動能：{m2_val:.2%}" if p2 != "CASH" else "🥈 第二名（40%）：**現金防禦**")
        st.markdown(f"### 🔥 建議槓桿：{lev_now}x")

    with col2:
        st.subheader("⚠️ 風控狀態")
        st.write(f"VIX：**{vix_now:.2f}**")
        st.write(f"BTC：**{btc_now:,.0f}** / {btc_ma_period}日均線：**{btc_ma_now:,.0f}**")
        if vix_now > 30:   st.error("🚨 市場極度恐慌（降槓防禦）")
        elif vix_now > 20: st.warning("⚠️ 市場波動加劇")
        else:              st.success("✅ 市場狀態正常")

    # 動能排名表
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

    # 每日報酬（注意：次日報酬需用 df_all 的下一個交易日，不能用 shift）
    asset_ret = df_all[TRADABLE].pct_change()
    spy_ret   = df_all["SPY"].pct_change()

    rows = []
    dates = bt.index[:-1]  # 最後一天沒有次日

    for date in dates:
        if date not in momentum.index:
            continue

        # 找下一個交易日
        loc = df_all.index.get_loc(date)
        if loc + 1 >= len(df_all):
            continue
        next_day = df_all.index[loc + 1]

        # 當日訊號
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

        vix_t = bt.loc[date, "VIX"]
        lev_t = get_leverage(vix_t)

        # 次日報酬
        def next_ret(pick):
            if pick == "CASH":
                return cash_daily_rate  # 【修正4】現金不乘槓桿（在外面處理）
            if pick not in asset_ret.columns or next_day not in asset_ret.index:
                return 0.0
            v = asset_ret.loc[next_day, pick]
            return 0.0 if pd.isna(v) else v

        r1 = next_ret(pk1)
        r2 = next_ret(pk2)

        # 【修正5】現金部位不乘槓桿，風險部位才乘槓桿
        risky = (0.6 * r1 if pk1 != "CASH" else 0) + (0.4 * r2 if pk2 != "CASH" else 0)
        cash  = (0.6 * cash_daily_rate if pk1 == "CASH" else 0) + \
                (0.4 * cash_daily_rate if pk2 == "CASH" else 0)
        day_ret = risky * lev_t + cash

        rows.append({"Date": date, "ret": day_ret, "pk1": pk1, "pk2": pk2, "lev": lev_t})

    if not rows:
        st.error("❌ 回測區間內沒有有效資料")
        st.stop()

    res = pd.DataFrame(rows).set_index("Date")
    res["cum_strat"] = (1 + res["ret"]).cumprod() - 1
    res["cum_spy"]   = (1 + spy_ret.reindex(res.index).fillna(0)).cumprod() - 1

    # 圖表
    chart = (res[["cum_strat", "cum_spy"]] * 100).rename(
        columns={"cum_strat": "SEGM策略", "cum_spy": "SPY基準"}
    )
    st.line_chart(chart, use_container_width=True)

    # 績效摘要
    m_s = calc_metrics(res["ret"], cash_daily_rate)
    m_b = calc_metrics(spy_ret.reindex(res.index).fillna(0), cash_daily_rate)

    st.subheader("📊 績效摘要")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("年化報酬（CAGR）",  f"{m_s['cagr']:.2%}",   f"SPY: {m_b['cagr']:.2%}")
    c2.metric("夏普比率",          f"{m_s['sharpe']:.2f}", f"SPY: {m_b['sharpe']:.2f}")
    c3.metric("最大回撤（MDD）",   f"{m_s['mdd']:.2%}",    f"SPY: {m_b['mdd']:.2%}")
    c4.metric("勝率",              f"{m_s['winrate']:.1%}", f"SPY: {m_b['winrate']:.1%}")

    d1, d2 = st.columns(2)
    d1.metric("SEGM 總累積報酬", f"{m_s['total']*100:.2f}%",
              f"{(m_s['total'] - m_b['total'])*100:.2f}% vs SPY")
    d2.metric("SPY 總累積報酬",  f"{m_b['total']*100:.2f}%")

    # 最近30天輪動紀錄
    st.subheader("🔄 最近 30 天輪動紀錄")
    recent = res[["pk1","pk2","lev","ret"]].tail(30).copy()
    recent["ret"] = recent["ret"].map(lambda x: f"{x*100:+.2f}%")
    recent["lev"] = recent["lev"].map(lambda x: f"{x}x")
    recent.index = recent.index.strftime("%Y-%m-%d")
    recent.columns = ["第一名 (60%)", "第二名 (40%)", "槓桿", "當日報酬"]
    st.dataframe(recent[::-1], use_container_width=True)

except Exception as e:
    import traceback
    st.error(f"⚠️ 執行錯誤：{e}")
    with st.expander("詳細錯誤訊息"):
        st.code(traceback.format_exc())
