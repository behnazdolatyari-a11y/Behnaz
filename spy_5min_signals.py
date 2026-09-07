# Run this once in your terminal or Colab cell (with the ! prefix) before running the script:
# pip install yfinance plotly scipy scikit-learn


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go

# ---- Configuration ----
SYMBOL = "SPY"
INTERVAL = "5m"        # 5-minute candles
PERIOD = "60d"         # Yahoo's max lookback for 5m bars — gives a wide backtest range
PIVOT_WINDOW = 5              # bars each side required to confirm a swing high/low
SR_TOLERANCE_PCT = 0.0015     # cluster tolerance for horizontal support/resistance (0.15%)
HARMONIC_TOLERANCE = 0.07     # tolerance band around ideal Fibonacci ratios
TRENDLINE_LOOKBACK = 6        # how many recent swing points to fit each trend line on
CAUSAL_LOOKBACK = 150         # bars of history the rule engine looks back over at each step
WARMUP_BARS = 60              # bars needed before the first signal can be computed

# ---- Machine learning configuration ----
FORWARD_HORIZON = 6           # bars ahead the model predicts (6 x 5min = 30 min)
LABEL_THRESHOLD = 0.0006      # forward move must exceed this (0.06%) to count as up/down, else "flat"
TRAIN_FRACTION = 0.7          # older 70% of history -> training, newer 30% -> honest out-of-sample test
ML_WEIGHT = 1.0               # how much the ML model's opinion counts vs. the rule engine's
RANDOM_STATE = 42
N_FOLDS = 4                   # walk-forward validation folds

# ---- Noise filters (these exist to cut down false signals) ----
SIGNAL_THRESHOLD = 4          # |final score| needed to fire a signal. Raised from 2: being pickier
                              # is the most direct lever against "too many wrong signals". Run
                              # tune_filters() on your own data and set this from the results.
COOLDOWN_BARS = 6             # min bars between consecutive signals, kills whipsaw repeats
ADX_MIN = 20                  # below this the market is ranging/choppy -> suppress signals entirely
VWAP_NEUTRAL_BAND = 0.0005    # price must sit >0.05% away from VWAP before VWAP votes either way
USE_ATR_STOPS = False         # False = score a signal purely on "did price go the way it said"
                              # (that's what 'wrong signal' means). Set True to add a 1xATR stop /
                              # 1.5xATR target instead — more realistic risk management, but a stop
                              # nearer than the target mechanically LOWERS the win rate while making
                              # the winners bigger, so don't compare the two modes' win rates directly.
ATR_STOP_MULT = 1.0           # stop distance = 1.0 x ATR
ATR_TARGET_MULT = 1.5         # target distance = 1.5 x ATR

DROP_INCOMPLETE_BAR = True    # ignore the still-forming candle — see drop_incomplete_bar()
CONFLICT_MAX = 0.35           # veto a signal when the opposing evidence is more than this
                              # fraction of the supporting evidence. Summing votes hid the
                              # difference between "+5, nothing against" and "+6 vs -1"; on real
                              # SPY data ~1 in 4 signals fired while its own indicators disagreed
                              # (BUYs on RSI 84, SELLs citing a confirmed Double Bottom).
AVOID_OVERNIGHT = True        # don't fire a signal whose exit would land past the closing bell.
                              # A 30-minute horizon starting at 15:55 is really an overnight hold:
                              # in the live run those 5 trades produced both the biggest loss
                              # (-0.64%) and the 2nd biggest win (+0.55%) — gap variance, not edge.

# ---- Chart ----
CHART_DAYS = 21               # trading days of history drawn on the signal chart (~1 month)

FEATURE_COLS = [
    "ema_stack", "ema9_21_pct", "ema21_50_pct", "macd_bias", "macd_hist_pct", "rsi",
    "vwap_diff_pct", "bb_pct", "atr_pct", "vol_ratio", "dist_support_pct", "dist_resistance_pct",
    "resistance_break", "support_break", "classic_bull", "classic_bear", "harmonic_bull",
    "harmonic_bear", "rule_score",
    "ret_1", "ret_3", "ret_6", "ret_12", "realized_vol_12", "minutes_since_open",
    "adx", "di_spread", "conflict", "bull_pts", "bear_pts",
]


def interval_minutes(interval=INTERVAL):
    return int("".join(ch for ch in interval if ch.isdigit()) or 5)


def drop_incomplete_bar(df, interval=INTERVAL):
    """Drops the final candle while it is still forming.

    This matters a lot if you re-run during the session. Fetch at 10:32 and the
    10:30 bar holds only two minutes of trading — its high/low/close, and every
    indicator built on them, will change by 10:35. Signals computed on that
    half-built candle flip or vanish on the next refresh, which looks exactly like
    the tool contradicting itself. Waiting for the bar to close costs up to five
    minutes of latency and buys a signal that won't be retracted."""
    if df.empty:
        return df
    bar_close = df.index[-1] + pd.Timedelta(minutes=interval_minutes(interval))
    if pd.Timestamp.now(tz=df.index.tz) < bar_close:
        return df.iloc[:-1]
    return df


def fetch_data(symbol=SYMBOL, interval=INTERVAL, period=PERIOD, drop_partial=DROP_INCOMPLETE_BAR):
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError("No data returned — market may be closed, symbol invalid, or rate-limited.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Datetime"
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df = df.tz_convert("America/New_York")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if drop_partial:
        df = drop_incomplete_bar(df, interval)
    return df


def add_indicators(df):
    df = df.copy()

    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_mid"] = sma20
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # ADX / directional index — measures how *trending* vs. *choppy* the tape is.
    # Most false intraday signals come from firing inside a sideways range, so this
    # is what the chop filter keys off of.
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr_for_di = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_for_di
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_for_di
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    df["PlusDI"] = plus_di.fillna(0)
    df["MinusDI"] = minus_di.fillna(0)
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)

    df["VolSMA20"] = df["Volume"].rolling(20).mean()

    # Session VWAP — resets every trading day. All of the above are rolling/ewm
    # (backward-looking only) so every value here is causal: it only uses bars
    # up to and including its own row.
    session_date = df.index.date
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv = typical * df["Volume"]
    df["VWAP"] = tpv.groupby(session_date).cumsum() / df["Volume"].groupby(session_date).cumsum()

    return df


def add_session_guard(df, horizon=FORWARD_HORIZON, interval=INTERVAL):
    """Flags bars whose `horizon`-bar exit would fall past that session's last bar.

    A 6-bar (30-minute) horizon starting at 15:55 doesn't exit 30 minutes later —
    it exits at tomorrow's open, straight through the overnight gap. That's a
    different trade with different risk, and in the live run those few trades
    supplied both the largest loss and the second-largest win: pure gap variance
    dressed up as signal performance.

    This looks only at the clock, never at future prices, so it introduces no
    lookahead — at 15:55 you already know the 6th bar from now is tomorrow. The
    per-session last bar is measured from the data, so half-days are handled; the
    final session is usually still in progress, so it borrows the typical close
    rather than mistaking "the day isn't over yet" for an early close."""
    df = df.copy()
    if df.empty:
        df["SpansClose"] = False
        return df

    mins = interval_minutes(interval)
    bar_minute = pd.Series(df.index.hour * 60 + df.index.minute, index=df.index)
    session = pd.Series(df.index.normalize(), index=df.index)

    per_session_last = bar_minute.groupby(session).max()
    typical_last = int(per_session_last.median())
    # The newest session may be mid-flight; don't let its partial data look like an early close.
    newest = per_session_last.index[-1]
    per_session_last.loc[newest] = max(per_session_last.loc[newest], typical_last)

    last_bar_minute = session.map(per_session_last)
    df["SpansClose"] = (bar_minute + horizon * mins) > last_bar_minute
    return df


def find_pivots(df, window=PIVOT_WINDOW):
    highs = df["High"].values
    lows = df["Low"].values
    piv_high_idx = argrelextrema(highs, np.greater, order=window)[0]
    piv_low_idx = argrelextrema(lows, np.less, order=window)[0]
    return sorted(piv_high_idx.tolist()), sorted(piv_low_idx.tolist())


def cluster_levels(df, piv_idx, price_col, tolerance_pct=SR_TOLERANCE_PCT, min_touches=2, max_levels=6):
    prices = df[price_col].values[piv_idx]
    if len(prices) == 0:
        return []
    levels = []
    for p in sorted(prices):
        placed = False
        for lvl in levels:
            if abs(p - lvl["price"]) / lvl["price"] <= tolerance_pct:
                lvl["prices"].append(p)
                lvl["price"] = float(np.mean(lvl["prices"]))
                lvl["touches"] += 1
                placed = True
                break
        if not placed:
            levels.append({"price": float(p), "prices": [p], "touches": 1})
    levels = [lvl for lvl in levels if lvl["touches"] >= min_touches]
    levels.sort(key=lambda x: -x["touches"])
    return levels[:max_levels]


def get_support_resistance(df, piv_high_idx, piv_low_idx):
    resistance = cluster_levels(df, piv_high_idx, "High")
    support = cluster_levels(df, piv_low_idx, "Low")
    return support, resistance


def fit_trendline(idx_list, price_list, n_recent=TRENDLINE_LOOKBACK):
    if len(idx_list) < 2:
        return None
    idx_arr = np.array(idx_list[-n_recent:], dtype=float)
    price_arr = np.array(price_list[-n_recent:], dtype=float)
    slope, intercept = np.polyfit(idx_arr, price_arr, 1)
    return float(slope), float(intercept)


def get_trendlines(df, piv_high_idx, piv_low_idx):
    resistance_line = None
    support_line = None
    if len(piv_high_idx) >= 2:
        resistance_line = fit_trendline(piv_high_idx, df["High"].values[piv_high_idx])
    if len(piv_low_idx) >= 2:
        support_line = fit_trendline(piv_low_idx, df["Low"].values[piv_low_idx])
    return support_line, resistance_line


def detect_double_top_bottom(df, piv_high_idx, piv_low_idx, tolerance=0.002):
    patterns = []
    if len(piv_high_idx) >= 2:
        i1, i2 = piv_high_idx[-2], piv_high_idx[-1]
        p1, p2 = df["High"].iloc[i1], df["High"].iloc[i2]
        if abs(p1 - p2) / p1 <= tolerance:
            neckline = df["Low"].iloc[i1:i2 + 1].min()
            confirmed = bool(df["Close"].iloc[-1] < neckline)
            patterns.append({"pattern": "Double Top", "bias": "bearish",
                              "neckline": float(neckline), "index": i2, "confirmed": confirmed})
    if len(piv_low_idx) >= 2:
        i1, i2 = piv_low_idx[-2], piv_low_idx[-1]
        p1, p2 = df["Low"].iloc[i1], df["Low"].iloc[i2]
        if abs(p1 - p2) / p1 <= tolerance:
            neckline = df["High"].iloc[i1:i2 + 1].max()
            confirmed = bool(df["Close"].iloc[-1] > neckline)
            patterns.append({"pattern": "Double Bottom", "bias": "bullish",
                              "neckline": float(neckline), "index": i2, "confirmed": confirmed})
    return patterns


def detect_head_shoulders(df, piv_high_idx, piv_low_idx, tolerance=0.01):
    patterns = []
    if len(piv_high_idx) >= 3:
        i1, i2, i3 = piv_high_idx[-3:]
        p1, p2, p3 = df["High"].iloc[i1], df["High"].iloc[i2], df["High"].iloc[i3]
        if p2 > p1 and p2 > p3 and abs(p1 - p3) / p1 <= tolerance:
            neckline = df["Low"].iloc[i1:i3 + 1].min()
            confirmed = bool(df["Close"].iloc[-1] < neckline)
            patterns.append({"pattern": "Head & Shoulders", "bias": "bearish",
                              "neckline": float(neckline), "index": i3, "confirmed": confirmed})
    if len(piv_low_idx) >= 3:
        i1, i2, i3 = piv_low_idx[-3:]
        p1, p2, p3 = df["Low"].iloc[i1], df["Low"].iloc[i2], df["Low"].iloc[i3]
        if p2 < p1 and p2 < p3 and abs(p1 - p3) / p1 <= tolerance:
            neckline = df["High"].iloc[i1:i3 + 1].max()
            confirmed = bool(df["Close"].iloc[-1] > neckline)
            patterns.append({"pattern": "Inverse Head & Shoulders", "bias": "bullish",
                              "neckline": float(neckline), "index": i3, "confirmed": confirmed})
    return patterns


def detect_triangle(support_line, resistance_line, avg_price, flat_thresh_pct=0.0002):
    if support_line is None or resistance_line is None or avg_price == 0:
        return None
    s_slope_pct = support_line[0] / avg_price
    r_slope_pct = resistance_line[0] / avg_price
    if abs(r_slope_pct) <= flat_thresh_pct and s_slope_pct > flat_thresh_pct:
        return {"pattern": "Ascending Triangle", "bias": "bullish", "confirmed": False}
    if abs(s_slope_pct) <= flat_thresh_pct and r_slope_pct < -flat_thresh_pct:
        return {"pattern": "Descending Triangle", "bias": "bearish", "confirmed": False}
    if r_slope_pct < -flat_thresh_pct and s_slope_pct > flat_thresh_pct:
        return {"pattern": "Symmetrical Triangle", "bias": "neutral", "confirmed": False}
    return None


HARMONIC_DEFS = {
    "Gartley":   {"AB_XA": (0.618, 0.618), "BC_AB": (0.382, 0.886), "CD_BC": (1.13, 1.618), "AD_XA": (0.786, 0.786)},
    "Bat":       {"AB_XA": (0.382, 0.5),   "BC_AB": (0.382, 0.886), "CD_BC": (1.618, 2.618), "AD_XA": (0.886, 0.886)},
    "Butterfly": {"AB_XA": (0.786, 0.786), "BC_AB": (0.382, 0.886), "CD_BC": (1.618, 2.24),  "AD_XA": (1.27, 1.618)},
    "Crab":      {"AB_XA": (0.382, 0.618), "BC_AB": (0.382, 0.886), "CD_BC": (2.24, 3.618),  "AD_XA": (1.618, 1.618)},
}


def get_alternating_pivots(df, piv_high_idx, piv_low_idx):
    """Merge swing highs/lows into one time-ordered, strictly alternating sequence."""
    points = [(i, float(df["High"].iloc[i]), "H") for i in piv_high_idx]
    points += [(i, float(df["Low"].iloc[i]), "L") for i in piv_low_idx]
    points.sort(key=lambda x: x[0])
    filtered = []
    for pt in points:
        if not filtered:
            filtered.append(pt)
            continue
        last = filtered[-1]
        if pt[2] == last[2]:
            if pt[2] == "H" and pt[1] > last[1]:
                filtered[-1] = pt
            elif pt[2] == "L" and pt[1] < last[1]:
                filtered[-1] = pt
        else:
            filtered.append(pt)
    return filtered


def ratio_in_range(value, lo, hi, tol=HARMONIC_TOLERANCE):
    return (lo - tol) <= value <= (hi + tol)


def detect_harmonic_patterns(alt_points):
    results = []
    if len(alt_points) < 5:
        return results
    X, A, B, C, D = alt_points[-5:]
    xi, xp, xt = X
    ai, ap, at = A
    bi, bp, bt = B
    ci, cp, ct = C
    di, dp, dt = D

    if not (xt == bt == dt and at == ct and xt != at):
        return results

    bullish = xt == "L"  # X,B,D are swing lows -> pattern completes into a bullish reversal at D
    XA, AB, BC, CD, AD = abs(ap - xp), abs(bp - ap), abs(cp - bp), abs(dp - cp), abs(dp - ap)
    if XA == 0 or AB == 0 or BC == 0:
        return results

    ab_xa, bc_ab, cd_bc, ad_xa = AB / XA, BC / AB, CD / BC, AD / XA
    for name, r in HARMONIC_DEFS.items():
        if (ratio_in_range(ab_xa, *r["AB_XA"]) and ratio_in_range(bc_ab, *r["BC_AB"]) and
                ratio_in_range(cd_bc, *r["CD_BC"]) and ratio_in_range(ad_xa, *r["AD_XA"])):
            results.append({
                "pattern": name,
                "bias": "bullish" if bullish else "bearish",
                "D_index": di,
                "D_price": dp,
            })
    return results


def generate_signal(df, support, resistance, support_line, resistance_line,
                     classic_patterns, harmonic_patterns):
    """Scores the LAST bar of `df` and also returns a dict of numeric features
    describing that same bar — the features feed the ML model below, while the
    score/reasons remain a fully human-readable rule-based opinion on their own.

    Every piece of evidence is recorded as a separate signed vote rather than
    being added straight into a running total. Summing immediately destroys the
    distinction between "+5 with nothing against it" and "+6 against -1", which
    are very different situations: the second one means the indicators disagree.
    Keeping the votes lets `conflict` below measure that disagreement."""
    last = df.iloc[-1]
    n = len(df) - 1
    votes = []          # (points, reason) — positive = bullish, negative = bearish
    feat = {}

    ema_stack = 0
    if last["EMA9"] > last["EMA21"] > last["EMA50"]:
        votes.append((2, "EMA9 > EMA21 > EMA50 (uptrend)")); ema_stack = 1
    elif last["EMA9"] < last["EMA21"] < last["EMA50"]:
        votes.append((-2, "EMA9 < EMA21 < EMA50 (downtrend)")); ema_stack = -1
    feat["ema_stack"] = ema_stack
    feat["ema9_21_pct"] = (last["EMA9"] - last["EMA21"]) / last["Close"]
    feat["ema21_50_pct"] = (last["EMA21"] - last["EMA50"]) / last["Close"]

    macd_bias = 0
    if last["MACD"] > last["MACD_signal"] and last["MACD_hist"] > 0:
        votes.append((1, "MACD bullish crossover")); macd_bias = 1
    elif last["MACD"] < last["MACD_signal"] and last["MACD_hist"] < 0:
        votes.append((-1, "MACD bearish crossover")); macd_bias = -1
    feat["macd_bias"] = macd_bias
    feat["macd_hist_pct"] = last["MACD_hist"] / last["Close"]

    if last["RSI"] < 30:
        votes.append((1, f"RSI oversold ({last['RSI']:.1f})"))
    elif last["RSI"] > 70:
        votes.append((-1, f"RSI overbought ({last['RSI']:.1f})"))
    feat["rsi"] = last["RSI"]

    # VWAP only votes when price is meaningfully away from it. Previously this was a
    # bare if/else, so it added +1 or -1 on EVERY bar — the score could never sit at
    # rest, which is a big part of why marginal setups kept crossing the threshold.
    vwap_diff_pct = (last["Close"] - last["VWAP"]) / last["VWAP"]
    if vwap_diff_pct > VWAP_NEUTRAL_BAND:
        votes.append((1, f"Price {vwap_diff_pct * 100:.2f}% above VWAP"))
    elif vwap_diff_pct < -VWAP_NEUTRAL_BAND:
        votes.append((-1, f"Price {abs(vwap_diff_pct) * 100:.2f}% below VWAP"))
    feat["vwap_diff_pct"] = vwap_diff_pct

    bb_range = last["BB_upper"] - last["BB_lower"]
    feat["bb_pct"] = (last["Close"] - last["BB_mid"]) / bb_range if bb_range > 0 else 0.0
    if last["Close"] <= last["BB_lower"]:
        votes.append((1, "Price at/below lower Bollinger Band"))
    elif last["Close"] >= last["BB_upper"]:
        votes.append((-1, "Price at/above upper Bollinger Band"))

    feat["atr_pct"] = last["ATR"] / last["Close"]
    feat["vol_ratio"] = (last["Volume"] / last["VolSMA20"]) if last["VolSMA20"] > 0 else 1.0

    support_dists = [abs(last["Close"] - lvl["price"]) / lvl["price"] for lvl in support]
    resistance_dists = [abs(last["Close"] - lvl["price"]) / lvl["price"] for lvl in resistance]
    feat["dist_support_pct"] = min(support_dists) if support_dists else np.nan
    feat["dist_resistance_pct"] = min(resistance_dists) if resistance_dists else np.nan
    for lvl in support:
        if abs(last["Close"] - lvl["price"]) / lvl["price"] <= 0.002:
            votes.append((1, f"Near support {lvl['price']:.2f}"))
    for lvl in resistance:
        if abs(last["Close"] - lvl["price"]) / lvl["price"] <= 0.002:
            votes.append((-1, f"Near resistance {lvl['price']:.2f}"))

    resistance_break = 0
    if resistance_line is not None:
        r_val = resistance_line[0] * n + resistance_line[1]
        if last["Close"] > r_val:
            votes.append((2, "Breakout above resistance trend line")); resistance_break = 1
    feat["resistance_break"] = resistance_break

    support_break = 0
    if support_line is not None:
        s_val = support_line[0] * n + support_line[1]
        if last["Close"] < s_val:
            votes.append((-2, "Breakdown below support trend line")); support_break = 1
    feat["support_break"] = support_break

    classic_bull = 0
    classic_bear = 0
    for pat in classic_patterns:
        if pat.get("confirmed"):
            if pat["bias"] == "bullish":
                votes.append((2, f"{pat['pattern']} confirmed (bullish)")); classic_bull = 1
            elif pat["bias"] == "bearish":
                votes.append((-2, f"{pat['pattern']} confirmed (bearish)")); classic_bear = 1
    feat["classic_bull"] = classic_bull
    feat["classic_bear"] = classic_bear

    harmonic_bull = 0
    harmonic_bear = 0
    for pat in harmonic_patterns:
        if abs(pat["D_index"] - n) <= 3:
            if pat["bias"] == "bullish":
                votes.append((3, f"{pat['pattern']} harmonic bullish completion at D")); harmonic_bull = 1
            else:
                votes.append((-3, f"{pat['pattern']} harmonic bearish completion at D")); harmonic_bear = 1
    feat["harmonic_bull"] = harmonic_bull
    feat["harmonic_bear"] = harmonic_bear

    # ---- Momentum / volatility / time-of-day features (for the ML model) ----
    closes = df["Close"]

    def pct_ago(k):
        return closes.iloc[-1] / closes.iloc[-1 - k] - 1 if len(closes) > k else 0.0

    feat["ret_1"] = pct_ago(1)
    feat["ret_3"] = pct_ago(3)
    feat["ret_6"] = pct_ago(6)
    feat["ret_12"] = pct_ago(12)
    rets = closes.pct_change().dropna()
    feat["realized_vol_12"] = rets.tail(12).std() if len(rets) >= 2 else 0.0

    ts = last.name
    feat["minutes_since_open"] = (ts.hour * 60 + ts.minute) - (9 * 60 + 30)
    feat["adx"] = last["ADX"]
    feat["di_spread"] = last["PlusDI"] - last["MinusDI"]

    # ---- Tally the votes, and measure how much they disagree ----
    score = sum(p for p, _ in votes)
    reasons = [r for _, r in votes]
    bull_pts = sum(p for p, _ in votes if p > 0)
    bear_pts = -sum(p for p, _ in votes if p < 0)
    strong, weak = max(bull_pts, bear_pts), min(bull_pts, bear_pts)
    # 0.0 = every indicator agrees, 1.0 = evidence is evenly split against itself.
    conflict = (weak / strong) if strong > 0 else 0.0
    feat["conflict"] = conflict
    feat["bull_pts"] = bull_pts
    feat["bear_pts"] = bear_pts
    feat["rule_score"] = score

    if score >= 4:
        signal = "STRONG BUY"
    elif score >= 2:
        signal = "BUY"
    elif score <= -4:
        signal = "STRONG SELL"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, score, reasons, feat


def compute_signal_history(df, lookback=CAUSAL_LOOKBACK, warmup=WARMUP_BARS):
    """Walks forward bar by bar. At bar i, only uses rows [i-lookback+1, i] — so a
    signal (and its features) at bar i never sees data from bar i+1 onward."""
    n = len(df)
    signals = ["HOLD"] * n
    scores = [0] * n
    reasons_list = [[] for _ in range(n)]
    feats = [{k: np.nan for k in FEATURE_COLS} for _ in range(n)]

    for i in range(warmup, n):
        start = max(0, i - lookback + 1)
        window = df.iloc[start:i + 1]

        piv_high_idx, piv_low_idx = find_pivots(window)
        support, resistance = get_support_resistance(window, piv_high_idx, piv_low_idx)
        support_line, resistance_line = get_trendlines(window, piv_high_idx, piv_low_idx)

        classic_patterns = []
        classic_patterns += detect_double_top_bottom(window, piv_high_idx, piv_low_idx)
        classic_patterns += detect_head_shoulders(window, piv_high_idx, piv_low_idx)
        triangle = detect_triangle(support_line, resistance_line, window["Close"].mean())
        if triangle:
            classic_patterns.append(triangle)

        alt_points = get_alternating_pivots(window, piv_high_idx, piv_low_idx)
        harmonic_patterns = detect_harmonic_patterns(alt_points)

        signal, score, reasons, feat = generate_signal(window, support, resistance, support_line,
                                                         resistance_line, classic_patterns, harmonic_patterns)
        signals[i] = signal
        scores[i] = score
        reasons_list[i] = reasons
        feats[i] = feat

    out = df.copy()
    out["Signal"] = signals
    out["Score"] = scores
    out["Reasons"] = reasons_list
    feat_df = pd.DataFrame(feats, index=df.index)
    out = pd.concat([out, feat_df], axis=1)
    return out


def add_forward_labels(df, horizon=FORWARD_HORIZON, threshold=LABEL_THRESHOLD):
    """Label = 1 if price is up more than `threshold` after `horizon` bars, -1 if
    down more than `threshold`, else 0 (flat/no edge)."""
    df = df.copy()
    fwd_ret = df["Close"].shift(-horizon) / df["Close"] - 1
    label = pd.Series(0.0, index=df.index)
    label[fwd_ret > threshold] = 1
    label[fwd_ret < -threshold] = -1
    label[fwd_ret.isna()] = np.nan
    df["FwdReturn"] = fwd_ret
    df["Label"] = label
    return df


def chronological_split(df, feature_cols, train_fraction=TRAIN_FRACTION, embargo=FORWARD_HORIZON):
    """Time-ordered split with an embargo: the `embargo` rows immediately before
    the split boundary are dropped from training, because their forward-return
    label windows overlap into the test period — without dropping them, a little
    test-period information leaks into training and inflates reported accuracy."""
    usable = df.dropna(subset=feature_cols + ["Label"])
    split_i = int(len(usable) * train_fraction)
    train_idx = usable.index[:max(0, split_i - embargo)]
    test_idx = usable.index[split_i:]
    return train_idx, test_idx


def prep_features(df, feature_cols=FEATURE_COLS):
    X = df[feature_cols].copy()
    X["dist_support_pct"] = X["dist_support_pct"].fillna(0.05)
    X["dist_resistance_pct"] = X["dist_resistance_pct"].fillna(0.05)
    return X.fillna(0.0)


def make_random_forest(random_state=RANDOM_STATE):
    return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20,
                                   class_weight="balanced", random_state=random_state, n_jobs=-1)


def make_hist_gb(random_state=RANDOM_STATE):
    return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300,
                                           random_state=random_state)


MODEL_CANDIDATES = {"RandomForest": make_random_forest, "HistGB": make_hist_gb}


def train_ml_model(df, train_idx, model_fn=make_random_forest, feature_cols=FEATURE_COLS,
                    random_state=RANDOM_STATE):
    X_train = prep_features(df.loc[train_idx], feature_cols).values
    y_train = df.loc[train_idx, "Label"].values
    model = model_fn(random_state)
    model.fit(X_train, y_train)
    return model


def add_ml_predictions(df, model, feature_cols=FEATURE_COLS):
    df = df.copy()
    mask = df[feature_cols].notna().all(axis=1)
    proba = model.predict_proba(prep_features(df.loc[mask], feature_cols).values)
    classes = list(model.classes_)
    df["P_up"] = np.nan
    df["P_down"] = np.nan
    df["P_flat"] = np.nan
    if 1 in classes:
        df.loc[mask, "P_up"] = proba[:, classes.index(1)]
    if -1 in classes:
        df.loc[mask, "P_down"] = proba[:, classes.index(-1)]
    if 0 in classes:
        df.loc[mask, "P_flat"] = proba[:, classes.index(0)]
    return df


def combine_rule_and_ml(df, ml_weight=ML_WEIGHT, threshold=SIGNAL_THRESHOLD, adx_min=ADX_MIN,
                         conflict_max=CONFLICT_MAX, avoid_overnight=AVOID_OVERNIGHT):
    """Final score = rule-engine score, nudged by how strongly the ML model leans
    up vs. down (scaled to roughly the same range as the rule score). This is
    additive, not an all-or-nothing gate: a strong rule signal can still fire even
    if the model is lukewarm, and vice versa.

    Three vetoes sit on top. Each removes a whole class of bad signal rather than
    trying to out-vote it, and each records why, so a suppressed bar can be
    explained instead of just silently going quiet:

      * `adx_min`       — the tape is ranging, not trending. Chop is where
                          trend-following rules produce their worst false positives.
      * `conflict_max`  — the indicators contradict each other. A +6/-1 split nets
                          to the same +5 as unanimous agreement, but means something
                          entirely different; this refuses to trade the disagreement.
      * `avoid_overnight` — the exit would land past the closing bell, turning a
                          30-minute trade into an overnight gap bet.

    `threshold` still sets how far from zero the score must travel to fire at all."""
    df = df.copy()
    conviction = df["P_up"].fillna(0) - df["P_down"].fillna(0)
    df["Conviction"] = conviction
    df["FinalScore"] = df["Score"].fillna(0) + ml_weight * conviction * 10

    adx = df["ADX"].fillna(0) if "ADX" in df.columns else pd.Series(100.0, index=df.index)
    df["Choppy"] = adx < adx_min

    conflict = df["conflict"].fillna(0) if "conflict" in df.columns else pd.Series(0.0, index=df.index)
    df["Conflicted"] = conflict > conflict_max

    if avoid_overnight and "SpansClose" in df.columns:
        df["Overnight"] = df["SpansClose"].fillna(False)
    else:
        df["Overnight"] = False

    df["Vetoed"] = df["Choppy"] | df["Conflicted"] | df["Overnight"]

    def classify(s, vetoed):
        if pd.isna(s) or vetoed:
            return "HOLD"
        if s >= threshold * 2:
            return "STRONG BUY"
        if s >= threshold:
            return "BUY"
        if s <= -threshold * 2:
            return "STRONG SELL"
        if s <= -threshold:
            return "SELL"
        return "HOLD"

    df["FinalSignal"] = [classify(s, v) for s, v in zip(df["FinalScore"], df["Vetoed"])]
    return df


def classify_raw(score, threshold=SIGNAL_THRESHOLD):
    """What the score alone would have called, ignoring the vetoes."""
    if pd.isna(score):
        return "HOLD"
    if score >= threshold * 2:
        return "STRONG BUY"
    if score >= threshold:
        return "BUY"
    if score <= -threshold * 2:
        return "STRONG SELL"
    if score <= -threshold:
        return "SELL"
    return "HOLD"


def veto_reason(row):
    """Which guard (if any) is holding a signal back on this bar."""
    if row.get("Choppy", False):
        return f"ADX {row.get('ADX', float('nan')):.0f} — tape is ranging, not trending"
    if row.get("Conflicted", False):
        return (f"indicators disagree — {row.get('bull_pts', 0):.0f} points bullish vs "
                f"{row.get('bear_pts', 0):.0f} bearish (conflict {row.get('conflict', 0):.2f})")
    if row.get("Overnight", False):
        return "exit would fall past the closing bell — that's an overnight gap bet"
    return ""


def add_arrow_markers(df, signal_col="FinalSignal", cooldown_bars=COOLDOWN_BARS):
    """Marks the bar where a signal *first* turns BUY or SELL, then refuses to mark
    another one for `cooldown_bars` afterwards.

    Without the cooldown, a score hovering right at the threshold flips
    HOLD->BUY->HOLD->BUY repeatedly through a single choppy stretch and paints a
    cluster of arrows that are really just one (bad) idea repeated. The cooldown
    is applied here, and the backtest reads these same columns, so what you see on
    the chart is exactly what gets scored."""
    df = df.copy()
    buy_states = {"BUY", "STRONG BUY"}
    sell_states = {"SELL", "STRONG SELL"}
    prev_signal = df[signal_col].shift(1).fillna("HOLD")
    raw_buy = (df[signal_col].isin(buy_states) & ~prev_signal.isin(buy_states)).values
    raw_sell = (df[signal_col].isin(sell_states) & ~prev_signal.isin(sell_states)).values

    buy_arrow = np.zeros(len(df), dtype=bool)
    sell_arrow = np.zeros(len(df), dtype=bool)
    last_fired = -10 ** 9
    for i in range(len(df)):
        if not (raw_buy[i] or raw_sell[i]):
            continue
        if i - last_fired < cooldown_bars:
            continue
        if raw_buy[i]:
            buy_arrow[i] = True
        else:
            sell_arrow[i] = True
        last_fired = i

    df["BuyArrow"] = buy_arrow
    df["SellArrow"] = sell_arrow
    return df


def simulate_trade(df, i, direction, horizon=FORWARD_HORIZON, use_atr_stops=USE_ATR_STOPS,
                    atr_stop_mult=ATR_STOP_MULT, atr_target_mult=ATR_TARGET_MULT):
    """Walks a single trade forward from bar `i` and returns how it actually ended.

    With ATR stops on, the trade exits at whichever comes first: stop hit, target
    hit, or `horizon` bars elapsed. When a bar's range covers both the stop and the
    target we assume the stop filled first — pessimistic, but the alternative
    flatters the backtest with fills we can't prove happened in that order."""
    n = len(df)
    if i >= n - 1:
        return None

    entry_price = float(df["Close"].iloc[i])
    atr = float(df["ATR"].iloc[i]) if not pd.isna(df["ATR"].iloc[i]) else 0.0
    last_i = min(i + horizon, n - 1)

    if use_atr_stops and atr > 0:
        if direction == "LONG":
            stop = entry_price - atr_stop_mult * atr
            target = entry_price + atr_target_mult * atr
        else:
            stop = entry_price + atr_stop_mult * atr
            target = entry_price - atr_target_mult * atr

        for j in range(i + 1, last_i + 1):
            bar_high = float(df["High"].iloc[j])
            bar_low = float(df["Low"].iloc[j])
            if direction == "LONG":
                if bar_low <= stop:
                    return _trade_record(df, i, j, direction, entry_price, stop, "stop")
                if bar_high >= target:
                    return _trade_record(df, i, j, direction, entry_price, target, "target")
            else:
                if bar_high >= stop:
                    return _trade_record(df, i, j, direction, entry_price, stop, "stop")
                if bar_low <= target:
                    return _trade_record(df, i, j, direction, entry_price, target, "target")

    return _trade_record(df, i, last_i, direction, entry_price, float(df["Close"].iloc[last_i]), "time")


def _trade_record(df, i, exit_i, direction, entry_price, exit_price, exit_reason):
    ret = (exit_price / entry_price - 1) if direction == "LONG" else (entry_price / exit_price - 1)
    return {"entry_time": df.index[i], "exit_time": df.index[exit_i], "direction": direction,
            "entry_price": entry_price, "exit_price": exit_price, "exit_reason": exit_reason,
            "bars_held": exit_i - i, "return_pct": ret * 100}


def backtest_signals(df, test_idx, horizon=FORWARD_HORIZON, use_atr_stops=USE_ATR_STOPS):
    """Scores every arrow inside the test window. Reads the BuyArrow/SellArrow
    columns produced by add_arrow_markers rather than re-deriving entries, so the
    backtested trades are exactly the arrows drawn on the chart — otherwise the two
    code paths can quietly drift apart and the numbers stop describing the picture.

    SELL trades are scored as if shorting is frictionless, which is a
    simplification: shorting SPY in practice means an inverse ETF or options."""
    if "BuyArrow" not in df.columns:
        raise ValueError("Call add_arrow_markers(df) before backtest_signals(df).")

    test_df = df.loc[test_idx]
    entries = test_df[test_df["BuyArrow"] | test_df["SellArrow"]]

    trades = []
    for ts, row in entries.iterrows():
        i = df.index.get_loc(ts)
        direction = "LONG" if row["BuyArrow"] else "SHORT"
        trade = simulate_trade(df, i, direction, horizon, use_atr_stops)
        if trade is not None:
            trades.append(trade)
    return pd.DataFrame(trades)


def add_signal_outcomes(df, horizon=FORWARD_HORIZON, use_atr_stops=USE_ATR_STOPS):
    """Labels every arrow with how it actually turned out, so the chart can show you
    which past signals worked and which didn't. Signals too recent to have resolved
    are marked 'pending' rather than silently counted as anything."""
    df = df.copy()
    df["Outcome"] = ""
    df["OutcomeReturn"] = np.nan
    df["ExitReason"] = ""

    n = len(df)
    arrows = df.index[df["BuyArrow"] | df["SellArrow"]]
    for ts in arrows:
        i = df.index.get_loc(ts)
        direction = "LONG" if df["BuyArrow"].iloc[i] else "SHORT"
        if i + horizon >= n:
            df.loc[ts, "Outcome"] = "pending"
            continue
        trade = simulate_trade(df, i, direction, horizon, use_atr_stops)
        if trade is None:
            df.loc[ts, "Outcome"] = "pending"
            continue
        df.loc[ts, "Outcome"] = "win" if trade["return_pct"] > 0 else "loss"
        df.loc[ts, "OutcomeReturn"] = trade["return_pct"]
        df.loc[ts, "ExitReason"] = trade["exit_reason"]
    return df


def summarize_backtest(trades):
    if trades is None or trades.empty:
        return {"n_trades": 0}
    wins = trades[trades["return_pct"] > 0]
    losses = trades[trades["return_pct"] <= 0]
    gross_win = wins["return_pct"].sum()
    gross_loss = abs(losses["return_pct"].sum())
    equity_curve = (1 + trades["return_pct"] / 100).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return {
        "n_trades": len(trades), "win_rate_pct": len(wins) / len(trades) * 100,
        "avg_return_pct": trades["return_pct"].mean(), "total_return_pct": trades["return_pct"].sum(),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown_pct": drawdown.min() * 100, "equity_curve": equity_curve,
    }


def summarize_folds(fold_stats_list):
    trades_total = sum(s.get("n_trades", 0) for s in fold_stats_list)
    if trades_total == 0:
        return {"n_trades": 0}
    total_win_trades = sum(s["win_rate_pct"] / 100 * s["n_trades"] for s in fold_stats_list if s.get("n_trades", 0) > 0)
    weighted_return = sum(s["avg_return_pct"] * s["n_trades"] for s in fold_stats_list if s.get("n_trades", 0) > 0)
    return {"n_trades": trades_total, "win_rate_pct": total_win_trades / trades_total * 100,
            "avg_return_pct": weighted_return / trades_total}


def walk_forward_evaluate(df, feature_cols=FEATURE_COLS, n_folds=N_FOLDS, embargo=FORWARD_HORIZON,
                           ml_weight=ML_WEIGHT, random_state=RANDOM_STATE,
                           model_candidates=None):
    """Expanding-window walk-forward validation: fold k trains on everything
    before its test block (with an embargo) and tests on the next block, so every
    fold is a genuine out-of-sample evaluation on a DIFFERENT slice of time. This
    is far more trustworthy than judging accuracy from a single train/test split,
    which can look good or bad purely by luck of which regime the test slice
    happened to land on."""
    if model_candidates is None:
        model_candidates = MODEL_CANDIDATES
    usable = df.dropna(subset=feature_cols + ["Label"]).index
    n = len(usable)
    block = n // (n_folds + 1)
    results = {name: [] for name in model_candidates}
    if block < 50:
        return results

    for fold in range(n_folds):
        train_end = block * (fold + 1)
        test_start = train_end
        test_end = block * (fold + 2) if fold < n_folds - 1 else n
        train_idx = usable[:max(0, train_end - embargo)]
        test_idx = usable[test_start:test_end]
        if len(train_idx) < 100 or len(test_idx) < 20:
            continue

        for name, model_fn in model_candidates.items():
            model = train_ml_model(df, train_idx, model_fn, feature_cols, random_state)
            fold_df = add_ml_predictions(df, model, feature_cols)
            fold_df = combine_rule_and_ml(fold_df, ml_weight)
            fold_df = add_arrow_markers(fold_df)
            trades = backtest_signals(fold_df, test_idx)
            stats = summarize_backtest(trades)
            stats["fold"] = fold + 1
            stats["test_start"] = test_idx.min()
            stats["test_end"] = test_idx.max()
            results[name].append(stats)

    return results


def tune_filters(full, test_idx, ml_weight=ML_WEIGHT, use_atr_stops=USE_ATR_STOPS,
                  thresholds=(2, 3, 4, 5), adx_mins=(0, 15, 20, 25, 30), cooldowns=(0, 6, 12)):
    """Sweeps the noise-filter settings and reports what each combination would have
    done on the held-out period, so you can pick settings from YOUR data instead of
    trusting defaults someone guessed for you.

    Read it with a sceptical eye: each row is a small sample, so a few percent of
    win-rate difference between rows is usually noise, not a real edge. Prefer
    settings that look decent across several neighbouring rows over the single best
    row, which is probably just the luckiest one."""
    rows = []
    for threshold in thresholds:
        for adx_min in adx_mins:
            for cooldown in cooldowns:
                tuned = combine_rule_and_ml(full, ml_weight, threshold=threshold, adx_min=adx_min)
                tuned = add_arrow_markers(tuned, cooldown_bars=cooldown)
                trades = backtest_signals(tuned, test_idx, use_atr_stops=use_atr_stops)
                stats = summarize_backtest(trades)
                rows.append({
                    "threshold": threshold, "adx_min": adx_min, "cooldown": cooldown,
                    "n_trades": stats.get("n_trades", 0),
                    "win_rate_pct": stats.get("win_rate_pct", np.nan),
                    "avg_return_pct": stats.get("avg_return_pct", np.nan),
                    "profit_factor": stats.get("profit_factor", np.nan),
                })
    out = pd.DataFrame(rows)

    print("=" * 78)
    print("Filter tuning on the held-out period (min 20 trades shown, best win rate first)")
    print("A few % between neighbouring rows is noise — look for a stable neighbourhood,")
    print("not the single best row.")
    print("-" * 78)
    shown = out[out["n_trades"] >= 20].sort_values("win_rate_pct", ascending=False)
    if shown.empty:
        print("No setting produced at least 20 trades — loosen the filters or fetch more data.")
    else:
        print(shown.head(15).to_string(index=False,
              formatters={"win_rate_pct": "{:.1f}".format, "avg_return_pct": "{:+.4f}".format,
                          "profit_factor": "{:.2f}".format}))
    print("=" * 78)
    return out


def print_backtest_report(trades, stats, label="Backtest"):
    print("-" * 64)
    print(f"{label} — {stats.get('n_trades', 0)} trades")
    if stats.get("n_trades", 0) == 0:
        print("No trades were triggered in this period.")
        print("-" * 64)
        return
    print(f"  Win rate:        {stats['win_rate_pct']:.1f}%")
    print(f"  Avg return/trade:{stats['avg_return_pct']:+.3f}%")
    print(f"  Total return:    {stats['total_return_pct']:+.2f}%  (sum of per-trade returns)")
    print(f"  Profit factor:   {stats['profit_factor']:.2f}  (gross wins / gross losses)")
    print(f"  Max drawdown:    {stats['max_drawdown_pct']:.2f}%  (on the trade equity curve)")
    print("-" * 64)


def print_feature_importance(model, df, idx, feature_cols=FEATURE_COLS, top_n=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        X = prep_features(df.loc[idx], feature_cols).values
        y = df.loc[idx, "Label"].values
        result = permutation_importance(model, X, y, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
        importances = result.importances_mean
    order = np.argsort(importances)[::-1][:top_n]
    print(f"Top {top_n} features by importance:")
    for i in order:
        print(f"  {feature_cols[i]:24s} {importances[i]:.4f}")


def plot_equity_curve(stats, symbol=SYMBOL):
    if stats.get("n_trades", 0) == 0:
        return
    curve = stats["equity_curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(curve) + 1)), y=curve.values,
                              mode="lines", line=dict(color="#00e676", width=2), name="Equity"))
    fig.update_layout(
        title=f"{symbol} — Out-of-Sample Backtest Equity Curve ({stats['n_trades']} trades)",
        xaxis_title="Trade #", yaxis_title="Equity (starting at 1.0)",
        template="plotly_dark", height=350,
    )
    fig.show()


def _arrow_hover(row, kind):
    """Hover card for one arrow: when, why, and — for resolved signals — how it went."""
    parts = [f"<b>{kind}</b>  {row.name:%a %b %d, %H:%M}",
             f"price {row['Close']:.2f}",
             f"score {row['FinalScore']:+.1f} (rule {row['Score']:+.0f}, ML {row['Conviction']:+.2f})",
             f"ADX {row['ADX']:.0f}"]
    outcome = row.get("Outcome", "")
    if outcome == "win":
        parts.append(f"<b>RIGHT</b>: {row['OutcomeReturn']:+.2f}% ({row['ExitReason']})")
    elif outcome == "loss":
        parts.append(f"<b>WRONG</b>: {row['OutcomeReturn']:+.2f}% ({row['ExitReason']})")
    elif outcome == "pending":
        parts.append("<i>not resolved yet</i>")
    reasons = row.get("Reasons") or []
    if len(reasons):
        parts.append("— " + "<br>— ".join(reasons))
    return "<br>".join(parts)


def plot_signals_chart(df, symbol=SYMBOL, mark_outcomes=True):
    """Interactive signal chart. Drag to pan, scroll to zoom, use the range buttons
    or the slider underneath to jump around, and hover any arrow to see why it fired
    and how it turned out."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=symbol, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    has_outcomes = mark_outcomes and "Outcome" in df.columns

    def add_arrows(rows, kind, symbol_shape, base_color, y_values):
        if not len(rows):
            return
        if has_outcomes:
            # Outline resolved signals by what actually happened, so a month of
            # history reads as an audit trail rather than just a pile of arrows.
            line_colors = ["#ffffff" if o == "win" else "#000000" if o == "loss" else "#888888"
                           for o in rows["Outcome"]]
            line_widths = [2 if o in ("win", "loss") else 1 for o in rows["Outcome"]]
        else:
            line_colors, line_widths = "black", 1
        fig.add_trace(go.Scatter(
            x=rows.index, y=y_values, mode="markers", name=kind,
            marker=dict(symbol=symbol_shape, size=14, color=base_color,
                         line=dict(width=line_widths, color=line_colors)),
            text=[_arrow_hover(r, kind) for _, r in rows.iterrows()],
            hoverinfo="text",
        ))

    buys = df[df["BuyArrow"]]
    sells = df[df["SellArrow"]]
    add_arrows(buys, "BUY", "triangle-up", "#00e676", buys["Low"] * 0.9985)
    add_arrows(sells, "SELL", "triangle-down", "#ff1744", sells["High"] * 1.0015)

    n_days = df.index.normalize().nunique()
    title = f"{symbol} — 5-Min Signals, last {n_days} trading days"
    if has_outcomes:
        resolved = df[df["Outcome"].isin(["win", "loss"])]
        if len(resolved):
            hit = (resolved["Outcome"] == "win").mean() * 100
            title += f"  |  {len(resolved)} resolved signals, {hit:.0f}% went the right way"
        title += "<br><sub>white outline = worked, black = didn't, grey = too recent to tell</sub>"

    fig.update_layout(
        title=title,
        template="plotly_dark", height=750,
        hovermode="closest",
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=110, b=40),
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.06),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=3, label="3D", step="day", stepmode="backward"),
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=14, label="2W", step="day", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#222", activecolor="#00e676", font=dict(color="#eee"), y=1.06,
        ),
        # Collapse nights/weekends — without this, a month of 5-minute bars is mostly
        # empty space and the candles are squeezed into unreadable slivers.
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[16, 9.5], pattern="hour"),
        ],
    )
    fig.update_yaxes(fixedrange=False)
    fig.show(config={"scrollZoom": True})


def run_pipeline(symbol=SYMBOL, interval=INTERVAL, period=PERIOD, plot=True, verbose=True,
                  run_walk_forward=True, ml_weight=ML_WEIGHT, tune=True):
    raw = fetch_data(symbol, interval, period)
    raw = add_indicators(raw)
    raw = add_session_guard(raw)
    full = compute_signal_history(raw)
    full = add_forward_labels(full)

    best_model_fn, best_name = make_random_forest, "RandomForest"
    if run_walk_forward:
        if verbose:
            print("=" * 64)
            print("Walk-forward validation across time-ordered folds (RandomForest vs HistGB)...")
        wf = walk_forward_evaluate(full, ml_weight=ml_weight)
        best_score = -1
        for name, folds in wf.items():
            agg = summarize_folds(folds)
            if verbose:
                if agg.get("n_trades", 0):
                    print(f"  {name}: {agg['n_trades']} trades across folds, "
                          f"aggregate win rate {agg['win_rate_pct']:.1f}%")
                else:
                    print(f"  {name}: no trades triggered across folds")
            if agg.get("n_trades", 0) >= 20 and agg.get("win_rate_pct", -1) > best_score:
                best_score = agg["win_rate_pct"]
                best_model_fn, best_name = MODEL_CANDIDATES[name], name
        if verbose:
            print(f"  -> using {best_name} for the live model"
                  + (" (highest walk-forward win rate)" if best_score >= 0 else " (default; not enough walk-forward trades to compare)"))

    train_idx, test_idx = chronological_split(full, FEATURE_COLS)
    model = train_ml_model(full, train_idx, best_model_fn)
    full = add_ml_predictions(full, model)
    full = combine_rule_and_ml(full, ml_weight)
    full = add_arrow_markers(full)
    full = add_signal_outcomes(full)

    if verbose:
        print("=" * 64)
        print(f"{symbol} | fetched {len(full)} bars ({period} @ {interval})")
        print(f"Train period: {train_idx.min()}  ->  {train_idx.max()}  ({len(train_idx)} bars)")
        print(f"Test period:  {test_idx.min()}  ->  {test_idx.max()}  ({len(test_idx)} bars, out-of-sample)")
        print("=" * 64)
        print(f"Final model ({best_name}) performance on the held-out test period:")
        y_test = full.loc[test_idx, "Label"]
        X_test = prep_features(full.loc[test_idx])
        mask = y_test.notna()
        print(classification_report(y_test[mask], model.predict(X_test[mask].values),
                                     target_names=["down", "flat", "up"], zero_division=0))
        print_feature_importance(model, full, train_idx)

        trades = backtest_signals(full, test_idx)
        stats = summarize_backtest(trades)
        print_backtest_report(trades, stats, label=f"Final holdout backtest ({best_name})")
        if plot:
            plot_equity_curve(stats, symbol)
        if tune:
            tune_filters(full, test_idx, ml_weight)

    chart_df = last_n_trading_days(full, CHART_DAYS)
    print_recent_signals(chart_df)
    print_current_signal(full, symbol)

    if plot:
        plot_signals_chart(chart_df, symbol)

    return full, model


def last_n_trading_days(df, n_days=CHART_DAYS):
    """Slice the last `n_days` distinct trading sessions (not calendar days, so
    weekends and holidays don't eat into the window)."""
    sessions = sorted(set(df.index.normalize()))
    if len(sessions) > n_days:
        cutoff = sessions[-n_days]
        return df[df.index.normalize() >= cutoff]
    return df


def print_recent_signals(df, max_rows=15):
    """Prints the most recent signals with how each one resolved, so the numbers in
    the console line up with the arrows on the chart."""
    arrows = df[df["BuyArrow"] | df["SellArrow"]]
    if arrows.empty:
        print("No signals fired in the charted window.")
        return
    resolved = arrows[arrows["Outcome"].isin(["win", "loss"])]
    print("=" * 78)
    header = f"Signals in charted window: {len(arrows)}"
    if len(resolved):
        hit = (resolved["Outcome"] == "win").mean() * 100
        avg = resolved["OutcomeReturn"].mean()
        header += f"  |  {len(resolved)} resolved, {hit:.1f}% right, avg {avg:+.3f}%/signal"
    print(header)
    print("-" * 78)
    print(f"{'time':<18}{'signal':<8}{'price':>9}{'score':>8}{'ADX':>6}{'outcome':>10}{'return':>10}")
    for ts, row in arrows.tail(max_rows).iterrows():
        kind = "BUY" if row["BuyArrow"] else "SELL"
        ret = "" if pd.isna(row["OutcomeReturn"]) else f"{row['OutcomeReturn']:+.2f}%"
        print(f"{ts:%Y-%m-%d %H:%M}  {kind:<8}{row['Close']:>9.2f}{row['FinalScore']:>8.1f}"
              f"{row['ADX']:>6.0f}{row['Outcome']:>10}{ret:>10}")
    print("=" * 78)


def print_current_signal(full, symbol=SYMBOL):
    last = full.iloc[-1]
    print(f"{symbol} | {full.index[-1]} | Last Close: {last['Close']:.2f}")
    state = last["FinalSignal"]
    reason = veto_reason(last)
    if reason:
        state += f"  (would be {classify_raw(last['FinalScore'])}, vetoed: {reason})"
    print(f"SIGNAL: {state}  (score {last['FinalScore']:+.1f} = rule {last['Score']:+.0f} + ML {last['Conviction']:+.2f}, "
          f"P(up)={last['P_up']:.2f}, P(down)={last['P_down']:.2f})")
    if last["Reasons"]:
        print("Rule-engine reasons:")
        for r in last["Reasons"]:
            print("  -", r)


full_history, ml_model = run_pipeline()


import time
from IPython.display import clear_output

def refresh_and_chart(model, symbol=SYMBOL, interval=INTERVAL, period=PERIOD,
                       ml_weight=ML_WEIGHT, plot=True):
    """Cheap refresh that reuses an already-trained model instead of retraining —
    use this for frequent intraday checks; it skips the walk-forward validation
    and backtest, which don't change meaningfully every 5 minutes anyway."""
    raw = fetch_data(symbol, interval, period)
    raw = add_indicators(raw)
    raw = add_session_guard(raw)
    full = compute_signal_history(raw)
    full = add_ml_predictions(full, model)
    full = combine_rule_and_ml(full, ml_weight)
    full = add_arrow_markers(full)
    full = add_signal_outcomes(full)

    chart_df = last_n_trading_days(full, CHART_DAYS)
    print_recent_signals(chart_df)
    print_current_signal(full, symbol)
    if plot:
        plot_signals_chart(chart_df, symbol)
    return full


def run_live(symbol=SYMBOL, interval=INTERVAL, period=PERIOD, refresh_seconds=300,
             iterations=100, retrain_every=12):
    """Fully retrains (walk-forward validation + backtest) every `retrain_every`
    iterations (default 12 x 5min = hourly) and does a cheap refresh with the
    existing model in between — retraining on nearly-identical data every 5
    minutes adds cost without adding much. Keep the Colab tab open; stop any
    time with the cell's stop button."""
    model = None
    for i in range(iterations):
        clear_output(wait=True)
        try:
            if model is None or i % retrain_every == 0:
                print(f"[iteration {i}] full retrain + walk-forward validation...")
                _, model = run_pipeline(symbol, interval, period, plot=True, verbose=True)
            else:
                print(f"[iteration {i}] quick refresh using existing model...")
                refresh_and_chart(model, symbol, interval, period, plot=True)
        except Exception as e:
            print("Error:", e)
        time.sleep(refresh_seconds)

# Uncomment to poll every 5 minutes for the rest of the session:
# run_live(refresh_seconds=300, iterations=100)
