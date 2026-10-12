# Run this once in your terminal or Colab cell (with the ! prefix) before running the script:
# pip install -U yfinance plotly pandas numpy
#
# S&P 500 DAILY LONG / SHORT SIGNALS
# ----------------------------------
# Buy weakness in an uptrend, short strength in a downtrend.
#
#   Trend  UP   : close > 200-day avg and not in a correction
#          DOWN : close < 200-day avg, or a correction (close < 50-day avg and the 50-day avg falling)
#   LONG   in an UP trend, after a pullback:  RSI(2) low   OR  closed near the day's low (IBS)
#   SHORT  in a DOWN trend, after a bounce:   RSI(2) high  OR  closed near the day's high (IBS)
#   SHORT  in an UP trend too, at an overbought top: RSI(2) very high AND closed near the high
#   An opposite signal closes the open trade and reverses it.
#
#   SELF-CHECK (cuts wrong signals): before a signal is shown, the script looks up how that exact
#   signal type did over the previous 3 years. Types that were losing (win rate < 50% or profit
#   factor < 1.0) are blocked — but if every type is losing, the best one stays on, so you always
#   get signals. Only past results are used at every point — no hindsight.
#   On the chart: SOLID arrows = signals to trade. HOLLOW arrows = weak signals (their type has been
#   losing) — shown so you can see every LONG and SHORT setup, but not traded.
#
#   MODE sets how many signals you get (the track record shows how each mode did):
#     "quality"  : RSI(2) < 10 / > 90 only            — a handful per month at most
#     "balanced" : RSI(2) < 20 / > 80, IBS < .2 / > .8 — about weekly
#     "active"   : RSI(2) < 35 / > 65, IBS < .35 / > .65 — about weekly, 1-5 day swings
#     "daily"    : RSI(2) < 40 / > 60, IBS < .5 / > .5  — most days; in at the open, out at the close
#
#   Entry : next day's open (the signal is confirmed at the close).
#   Exit  : first close above the previous day's high or the 5-day average (LONG; mirror for SHORT),
#           or the stop (2 x ATR), or after the mode's max days — whichever comes first.
#   Only one trade at a time, so the chart shows one arrow per trade.
#
#   python sp500_daily_long_short.py              # today's status + chart
#   python sp500_daily_long_short.py --live       # re-check every 15 min during market hours
#   python sp500_daily_long_short.py --long-only  # no SHORT signals
#   python sp500_daily_long_short.py --mode balanced
#   python sp500_daily_long_short.py --symbol SPY
#
# The chart opens by itself: under the cell in Colab/Jupyter, in your browser from a terminal.

import argparse
import os
import time
import warnings
import webbrowser
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ---- Configuration ----
SYMBOL = "^GSPC"            # S&P 500 index. "SPY" = ETF, "ES=F" = futures
START = "2000-01-01"
CSV_PATH = None             # optional: daily CSV (Date,Open,High,Low,Close) instead of downloading

MODE = "active"             # "quality" (fewest) · "balanced" · "active" (best balance) · "daily" (most, more wrong)
MODES = {   # RSI(2) long/short, IBS long/short (None = off), max holding days,
            # top = (RSI(2), IBS) needed to SHORT an overbought top inside an uptrend (None = off)
    "quality":  dict(rsi=(10, 90), ibs=None,         max_days=10, top=None),
    "balanced": dict(rsi=(20, 80), ibs=(0.2, 0.8),   max_days=5,  top=(95, 0.8)),
    "active":   dict(rsi=(35, 65), ibs=(0.35, 0.65), max_days=5,  top=(90, 0.75)),
    "daily":    dict(rsi=(40, 60), ibs=(0.5, 0.5),   max_days=1,  top=(85, 0.7)),   # day trades
}
EXIT_MA = 5                 # exit on a close back across the 5-day average (or past yesterday's high/low)
STOP_ATR = 2.0              # protective stop distance in daily ATRs
ALLOW_SHORTS = True
SELF_CHECK = True           # block signal types that were losing over the previous LOOKBACK days
CHECK_LOOKBACK = 756        # ~3 years of trading days
CHECK_MIN_TRADES = 15       # need at least this many past signals of the type to judge it
CHECK_MIN_WIN = 0.50        # past win rate needed
CHECK_MIN_PF = 1.0          # past profit factor needed (1.0 = broke even)
SHOW_WEAK = True            # also draw blocked (weak) signals as hollow arrows
COST_PCT = 0.0002           # commission + slippage per side

CHART_DAYS = 126            # ~6 months on the chart
HTML_PATH = "sp500_signals.html"
LIVE_EVERY_MIN = 15
NY = "America/New_York"
LONG_COLOR, SHORT_COLOR = "#0f8a5f", "#d95f02"


# ---------------------------------------------------------------- data
def _clean(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).capitalize() for c in df.columns]
    if not {"Open", "High", "Low", "Close"} <= set(df.columns):
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna().sort_index()
    idx = pd.to_datetime(df.index)
    df.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return df[~df.index.duplicated(keep="last")]


def _from_yahoo(symbol, start, tries=3):
    import yfinance as yf
    for k in range(tries):
        try:
            df = _clean(yf.download(symbol, start=start, interval="1d", auto_adjust=True,
                                    progress=False, threads=False))
            if not df.empty:
                return df
        except Exception as e:                       # rate limit, network, Yahoo format change
            print(f"  Yahoo attempt {k + 1} failed: {type(e).__name__}: {e}")
        time.sleep(2 * (k + 1))
    return pd.DataFrame()


def _from_stooq(symbol, start):
    code = {"^GSPC": "^spx", "^SPX": "^spx", "ES=F": "es.f"}.get(symbol.upper(), symbol.lower() + ".us")
    try:
        df = _clean(pd.read_csv(f"https://stooq.com/q/d/l/?s={code}&i=d", parse_dates=["Date"], index_col="Date"))
        return df[df.index >= pd.Timestamp(start)]
    except Exception as e:
        print(f"  Stooq failed: {type(e).__name__}: {e}")
        return pd.DataFrame()


def fetch_daily(symbol=SYMBOL, start=START, csv_path=CSV_PATH):
    if csv_path:
        df = _clean(pd.read_csv(csv_path, parse_dates=[0], index_col=0))
    else:
        df = _from_yahoo(symbol, start)
        if df.empty:
            print("  Yahoo Finance returned no data — trying the Stooq backup…")
            df = _from_stooq(symbol, start)
    if len(df) < 250:
        raise RuntimeError(
            f"Could not download enough daily data for {symbol} ({len(df)} rows).\n"
            "  Fixes: 1) run  !pip install -U yfinance  and restart the runtime,\n"
            "         2) wait a few minutes (Yahoo rate-limits Colab) and run again,\n"
            "         3) or download a daily CSV (Date,Open,High,Low,Close) and set CSV_PATH.")
    return df


def fetch_today(symbol=SYMBOL):
    """Today's still-forming daily candle, built from 5-minute bars. Empty if unavailable."""
    try:
        import yfinance as yf
        df = yf.download(symbol, period="1d", interval="5m", auto_adjust=True, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = df.index.tz_convert(NY) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(NY)
    df = df.set_axis(idx).between_time("09:30", "15:59").dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame({"Open": [df["Open"].iloc[0]], "High": [df["High"].max()],
                         "Low": [df["Low"].min()], "Close": [df["Close"].iloc[-1]]},
                        index=[pd.Timestamp(df.index[-1].date())])


def market_open_now():
    now = pd.Timestamp.now(tz=NY)
    return now.weekday() < 5 and (now.hour, now.minute) >= (9, 30) and now.hour < 16


# ---------------------------------------------------------------- signals
def add_indicators(df):
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    df["ma_exit"] = c.rolling(EXIT_MA).mean()
    df["ma50"] = c.rolling(50).mean()
    df["ma200"] = c.rolling(200).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    df["rsi2"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    rng = (h - l).replace(0, np.nan)
    df["ibs"] = ((c - l) / rng).fillna(0.5)          # 0 = closed at the day's low, 1 = at the high

    correction = (c < df["ma50"]) & (df["ma50"] < df["ma50"].shift(10))   # pullback inside a bull market
    downtrend = (c < df["ma200"]) | correction
    uptrend = ~downtrend

    m = MODES[MODE]
    lo_rsi, hi_rsi = m["rsi"]
    no = pd.Series(False, index=df.index)
    ibs_weak = ibs_strong = no
    if m["ibs"]:
        lo_ibs, hi_ibs = m["ibs"]
        ibs_weak = (df["ibs"] < lo_ibs) & (c < c.shift())
        ibs_strong = (df["ibs"] > hi_ibs) & (c > c.shift())
    top = no
    if m["top"]:                                      # overbought top inside an uptrend
        top_rsi, top_ibs = m["top"]
        top = uptrend & (df["rsi2"] > top_rsi) & (df["ibs"] > top_ibs)
    sh = ALLOW_SHORTS
    # signal types, in priority order (the first one that matches names the signal)
    kinds = [("SHORT top",    top & sh),
             ("SHORT bounce", downtrend & (df["rsi2"] > hi_rsi) & sh),
             ("SHORT ibs",    downtrend & ibs_strong & sh),
             ("LONG dip",     uptrend & (df["rsi2"] < lo_rsi)),
             ("LONG ibs",     uptrend & ibs_weak)]
    df["kind"] = np.select([k[1] for k in kinds], [k[0] for k in kinds], "")
    df.loc[df["ma200"].isna(), "kind"] = ""
    df["setup"] = np.where(df["kind"].str.startswith("SHORT"), -1, np.where(df["kind"] != "", 1, 0))
    return df


def _play(arr, setup, i, d, complete):
    """Trade a signal from bar i: enter at the next open, return (entry, stop, exit_i, exit_px, reason).
    exit_i is None while the trade is still open at the last finished bar."""
    o, h, l, c, ma, atr = arr
    max_days = MODES[MODE]["max_days"]
    e = i + 1
    entry, stop = o[e], o[e] - d * STOP_ATR * atr[i]
    for j in range(e, complete):
        if (d == 1 and l[j] <= stop) or (d == -1 and h[j] >= stop):
            return entry, stop, j, (min(o[j], stop) if d == 1 else max(o[j], stop)), "stop"
        if (d == 1 and (c[j] > ma[j] or c[j] > h[j - 1])) or (d == -1 and (c[j] < ma[j] or c[j] < l[j - 1])):
            return entry, stop, j, c[j], "target"
        if setup[j] == -d:                           # opposite signal: close here and reverse
            return entry, stop, j, c[j], "reverse"
        if j - e + 1 >= max_days:
            return entry, stop, j, c[j], "time"
    return entry, stop, None, None, "open"


def _arrays(df):
    return tuple(df[k].to_numpy() for k in ("Open", "High", "Low", "Close", "ma_exit", "atr"))


def _grade(exits, rets, i):
    """(passes, profit factor) for a type's signals that finished in the lookback before bar i,
    or None if there are too few to judge."""
    past = rets[(exits >= i - CHECK_LOOKBACK) & (exits < i)]
    if len(past) < CHECK_MIN_TRADES:
        return None
    loss = -past[past <= 0].sum()
    pf = past[past > 0].sum() / loss if loss > 0 else np.inf
    return ((past > 0).mean() >= CHECK_MIN_WIN and pf >= CHECK_MIN_PF), pf


def self_check(df, complete):
    """For every signal day, look only at signals of the same type that had already FINISHED within
    the lookback window, and allow the new one only if that type was winning. Adds `allowed`."""
    arr, setup, kind = _arrays(df), df["setup"].to_numpy(), df["kind"].to_numpy()
    idx = np.flatnonzero(setup != 0)
    # every past signal traded on its own (overlaps allowed) — the type's report card
    done = {}
    for i in idx[idx < complete - 1]:
        d = int(setup[i])
        entry, _, ex, px, why = _play(arr, setup, i, d, complete)
        if ex is not None:
            done.setdefault(kind[i], []).append((ex, d * (px - entry) / entry - 2 * COST_PCT))
    done = {k: (np.array([x for x, _ in v]), np.array([r for _, r in v])) for k, v in done.items()}
    allowed = np.zeros(len(df), dtype=bool)
    for i in idx:
        if not SELF_CHECK:
            allowed[i] = True
            continue
        grades = {k: _grade(ex, r, i) for k, (ex, r) in done.items()}
        g = grades.get(kind[i])
        if g is None:                                # not enough history yet: give it a chance
            allowed[i] = True
        elif g[0]:                                   # this type is winning
            allowed[i] = True
        elif not any(x and x[0] for x in grades.values()):
            # every judged type is losing: keep only the best one (highest profit factor) on
            best = max((k for k, x in grades.items() if x), key=lambda k: grades[k][1])
            allowed[i] = kind[i] == best
    df = df.copy()
    df["allowed"] = allowed
    return df, done


def kind_report(df, done):
    """Each signal type's record over the last CHECK_LOOKBACK days and whether it is switched on now."""
    n = len(df)
    grades = {k: _grade(ex, r, n) for k, (ex, r) in done.items()}
    any_pass = any(g and g[0] for g in grades.values())
    judged = [k for k, g in grades.items() if g]
    best = max(judged, key=lambda k: grades[k][1]) if judged else None
    lines = []
    for k in sorted({k for k in df["kind"] if k}):
        exits, rets = done.get(k, (np.array([]), np.array([])))
        past = rets[(exits >= n - CHECK_LOOKBACK) & (exits < n)]
        g = grades.get(k)
        if g is None:
            state = "ON (not enough history to judge)"
        elif g[0]:
            state = "ON"
        elif not any_pass and k == best:
            state = "ON (best of a weak bunch)"
        else:
            state = "blocked (was losing)"
        if len(past) == 0:
            lines.append(f"    {k:<13} no finished signals yet   -> {state}")
            continue
        loss = -past[past <= 0].sum()
        pf = past[past > 0].sum() / loss if loss > 0 else np.inf
        lines.append(f"    {k:<13} {len(past):4d} signals   win {100 * (past > 0).mean():5.1f}%   "
                     f"profit factor {pf:4.2f}   -> {state if SELF_CHECK else 'ON'}")
    return lines


def simulate(df, complete):
    """One trade at a time, only signals the self-check allows. `complete` = number of finished bars
    (excludes a live candle). Returns (trades, status): status says what to do now (new/open/flat)."""
    arr = _arrays(df)
    o, h, l, c, ma, atr = arr
    setup = np.where(df["allowed"].to_numpy(), df["setup"].to_numpy(), 0)
    trades, i, status = [], 0, None
    while i < complete:
        if setup[i] == 0:
            i += 1
            continue
        d = int(setup[i])
        if i == complete - 1:                       # signal at the last finished close
            status = {"kind": "new", "dir": d, "date": df.index[i], "close": c[i],
                      "stop": c[i] - d * STOP_ATR * atr[i], "type": df["kind"].iat[i]}
            break
        e = i + 1
        entry, stop, exit_i, exit_px, reason = _play(arr, setup, i, d, complete)
        t = {"dir": d, "type": df["kind"].iat[i], "signal_date": df.index[i], "entry_date": df.index[e],
             "entry": entry, "stop": stop}
        if exit_i is None:                          # still open
            t.update(exit_date=None, exit=c[complete - 1], reason="open", days=complete - e,
                     ret=100 * d * (c[complete - 1] - entry) / entry)
            trades.append(t)
            status = {"kind": "open", **t, "ma": ma[complete - 1]}
            break
        t.update(exit_date=df.index[exit_i], exit=exit_px, reason=reason, days=exit_i - e + 1,
                 ret=100 * (d * (exit_px - entry) / entry - 2 * COST_PCT))
        trades.append(t)
        i = exit_i                                  # a new signal can fire on the exit day's close
    return pd.DataFrame(trades), status or {"kind": "flat"}


# ---------------------------------------------------------------- report
def summary(trades):
    done = trades[trades["reason"] != "open"] if not trades.empty else trades
    lines = []
    for name, d in (("LONG ", 1), ("SHORT", -1)):
        g = done[done["dir"] == d] if not done.empty else done
        if g.empty:
            continue
        r = g["ret"]
        pf = r[r > 0].sum() / -r[r <= 0].sum() if (r <= 0).any() and r[r <= 0].sum() < 0 else np.inf
        lines.append(f"  {name} {len(r):4d} trades   win {100 * (r > 0).mean():5.1f}%   "
                     f"avg {r.mean():+.2f}%   profit factor {pf:4.2f}   avg hold {g['days'].mean():.1f} days")
    return lines or ["  no completed trades yet"]


def print_report(df, trades, status, live, symbol):
    side = {1: "LONG", -1: "SHORT"}
    now = pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d %H:%M ET")
    bar = "=" * 72
    print(f"\n{bar}\n {symbol}  daily signals ({MODE} mode)   {now}\n{bar}")
    k = status["kind"]
    if k == "new":
        d = status["dir"]
        when = "today's open" if live is not None else "the next open"
        print(f"\n  >>> NEW {side[d]} SIGNAL [{status['type']}] (close of {status['date'].date()}) — "
              f"{'buy' if d == 1 else 'sell short'} at {when} (~{status['close']:,.2f})")
        if MODES[MODE]["max_days"] == 1:
            print(f"      stop ~{status['stop']:,.2f}   exit at the same day's close (day trade)")
        else:
            print(f"      stop ~{status['stop']:,.2f}   exit on the first close {'above' if d == 1 else 'below'} "
                  f"the day before's {'high' if d == 1 else 'low'} or the {EXIT_MA}-day average, "
                  f"max {MODES[MODE]['max_days']} days")
    elif k == "open":
        d = status["dir"]
        print(f"\n  >>> IN {side[d]} since {status['entry_date'].date()} @ {status['entry']:,.2f}   "
              f"P/L {status['ret']:+.2f}%   day {status['days']} of {MODES[MODE]['max_days']}")
        prev = df["High"].iat[-2] if d == 1 else df["Low"].iat[-2]
        lvl = min(status["ma"], prev) if d == 1 else max(status["ma"], prev)
        print(f"      stop {status['stop']:,.2f}   exit on a close {'above' if d == 1 else 'below'} "
              f"{lvl:,.2f}")
    else:
        print("\n  >>> NO SIGNAL — stay flat.")
    n_done = len(df) - (1 if live is not None else 0)
    last = df.iloc[n_done - 1]
    if k != "new" and last.setup != 0 and not last.allowed:
        print(f"      (weak {side[int(last.setup)]} setup at the last close [{last.kind}] — skipped: "
              f"this type has been losing; hollow arrow on the chart)")

    if live is not None:
        r = df.iloc[-1]
        print(f"\n  LIVE today: price {r.Close:,.2f}")
        if k == "open":
            d = status["dir"]
            if (r.Low <= status["stop"]) if d == 1 else (r.High >= status["stop"]):
                print("      stop hit today — the trade is closed.")
            elif ((r.Close > r.ma_exit or r.Close > df["High"].iat[-2]) if d == 1
                  else (r.Close < r.ma_exit or r.Close < df["Low"].iat[-2])):
                print("      exit condition met — plan to close at today's close.")
            else:
                print("      hold — exit condition not met yet.")
        elif r.setup != 0 and r.allowed:
            print(f"      {side[int(r.setup)]} signal FORMING [{r.kind}] — valid only if it still holds at the close.")
        elif r.setup != 0:
            print(f"      weak {side[int(r.setup)]} setup forming [{r.kind}] — this type has been losing, not traded.")
        else:
            print("      no signal forming.")

    months = max((df.index[-1] - df.index[0]).days / 30.4, 1)
    print(f"\n  Track record, mode '{MODE}', {df.index[0].date()} -> {df.index[-1].date()} "
          f"(enter next open, costs included) — about {len(trades) / months:.1f} signals per month:")
    for line in summary(trades):
        print(line)
    if SELF_CHECK and "done" in df.attrs:
        print(f"\n  Signal types over the last {CHECK_LOOKBACK} days (need win >= {CHECK_MIN_WIN:.0%} "
              f"and profit factor >= {CHECK_MIN_PF}):")
        for line in kind_report(df, df.attrs["done"]):
            print(line)
    print(bar)


# ---------------------------------------------------------------- chart
def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:
        return False


def plot_chart(df, trades, status, live, symbol, html_path=HTML_PATH, show=True):
    import plotly.graph_objects as go

    v = df.iloc[-CHART_DAYS:]
    fig = go.Figure(go.Candlestick(
        x=v.index, open=v.Open, high=v.High, low=v.Low, close=v.Close, name=symbol, showlegend=False,
        increasing=dict(line=dict(color="#9aa0a6", width=1), fillcolor="#ffffff"),
        decreasing=dict(line=dict(color="#5f6368", width=1), fillcolor="#5f6368")))

    # one arrow per trade, plus a new / forming signal
    pts = []
    if not trades.empty:
        for _, t in trades[trades["signal_date"] >= v.index[0]].iterrows():
            res = "still open" if t.reason == "open" else f"exit {t.exit:,.2f} on {t.exit_date.date()} ({t.reason})"
            pts.append((t.signal_date, t.dir, "",
                        f"{t.type}<br>entry {t.entry:,.2f} on {t.entry_date.date()}<br>{res}<br>result {t.ret:+.2f}%"))
    if status["kind"] == "new":
        pts.append((status["date"], status["dir"], " (new)", "enter at the next open"))
    if live is not None and df["setup"].iat[-1] != 0 and df["allowed"].iat[-1] and status["kind"] != "open":
        pts.append((df.index[-1], int(df["setup"].iat[-1]), "?", "forming — confirms at the close"))

    for d, name, sym, color in ((1, "LONG", "triangle-up", LONG_COLOR), (-1, "SHORT", "triangle-down", SHORT_COLOR)):
        p = [q for q in pts if q[1] == d]
        if not p:
            continue
        x = [q[0] for q in p]
        y = [v.at[t, "Low"] - 0.6 * v.at[t, "atr"] if d == 1 else v.at[t, "High"] + 0.6 * v.at[t, "atr"] for t in x]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers+text", name=name,
            text=[name + q[2] if q[2] else "" for q in p],          # label only the new / forming one
            textposition="bottom center" if d == 1 else "top center",
            textfont=dict(size=12, color=color),
            marker=dict(symbol=sym, size=12, color=color, line=dict(width=1, color="white")),
            hovertemplate=[f"<b>{name}</b> signal {q[0].date()}<br>{q[3]}<extra></extra>" for q in p]))

    if SHOW_WEAK:                                    # blocked signals: hollow, first day of each streak
        n_done = len(df) - (1 if live is not None else 0)
        weak = (v["setup"] != 0) & ~v["allowed"] & (v["setup"] != v["setup"].shift())
        weak &= np.arange(len(v)) < n_done - (len(df) - len(v))
        stats = {k: _grade(ex, r, n_done) for k, (ex, r) in df.attrs.get("done", {}).items()}
        for d, name, sym, color in ((1, "LONG (weak)", "triangle-up-open", LONG_COLOR),
                                    (-1, "SHORT (weak)", "triangle-down-open", SHORT_COLOR)):
            w = v[weak & (v["setup"] == d)]
            if w.empty:
                continue
            y = w["Low"] - 0.6 * w["atr"] if d == 1 else w["High"] + 0.6 * w["atr"]
            hover = []
            for t, k in zip(w.index, w["kind"]):
                g = stats.get(k)
                why = f"profit factor {g[1]:.2f} over the last 3 years" if g else "not enough history"
                hover.append(f"<b>{k}</b> (weak) {t.date()}<br>blocked: this type has been losing<br>{why}"
                             f"<extra></extra>")
            fig.add_trace(go.Scatter(x=w.index, y=y, mode="markers", name=name, hovertemplate=hover,
                                     marker=dict(symbol=sym, size=11, color=color, line=dict(width=1.5))))

    missing = pd.bdate_range(v.index[0], v.index[-1]).difference(v.index)
    fig.update_layout(
        title=dict(text=f"<b>{symbol}</b> daily — LONG / SHORT signals", x=0.01),
        template="plotly_white", height=650, hovermode="closest",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.06, font=dict(size=13)),
        xaxis_rangeslider_visible=False, margin=dict(l=60, r=30, t=60, b=40),
        font=dict(family="Inter, Segoe UI, Arial", size=12, color="#3c4043"))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=list(missing))], showgrid=False)
    fig.update_yaxes(gridcolor="#f1f3f4", tickformat=",.0f")

    if html_path:
        fig.write_html(html_path, include_plotlyjs=True)
        print(f"Chart saved to {os.path.abspath(html_path)}")
    if show:
        if in_notebook():
            fig.show()
        elif html_path:
            webbrowser.open("file://" + os.path.abspath(html_path))
    return fig


# ---------------------------------------------------------------- run
def run_once(symbol=SYMBOL, start=START, csv_path=CSV_PATH, plot=True, show=True):
    df = fetch_daily(symbol, start, csv_path)
    live = None
    if not csv_path and market_open_now():
        today = fetch_today(symbol)
        if not today.empty and today.index[-1].date() == pd.Timestamp.now(tz=NY).date():
            live = today
            df = pd.concat([df[df.index < today.index[-1]], today])     # today's forming candle
    df = add_indicators(df)
    complete = len(df) - 1 if live is not None else len(df)
    df, done = self_check(df, complete)
    df.attrs["done"] = done
    trades, status = simulate(df, complete)
    print_report(df, trades, status, live, symbol)
    if plot:
        plot_chart(df, trades, status, live, symbol, show=show)
    return df, trades


def run_live(symbol=SYMBOL, every_min=LIVE_EVERY_MIN, **kw):
    first = True
    while True:
        if first or market_open_now():
            if in_notebook() and not first:
                from IPython.display import clear_output
                clear_output(wait=True)
            run_once(symbol, **kw)
            first = False
        else:
            print(f"{pd.Timestamp.now(tz=NY):%H:%M} ET market closed — waiting…")
        time.sleep(every_min * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="S&P 500 daily LONG/SHORT signals")
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--start", default=START)
    ap.add_argument("--csv", default=CSV_PATH, help="daily OHLC CSV instead of downloading")
    ap.add_argument("--live", action="store_true", help="re-check every --every minutes in market hours")
    ap.add_argument("--every", type=int, default=LIVE_EVERY_MIN)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--mode", default=MODE, choices=list(MODES))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="save the chart but don't open/show it")
    args, _ = ap.parse_known_args()       # parse_known_args so it also runs inside Colab/Jupyter
    if args.long_only:
        ALLOW_SHORTS = False
    MODE = args.mode
    kw = dict(start=args.start, csv_path=args.csv, plot=not args.no_plot, show=not args.no_open)
    if args.live:
        run_live(args.symbol, args.every, **kw)
    else:
        run_once(args.symbol, **kw)
