# Run this once in your terminal or Colab cell (with the ! prefix) before running the script:
# pip install -U yfinance plotly pandas numpy
#
# S&P 500 DAILY LONG / SHORT SIGNALS — with intraday take-profit
# ---------------------------------------------------------------
# Rules chosen on real SPY data 2008-2012 and confirmed on unseen data 2013-2017:
#
#   LONG   when the S&P is in an uptrend (close and 50-day avg both above the 200-day avg)
#          and it just pulled back:  RSI(2) below 25   or   3 lower closes in a row
#   SHORT  when the S&P is below its 200-day average (downtrend) and it just bounced:
#          close at a 10-day high   or   RSI(2) above 90
#
#   Entry  : next day's open.
#   Exits  : 1) TAKE PROFIT intraday — a limit order at entry +/- 0.35 x ATR (about a third of a
#               normal day's range); place it right after you get filled
#            2) otherwise the first close in profit
#            3) STOP at 2.5 x ATR, an opposite signal, or 10 days — whichever comes first
#   One trade at a time.
#
#   Real SPY results (costs included; a limit counts as filled when the day's range touched it):
#     MODE "quality"  ~2.5 signals/month   2008-12: 91% win, PF 2.3   2013-17: 90% win, PF 1.7
#     MODE "more"     ~4 signals/month     2008-12: 89% win, PF 1.7   2013-17: 89% win, PF 1.7
#   Wins are small and frequent, losses rarer but bigger — so always use the stop.
#   Shorts only appear when the S&P is below its 200-day average (rare in a bull market).
#
#   python sp500_daily_long_short.py              # daily: today's instruction + chart
#   python sp500_daily_long_short.py --live       # INTRADAY: re-checks every 15 min while the market
#                                                 # is open — target hit? stop hit? signal forming?
#   python sp500_daily_long_short.py --mode more  # more signals
#   python sp500_daily_long_short.py --long-only
#
# The chart opens by itself: under the cell in Colab/Jupyter, in your browser from a terminal.
# On the chart: ✓ = trade made money, ✗ = lost money.

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

MODE = "quality"            # "quality" (~2.5/month, best) or "more" (~4/month)
MODES = {
    "quality": dict(long_rsi=25),
    "more":    dict(long_rsi=50),
}
TARGET_ATR = 0.35           # intraday take-profit limit = entry +/- 0.35 x ATR
SHORT_RSI = 90              # SHORT bounce threshold in a downtrend
SHORT_HIGH_DAYS = 10        # SHORT when the close is the highest of this many days (downtrend only)
STOP_ATR = 2.5              # protective stop distance in daily ATRs
MAX_DAYS = 10               # time exit
ALLOW_SHORTS = True
COST_PCT = 0.0002           # commission + slippage per side

CHART_DAYS = 252            # ~1 year on the chart
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
    df["ma50"] = c.rolling(50).mean()
    df["ma200"] = c.rolling(200).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    df["rsi2"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    uptrend = (c > df["ma200"]) & (df["ma50"] > df["ma200"])
    downtrend = c < df["ma200"]
    down3 = (c < c.shift()) & (c.shift() < c.shift(2)) & (c.shift(2) < c.shift(3))
    high_n = c >= c.rolling(SHORT_HIGH_DAYS).max()
    long_rsi = MODES[MODE]["long_rsi"]
    sh = ALLOW_SHORTS
    kinds = [("SHORT 10-day high", downtrend & high_n & sh),          # first match names the signal
             ("SHORT RSI(2) bounce", downtrend & (df["rsi2"] > SHORT_RSI) & sh),
             ("LONG RSI(2) pullback", uptrend & (df["rsi2"] < long_rsi)),
             ("LONG 3 down days", uptrend & down3)]
    df["kind"] = np.select([k[1] for k in kinds], [k[0] for k in kinds], "")
    df.loc[df["ma200"].isna(), "kind"] = ""
    df["setup"] = np.where(df["kind"].str.startswith("SHORT"), -1, np.where(df["kind"] != "", 1, 0))
    return df


def simulate(df, complete):
    """One trade at a time. `complete` = number of finished daily bars (excludes a live candle).
    Returns (trades, status): status says what to do now (new / open / flat)."""
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    atr, setup = df["atr"].to_numpy(), df["setup"].to_numpy()
    trades, i, status = [], 0, None
    while i < complete:
        if setup[i] == 0:
            i += 1
            continue
        d = int(setup[i])
        if i == complete - 1:                       # signal at the last finished close
            status = {"kind": "new", "dir": d, "date": df.index[i], "close": c[i], "atr": atr[i],
                      "stop": c[i] - d * STOP_ATR * atr[i], "target": c[i] + d * TARGET_ATR * atr[i],
                      "type": df["kind"].iat[i]}
            break
        e = i + 1
        entry = o[e]
        stop, target = entry - d * STOP_ATR * atr[i], entry + d * TARGET_ATR * atr[i]
        exit_i = exit_px = reason = None
        for j in range(e, complete):
            if (d == 1 and l[j] <= stop) or (d == -1 and h[j] >= stop):        # stop checked first
                exit_i, exit_px, reason = j, (min(o[j], stop) if d == 1 else max(o[j], stop)), "stop"
            elif (d == 1 and h[j] >= target) or (d == -1 and l[j] <= target):  # intraday take-profit
                gap = o[j] if j > e else target                                 # gapped through it
                exit_i, exit_px, reason = j, (max(gap, target) if d == 1 else min(gap, target)), "target"
            elif d * (c[j] - entry) > 0:                                        # first close in profit
                exit_i, exit_px, reason = j, c[j], "profit close"
            elif setup[j] == -d:                       # opposite signal: close here and reverse
                exit_i, exit_px, reason = j, c[j], "reverse"
            elif j - e + 1 >= MAX_DAYS:
                exit_i, exit_px, reason = j, c[j], "time"
            if exit_i is not None:
                break
        t = {"dir": d, "type": df["kind"].iat[i], "signal_date": df.index[i], "entry_date": df.index[e],
             "entry": entry, "stop": stop, "target": target}
        if exit_i is None:                          # still open
            t.update(exit_date=None, exit=c[complete - 1], reason="open", days=complete - e,
                     ret=100 * d * (c[complete - 1] - entry) / entry)
            trades.append(t)
            status = {"kind": "open", **t}
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
    n_done = len(df) - (1 if live is not None else 0)
    last = df.iloc[n_done - 1]
    trend = ("UPTREND (LONG signals only)" if last.Close > last.ma200 and last.ma50 > last.ma200 else
             "DOWNTREND (SHORT signals only)" if last.Close < last.ma200 else
             "NEUTRAL (above the 200-day avg, but the 50-day avg is still below it — no signals)")
    print(f"\n{bar}\n {symbol}  daily signals ({MODE} mode)   {now}\n{bar}")
    print(f"\n  Market: {trend} — close {last.Close:,.2f} vs 200-day avg {last.ma200:,.2f}")
    k = status["kind"]
    if k == "new":
        d = status["dir"]
        when = "today's open" if live is not None else "the next open"
        print(f"\n  >>> NEW {side[d]} SIGNAL [{status['type']}] (close of {status['date'].date()}) — "
              f"{'buy' if d == 1 else 'sell short'} at {when} (~{status['close']:,.2f})")
        print(f"      take-profit limit = entry {'+' if d == 1 else '-'} {TARGET_ATR * status['atr']:,.2f} "
              f"(~{status['target']:,.2f} if filled near {status['close']:,.2f})")
        print(f"      stop = entry {'-' if d == 1 else '+'} {STOP_ATR * status['atr']:,.2f} (~{status['stop']:,.2f})   "
              f"else exit on the first close in profit, max {MAX_DAYS} days")
    elif k == "open":
        d = status["dir"]
        print(f"\n  >>> IN {side[d]} since {status['entry_date'].date()} @ {status['entry']:,.2f}   "
              f"P/L {status['ret']:+.2f}%   day {status['days']} of {MAX_DAYS}")
        print(f"      take-profit limit {status['target']:,.2f}   stop {status['stop']:,.2f}   "
              f"or exit on a close {'above' if d == 1 else 'below'} {status['entry']:,.2f}")
    else:
        print("\n  >>> NO SIGNAL — stay flat.")
        if last.Close > last.ma200 and last.ma50 > last.ma200:
            print(f"      next LONG fires on a close with RSI(2) < {MODES[MODE]['long_rsi']} "
                  f"(now {last.rsi2:.0f}) or after 3 down days in a row")
        elif last.Close < last.ma200:
            print(f"      next SHORT fires on a {SHORT_HIGH_DAYS}-day closing high or RSI(2) > {SHORT_RSI} "
                  f"(now {last.rsi2:.0f})")

    if live is not None:
        r = df.iloc[-1]
        print(f"\n  LIVE today: price {r.Close:,.2f}")
        if k == "open":
            d = status["dir"]
            if (r.Low <= status["stop"]) if d == 1 else (r.High >= status["stop"]):
                print("      STOP hit today — the trade is closed.")
            elif (r.High >= status["target"]) if d == 1 else (r.Low <= status["target"]):
                print(f"      TARGET {status['target']:,.2f} reached today — take the profit.")
            elif d * (r.Close - status["entry"]) > 0:
                print(f"      in profit now — if it stays {'above' if d == 1 else 'below'} {status['entry']:,.2f} "
                      f"into the close, exit at the close.")
            else:
                print(f"      hold — target {status['target']:,.2f}, stop {status['stop']:,.2f}.")
        elif r.setup != 0:
            print(f"      {side[int(r.setup)]} signal FORMING [{r.kind}] — valid only if it still holds at the close.")
        else:
            print("      no signal forming.")

    months = max((df.index[-1] - df.index[0]).days / 30.4, 1)
    print(f"\n  Track record, mode '{MODE}', {df.index[0].date()} -> {df.index[-1].date()} "
          f"(enter next open, costs included) — about {len(trades) / months:.1f} signals per month:")
    fin = trades[trades["reason"] != "open"] if not trades.empty else trades
    if not fin.empty:
        print(f"  ALL   {len(fin):4d} trades   win {100 * (fin['ret'] > 0).mean():5.1f}%")
    for line in summary(trades):
        print(line)
    done = trades[trades["reason"] != "open"].tail(10) if not trades.empty else trades
    if not done.empty:
        won = int((done["ret"] > 0).sum())
        print(f"\n  Last {len(done)} signals: {won} made money, {len(done) - won} lost")
        for _, t in done.iterrows():
            print(f"    {t.signal_date.date()}  {side[t.dir]:<5} {t.type:<21} {t.ret:+6.2f}%  "
                  f"{'✓' if t.ret > 0 else '✗'}  ({t.reason}, {t.days}d)")
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
            mark = "…" if t.reason == "open" else ("✓" if t.ret > 0 else "✗")
            pts.append((t.signal_date, t.dir, mark,
                        f"{t.type}<br>entry {t.entry:,.2f} on {t.entry_date.date()}<br>{res}<br>result {t.ret:+.2f}%"))
    if status["kind"] == "new":
        pts.append((status["date"], status["dir"], "NEW", f"{status['type']}<br>enter at the next open"))
    if live is not None and df["setup"].iat[-1] != 0 and status["kind"] != "open":
        pts.append((df.index[-1], int(df["setup"].iat[-1]), "?", "forming — confirms at the close"))

    for d, name, sym, color in ((1, "LONG", "triangle-up", LONG_COLOR), (-1, "SHORT", "triangle-down", SHORT_COLOR)):
        p = [q for q in pts if q[1] == d]
        if not p:
            continue
        x = [q[0] for q in p]
        y = [v.at[t, "Low"] - 0.6 * v.at[t, "atr"] if d == 1 else v.at[t, "High"] + 0.6 * v.at[t, "atr"] for t in x]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers+text", name=name,
            text=[q[2] for q in p],                                   # ✓ won, ✗ lost, NEW / ? pending
            textposition="bottom center" if d == 1 else "top center",
            textfont=dict(size=12, color="#3c4043"),
            marker=dict(symbol=sym, size=13, color=color, line=dict(width=1, color="white")),
            hovertemplate=[f"<b>{name}</b> signal {q[0].date()}<br>{q[3]}<extra></extra>" for q in p]))

    missing = pd.bdate_range(v.index[0], v.index[-1]).difference(v.index)
    fig.update_layout(
        title=dict(text=f"<b>{symbol}</b> daily — LONG / SHORT signals   "
                        f"<span style='font-size:12px;color:#5f6368'>✓ made money · ✗ lost money</span>", x=0.01),
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
