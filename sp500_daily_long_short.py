# Run this once in your terminal or Colab cell (with the ! prefix) before running the script:
# pip install yfinance plotly pandas numpy
#
# S&P 500 DAILY BIAS — "should I be LONG or SHORT today?"
# --------------------------------------------------------
# Every day the script scores the S&P 500 daily chart from -9 (strong short) to +9 (strong long)
# and turns that into one call: LONG, SHORT or STAY OUT.
#
#   python sp500_daily_long_short.py                 # today's call + chart (sp500_daily_bias.html)
#   python sp500_daily_long_short.py --symbol SPY    # trade the ETF instead of the index
#   python sp500_daily_long_short.py --live          # re-check every 15 min during market hours
#   python sp500_daily_long_short.py --long-only     # SHORT calls become STAY OUT
#
# The chart opens by itself: under the cell in Colab/Jupyter, in your web browser from a terminal.
# It is also saved as sp500_daily_bias.html next to where you ran the script.
#
# Before the open : the call comes from yesterday's close (final, does not change during the day).
# During the day  : the script rebuilds today's daily candle from live 5-minute data and shows
#                   (1) whether price action confirms the call (above/below today's open and VWAP)
#                   (2) the provisional call if today closed right now — a flip is an early warning.
#
# The backtest follows the same rules day by day (call at close t -> position on day t+1),
# so the track record printed under the call is what the rules would actually have done.

import argparse
import os
import time
import webbrowser
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ---- Configuration ----
SYMBOL = "^GSPC"            # S&P 500 index. "SPY" = ETF, "ES=F" = futures
START = "2000-01-01"
CSV_PATH = None             # optional: Date,Open,High,Low,Close CSV instead of Yahoo (no live update)

LONG_THRESHOLD = 3          # score >= +3 -> LONG
SHORT_THRESHOLD = -3        # score <= -3 -> SHORT   (in between -> STAY OUT)
EXIT_LEVEL = 0              # hold rule: a LONG call stays on until the score falls to <= 0
                            # (a SHORT until it rises to >= 0). Stops the call flip-flopping
                            # every time the score wiggles around +/-3.
ALLOW_SHORTS = True

INTRADAY_STOP_ATR = 0.5     # intraday stop  = entry -/+ 0.5 x daily ATR
INTRADAY_T1_ATR = 0.5       # first target   = entry +/- 0.5 x ATR
INTRADAY_T2_ATR = 1.0       # second target  = entry +/- 1.0 x ATR
COST_PCT = 0.0002           # commission + slippage per side (0.02%, SPY/ES-like)

CHART_DAYS = 126            # trading days on the chart (~6 months)
HTML_PATH = "sp500_daily_bias.html"
OOS_FRACTION = 0.3          # most recent 30% of history reported separately
LIVE_EVERY_MIN = 15
NY = "America/New_York"

LONG_COLOR, SHORT_COLOR, FLAT_COLOR = "#0f8a5f", "#d95f02", "#9aa0a6"


# ---------------------------------------------------------------- data
def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.capitalize() for c in df.columns]
    return df


def fetch_daily(symbol=SYMBOL, start=START, csv_path=CSV_PATH):
    if csv_path:
        df = _flatten(pd.read_csv(csv_path, parse_dates=[0], index_col=0))
    else:
        import yfinance as yf
        df = _flatten(yf.download(symbol, start=start, interval="1d", auto_adjust=True, progress=False))
    df = df[["Open", "High", "Low", "Close"]].dropna().sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    if df.empty:
        raise RuntimeError(f"No daily data for {symbol}")
    return df


def fetch_intraday(symbol=SYMBOL):
    """Today's 5-minute bars (regular session). Empty DataFrame if unavailable."""
    try:
        import yfinance as yf
        intr = _flatten(yf.download(symbol, period="1d", interval="5m", auto_adjust=True, progress=False))
    except Exception:
        return pd.DataFrame()
    if intr.empty:
        return intr
    intr.index = (intr.index.tz_convert(NY) if intr.index.tz is not None
                  else intr.index.tz_localize("UTC").tz_convert(NY))
    return intr.between_time("09:30", "15:59")


def market_open_now():
    now = pd.Timestamp.now(tz=NY)
    return now.weekday() < 5 and (now.hour, now.minute) >= (9, 30) and now.hour < 16


def merge_live_bar(daily, intr):
    """Overwrite/append today's daily candle with one rebuilt from the 5-minute bars."""
    if intr.empty:
        return daily
    day = pd.Timestamp(intr.index[-1].date())
    bar = [intr["Open"].iloc[0], intr["High"].max(), intr["Low"].min(), intr["Close"].iloc[-1]]
    daily = daily.copy()
    daily.loc[day, ["Open", "High", "Low", "Close"]] = bar
    return daily.sort_index()


# ---------------------------------------------------------------- indicators
def wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close, n):
    d = close.diff()
    rs = wilder(d.clip(lower=0), n) / wilder(-d.clip(upper=0), n).replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def add_indicators(df):
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    for n in (5, 20, 50, 200):
        df[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    df["rsi14"] = rsi(c, 14)
    df["rsi2"] = rsi(c, 2)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = wilder(tr, 14)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    df["prev_high"], df["prev_low"] = h.shift(), l.shift()
    return df


# ---------------------------------------------------------------- daily score
# Each rule: (name, long condition, short condition, points). Score = sum(long pts) - sum(short pts).
def score_rules(df):
    c = df["Close"]
    up, down = c > df["ema200"], c < df["ema200"]
    return [
        ("Long-term trend: close vs 200-day EMA", up, down, 2),
        ("Medium trend: 50-day EMA vs 200-day EMA", df["ema50"] > df["ema200"], df["ema50"] < df["ema200"], 1),
        ("Short trend: 20-day EMA vs 50-day EMA", df["ema20"] > df["ema50"], df["ema20"] < df["ema50"], 1),
        ("Momentum: MACD histogram", df["macd_hist"] > 0, df["macd_hist"] < 0, 1),
        ("Momentum: RSI(14) above 55 / below 45", df["rsi14"] > 55, df["rsi14"] < 45, 1),
        ("Price action: close beyond yesterday's high / low", c > df["prev_high"], c < df["prev_low"], 1),
        ("Pullback: RSI(2) dip in uptrend / pop in downtrend",
         up & (df["rsi2"] < 10), down & (df["rsi2"] > 90), 2),
    ]


def add_bias(df):
    df = df.copy()
    rules = score_rules(df)
    df["score"] = sum(np.where(lc, p, 0) - np.where(sc, p, 0) for _, lc, sc, p in rules)
    df.loc[df.index[:200], "score"] = 0                      # EMA200 warm-up
    bias, state = np.zeros(len(df), dtype=int), 0
    for i, sc in enumerate(df["score"].to_numpy()):
        if sc >= LONG_THRESHOLD:
            state = 1
        elif sc <= SHORT_THRESHOLD and ALLOW_SHORTS:
            state = -1
        elif (state == 1 and sc <= EXIT_LEVEL) or (state == -1 and sc >= -EXIT_LEVEL) or \
                (state == -1 and not ALLOW_SHORTS):
            state = 0
        bias[i] = state
    df["bias"] = bias
    return df


def explain(df, i):
    """Rule-by-rule breakdown of the score on row i."""
    out = []
    for name, lc, sc, p in score_rules(df):
        pts = p if lc.iloc[i] else (-p if sc.iloc[i] else 0)
        out.append((name, pts))
    return out


def label(b):
    return {1: "LONG", -1: "SHORT", 0: "STAY OUT"}[int(b)]


def strength(score):
    a = abs(score)
    return "strong" if a >= 6 else "moderate" if a >= 4 else "weak" if a >= 3 else "no edge"


# ---------------------------------------------------------------- backtest
def backtest(df):
    """Follow the call: bias at close t is held on day t+1.
    cc = hold close-to-close (swing).  oc = day-trade it: in at the open, out at the close."""
    df = df.copy()
    pos = df["bias"].shift().fillna(0)
    cc_ret = df["Close"].pct_change().fillna(0)
    oc_ret = df["Close"] / df["Open"] - 1
    df["ret_cc"] = pos * cc_ret - COST_PCT * pos.diff().abs().fillna(0)
    df["ret_oc"] = pos * oc_ret - 2 * COST_PCT * pos.abs()
    df["pos"] = pos
    df["eq_cc"] = (1 + df["ret_cc"]).cumprod()
    df["eq_oc"] = (1 + df["ret_oc"]).cumprod()
    df["eq_bh"] = df["Close"] / df["Close"].iloc[0]
    return df


def perf(rets, pos):
    traded = rets[pos != 0]
    eq = (1 + rets).cumprod()
    years = max(len(rets) / 252, 1e-9)
    return {
        "days": len(traded),
        "hit %": 100 * (traded > 0).mean() if len(traded) else np.nan,
        "avg day %": 100 * traded.mean() if len(traded) else np.nan,
        "CAGR %": 100 * (eq.iloc[-1] ** (1 / years) - 1),
        "Sharpe": np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else np.nan,
        "max DD %": 100 * (eq / eq.cummax() - 1).min(),
    }


def fmt(d):
    return "  ".join(f"{k} {v:,.2f}" if isinstance(v, (float, np.floating)) else f"{k} {v}"
                     for k, v in d.items())


def print_track_record(bt):
    bt = bt.iloc[200:]
    split = bt.index[int(len(bt) * (1 - OOS_FRACTION))]
    print(f"\nTRACK RECORD  {bt.index[0].date()} -> {bt.index[-1].date()}   (call at close, acted on next day)")
    print(f"  Swing  (hold close->close) {fmt(perf(bt.ret_cc, bt.pos))}")
    print(f"  Day-trade (open->close)    {fmt(perf(bt.ret_oc, bt.pos))}")
    bh = bt["Close"].pct_change().fillna(0)
    print(f"  Buy & hold                 {fmt(perf(bh, pd.Series(1, index=bt.index)))}")
    for side, v in (("LONG", 1), ("SHORT", -1)):
        m = bt.pos == v
        if m.any():
            print(f"  {side:<5} days: {m.sum():>5}   hit {100 * (bt.ret_cc[m] > 0).mean():5.1f}%   "
                  f"avg {100 * bt.ret_cc[m].mean():+.3f}%/day (swing)   "
                  f"hit {100 * (bt.ret_oc[m] > 0).mean():5.1f}%   avg {100 * bt.ret_oc[m].mean():+.3f}%/day (day-trade)")
    for name, part in (("Older", bt[bt.index < split]), (f"Recent {int(OOS_FRACTION * 100)}%", bt[bt.index >= split])):
        print(f"  {name:<11} swing {fmt(perf(part.ret_cc, part.pos))}")


# ---------------------------------------------------------------- today's report
def todays_plan(df, intr, provisional):
    """The call for today comes from the last COMPLETED close."""
    done = df.iloc[:-1] if provisional else df
    i = len(done) - 1
    row = done.iloc[i]
    d = int(row["bias"])
    pivot = (row.High + row.Low + row.Close) / 3
    lv = {"R2": pivot + (row.High - row.Low), "R1": 2 * pivot - row.Low, "Pivot": pivot,
          "S1": 2 * pivot - row.High, "S2": pivot - (row.High - row.Low),
          "Yday high": row.High, "Yday low": row.Low}
    entry = intr["Open"].iloc[0] if (provisional and not intr.empty) else row.Close
    plan = None
    if d != 0:
        a = row.atr
        plan = {"entry": entry, "stop": entry - d * INTRADAY_STOP_ATR * a,
                "t1": entry + d * INTRADAY_T1_ATR * a, "t2": entry + d * INTRADAY_T2_ATR * a}
    return {"date": done.index[-1], "row": row, "bias": d, "levels": lv, "plan": plan,
            "why": explain(done, i)}


def print_report(df, intr, provisional, symbol=SYMBOL):
    p = todays_plan(df, intr, provisional)
    row, d = p["row"], p["bias"]
    now = pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d %H:%M ET")
    bar = "=" * 78
    print(f"\n{bar}\n {symbol} DAILY BIAS   {now}   (based on close of {p['date'].date()})\n{bar}")
    note = strength(row.score) if abs(row.score) >= LONG_THRESHOLD or d == 0 else "holding — signal weakening"
    print(f"\n  >>> TODAY: {label(d)}   score {int(row.score):+d} / 9  ({note}) <<<\n")
    print(f"  Last close {row.Close:,.2f}   daily ATR {row.atr:,.2f} ({100 * row.atr / row.Close:.2f}%)   "
          f"RSI14 {row.rsi14:.0f}   RSI2 {row.rsi2:.0f}")

    print("\n  Why:")
    for name, pts in p["why"]:
        tag = "LONG " if pts > 0 else "SHORT" if pts < 0 else "  -  "
        print(f"    {pts:+d}  {tag}  {name}")

    if p["plan"]:
        pl = p["plan"]
        ref = "today's open" if provisional and not intr.empty else "last close"
        print(f"\n  Intraday plan ({label(d)}, entry ref = {ref} {pl['entry']:,.2f}):")
        print(f"    stop {pl['stop']:,.2f}   target 1 {pl['t1']:,.2f}   target 2 {pl['t2']:,.2f}")
        if d == 1:
            print("    Buy dips toward Pivot/S1 while price holds above the stop; don't chase above R1.")
        else:
            print("    Sell rallies toward Pivot/R1 while price holds below the stop; don't chase below S1.")
    else:
        print("\n  No edge today: stay out, or trade only a clean break of yesterday's high/low.")
    print("  Levels: " + "   ".join(f"{k} {v:,.2f}" for k, v in p["levels"].items()))

    if provisional and not intr.empty:
        live = df.iloc[-1]
        price, opn = intr["Close"].iloc[-1], intr["Open"].iloc[0]
        tp = (intr["High"] + intr["Low"] + intr["Close"]) / 3
        vol = intr["Volume"] if "Volume" in intr and intr["Volume"].sum() > 0 else pd.Series(1.0, index=intr.index)
        vwap = (tp * vol).sum() / vol.sum()
        side = np.sign(price - opn) + np.sign(price - vwap)
        print(f"\n  LIVE {intr.index[-1].strftime('%H:%M')} ET: price {price:,.2f}  "
              f"({100 * (price / row.Close - 1):+.2f}% vs yday)   open {opn:,.2f}   VWAP {vwap:,.2f}")
        hit_stop = p["plan"] and ((live.Low <= p["plan"]["stop"]) if d == 1 else (live.High >= p["plan"]["stop"]))
        if hit_stop:
            print("    Today's range already touched the stop level — the call has failed for today; stand aside.")
        elif d != 0:
            if side * d == 2:
                print(f"    Intraday CONFIRMS the {label(d)} call (price on the right side of open and VWAP).")
            elif side * d == -2:
                print(f"    Intraday is AGAINST the call — wait for price to get back "
                      f"{'above' if d == 1 else 'below'} VWAP {vwap:,.2f} before acting.")
            else:
                print("    Intraday is MIXED — no confirmation yet.")
        print(f"    If today closed now: {label(live.bias)}  (score {int(live.score):+d})"
              + ("   <-- FLIP WARNING" if int(live.bias) != d else ""))
    print(bar)


# ---------------------------------------------------------------- chart
def in_notebook():
    """True inside Colab / Jupyter, where the chart can be drawn right under the cell."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:
        return False


def plot_chart(df, provisional, symbol=SYMBOL, days=CHART_DAYS, html_path=HTML_PATH, show=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    v = df.iloc[-days:].copy()
    p_bias = v["bias"].shift()                 # what you were told to do ON each day
    p_bias.iloc[0] = df["bias"].iloc[-days - 1] if len(df) > days else 0
    today_call = int(df["bias"].iloc[-2] if provisional else df["bias"].iloc[-1])

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.62, 0.18, 0.20],
                        subplot_titles=("Price — shaded by the call for that day",
                                        f"Daily score (LONG ≥ {LONG_THRESHOLD}, SHORT ≤ {SHORT_THRESHOLD})",
                                        "Growth of $1 following the calls"))

    # shading: one band per run of identical calls
    run_id = (p_bias != p_bias.shift()).cumsum()
    nxt = pd.Series(list(v.index[1:]) + [v.index[-1] + pd.Timedelta(days=1)], index=v.index)
    for _, run in v.groupby(run_id):
        b = p_bias.loc[run.index[0]]
        if b == 0:
            continue
        fig.add_shape(type="rect", xref="x", yref="y domain", x0=run.index[0], x1=nxt[run.index[-1]],
                      y0=0, y1=1, fillcolor=LONG_COLOR if b > 0 else SHORT_COLOR,
                      opacity=0.10, line_width=0, layer="below")

    fig.add_trace(go.Candlestick(
        x=v.index, open=v.Open, high=v.High, low=v.Low, close=v.Close, name="S&P 500",
        increasing=dict(line=dict(color="#4d8f76", width=1), fillcolor="#a9d3c2"),
        decreasing=dict(line=dict(color="#b5643a", width=1), fillcolor="#f0b48f"),
        showlegend=False), row=1, col=1)
    for col, name, color, dash in (("ema20", "EMA 20", "#2166ac", "solid"),
                                   ("ema50", "EMA 50", "#6a51a3", "solid"),
                                   ("ema200", "EMA 200", "#3c4043", "dot")):
        fig.add_trace(go.Scatter(x=v.index, y=v[col], name=name, mode="lines",
                                 line=dict(color=color, width=1.5, dash=dash), hoverinfo="skip"), row=1, col=1)

    # arrows where the call changes (signal fires at that close -> act next day)
    flips = v[(v["bias"] != v["bias"].shift()) & (v.index != v.index[0])]
    pad = v["atr"] * 0.6
    for b, sym, txt, color, y, pos in ((1, "triangle-up", "LONG", LONG_COLOR, flips.Low - pad, "bottom center"),
                                       (-1, "triangle-down", "SHORT", SHORT_COLOR, flips.High + pad, "top center"),
                                       (0, "circle", "OUT", FLAT_COLOR, flips.High + pad, "top center")):
        f = flips[flips["bias"] == b]
        if f.empty:
            continue
        fig.add_trace(go.Scatter(
            x=f.index, y=y.loc[f.index], mode="markers+text", name=f"→ {txt}",
            text=[txt if b else ""] * len(f), textposition=pos, textfont=dict(size=10, color="#3c4043"),
            marker=dict(symbol=sym, size=13 if b else 8, color=color, line=dict(width=1, color="white")),
            hovertemplate=[f"{d.date()} close: call → {txt}<br>score {s:+d}<extra></extra>"
                           for d, s in zip(f.index, f.score.astype(int))]), row=1, col=1)

    # today's levels
    plan = todays_plan(df, pd.DataFrame(), provisional)
    x_end = v.index[-1] + pd.Timedelta(days=5)
    lines = [("Pivot", plan["levels"]["Pivot"], "#9aa0a6")]
    if plan["plan"]:
        lines += [("Stop", plan["plan"]["stop"], "#c5221f"), ("T1", plan["plan"]["t1"], "#3c4043"),
                  ("T2", plan["plan"]["t2"], "#3c4043")]
    for name, y, color in lines:
        fig.add_trace(go.Scatter(x=[v.index[-1], x_end], y=[y, y], mode="lines",
                                 line=dict(color=color, width=1, dash="dash"), showlegend=False,
                                 hovertemplate=f"{name} {y:,.2f}<extra></extra>"), row=1, col=1)
        fig.add_annotation(x=x_end, y=y, text=f"{name} {y:,.0f}", showarrow=False, xanchor="left",
                           font=dict(size=11, color=color), row=1, col=1)

    # score panel
    sc_color = np.where(v.score >= LONG_THRESHOLD, LONG_COLOR,
                        np.where(v.score <= SHORT_THRESHOLD, SHORT_COLOR, FLAT_COLOR))
    fig.add_trace(go.Bar(x=v.index, y=v.score, marker_color=sc_color, name="Score", showlegend=False,
                         hovertemplate="score %{y:+d}<extra></extra>"), row=2, col=1)
    for t in (LONG_THRESHOLD, SHORT_THRESHOLD):
        fig.add_hline(y=t, line_dash="dot", line_color="#9aa0a6", line_width=1, row=2, col=1)

    # equity panel, both indexed to 1 at the start of the window
    for col, name, color, dash in (("eq_cc", "Following the calls", LONG_COLOR, "solid"),
                                   ("eq_bh", "Buy & hold", "#5f6368", "dot")):
        fig.add_trace(go.Scatter(x=v.index, y=v[col] / v[col].iloc[0], name=name, mode="lines",
                                 line=dict(color=color, width=2, dash=dash),
                                 hovertemplate=name + " %{y:.3f}<extra></extra>"), row=3, col=1)

    status = f"TODAY: {label(today_call)}" + ("  (live, provisional candle)" if provisional else "")
    fig.update_layout(
        title=dict(text=f"<b>{symbol} daily bias</b> — {status}", x=0.01, y=0.985, yanchor="top"),
        height=950, template="plotly_white", hovermode="x unified",
        xaxis_rangeslider_visible=False, margin=dict(l=60, r=120, t=120, b=40),
        legend=dict(orientation="h", y=1.06, x=0.0, xanchor="left"),
        font=dict(family="Inter, Segoe UI, Arial", size=12, color="#3c4043"))
    missing = pd.bdate_range(v.index[0], x_end).difference(df.index)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=missing.difference(
        pd.bdate_range(v.index[-1] + pd.Timedelta(days=1), x_end)))],
        showgrid=False, range=[v.index[0] - pd.Timedelta(days=1), x_end])
    fig.update_yaxes(gridcolor="#eceff1", zeroline=False)
    fig.update_yaxes(tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(dtick=3, row=2, col=1)

    if html_path:
        fig.write_html(html_path, include_plotlyjs=True)      # self-contained, opens offline too
        print(f"Chart saved to {os.path.abspath(html_path)}")
    if show:
        if in_notebook():
            fig.show()                                         # Colab / Jupyter: draws under the cell
        elif html_path:
            webbrowser.open("file://" + os.path.abspath(html_path))   # terminal: opens your browser
    return fig


# ---------------------------------------------------------------- run
def run_once(symbol=SYMBOL, start=START, csv_path=CSV_PATH, plot=True, show=True, history=True):
    df = fetch_daily(symbol, start, csv_path)
    intr = pd.DataFrame() if csv_path else fetch_intraday(symbol)
    provisional = False
    if not intr.empty and market_open_now() and intr.index[-1].date() == pd.Timestamp.now(tz=NY).date():
        df = merge_live_bar(df, intr)
        provisional = True
    else:
        intr = pd.DataFrame()
    df = backtest(add_bias(add_indicators(df)))
    print_report(df, intr, provisional, symbol)
    if history:
        print_track_record(df.iloc[:-1] if provisional else df)
    if plot:
        plot_chart(df, provisional, symbol, show=show)
    return df


def run_live(symbol=SYMBOL, every_min=LIVE_EVERY_MIN, **kw):
    first = True
    while True:
        if market_open_now() or first:
            if in_notebook() and not first:
                from IPython.display import clear_output
                clear_output(wait=True)                        # replace the old report + chart
            run_once(symbol, history=first, **kw)
            first = False
        else:
            print(f"{datetime.now():%H:%M} market closed — waiting…")
        time.sleep(every_min * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="S&P 500 daily LONG/SHORT bias with live intraday check")
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--start", default=START)
    ap.add_argument("--csv", default=CSV_PATH, help="load daily OHLC from CSV instead of Yahoo")
    ap.add_argument("--live", action="store_true", help="re-check every --every minutes during market hours")
    ap.add_argument("--every", type=int, default=LIVE_EVERY_MIN)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="save the chart but don't open/show it")
    args, _ = ap.parse_known_args()       # parse_known_args so it also runs inside Colab/Jupyter
    if args.long_only:
        ALLOW_SHORTS = False
    kw = dict(start=args.start, csv_path=args.csv, plot=not args.no_plot, show=not args.no_open)
    if args.live:
        run_live(args.symbol, args.every, **kw)
    else:
        run_once(args.symbol, **kw)
