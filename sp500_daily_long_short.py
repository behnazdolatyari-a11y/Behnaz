# Run this once in your terminal or Colab cell (with the ! prefix) before running the script:
# pip install yfinance plotly pandas numpy
#
# S&P 500 DAILY long/short swing strategy
# ---------------------------------------
# Trend filter : trade only in the direction of the 200-day trend
#                LONG regime  -> close > EMA200 and EMA50 > EMA200
#                SHORT regime -> close < EMA200 and EMA50 < EMA200
# Strength     : ADX >= ADX_MIN (skip choppy, trendless markets)
# Trigger      : pullback-and-resume — price crosses back over EMA20, or MACD histogram
#                crosses zero, in the direction of the trend
# Momentum     : RSI must not be stretched (no longs when overbought, no shorts when oversold)
# Risk         : ATR stop loss, ATR take profit, ATR trailing stop, max holding days,
#                and exit/reverse on an opposite signal
#
# Signals are computed on the CLOSE of day t and filled at the OPEN of day t+1,
# so the backtest never uses information it would not have had.

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ---- Configuration ----
SYMBOL = "^GSPC"            # S&P 500 index. Use "SPY" (ETF) or "ES=F" (futures) for a tradable instrument
START = "2000-01-01"
CSV_PATH = None             # optional: load OHLC data from a CSV (Date,Open,High,Low,Close[,Volume]) instead of Yahoo

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200
RSI_LEN = 14
ATR_LEN = 14
ADX_LEN = 14
ADX_MIN = 18                # below this the market is trendless -> no new entries

RSI_LONG_MAX = 70           # don't buy when already overbought
RSI_SHORT_MIN = 30          # don't short when already oversold

ATR_STOP_MULT = 2.0         # initial stop  = entry -/+ 2.0 x ATR
ATR_TARGET_MULT = 4.0       # profit target = entry +/- 4.0 x ATR  (2:1 reward/risk)
ATR_TRAIL_MULT = 3.0        # trailing stop = best close -/+ 3.0 x ATR
MAX_HOLD_DAYS = 40          # time exit
ALLOW_SHORTS = True
COST_PCT = 0.0005           # commission + slippage per side (0.05%)

CHART_DAYS = 252            # trading days drawn on the chart (~1 year)
OOS_FRACTION = 0.3          # most recent 30% of history reported separately as "out-of-sample"


# ---------------------------------------------------------------- data
def fetch_data(symbol=SYMBOL, start=START, csv_path=CSV_PATH):
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
    else:
        import yfinance as yf
        df = yf.download(symbol, start=start, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    df.columns = [c.capitalize() for c in df.columns]
    df = df[["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df.columns else [])]
    df = df.dropna().sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    return df


# ---------------------------------------------------------------- indicators
def wilder(series, n):
    return series.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df):
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]

    df["ema_fast"] = c.ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_mid"] = c.ewm(span=EMA_MID, adjust=False).mean()
    df["ema_slow"] = c.ewm(span=EMA_SLOW, adjust=False).mean()

    delta = c.diff()
    gain = wilder(delta.clip(lower=0), RSI_LEN)
    loss = wilder(-delta.clip(upper=0), RSI_LEN)
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    prev_c = c.shift()
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = wilder(tr, ATR_LEN)

    up, down = h.diff(), -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr_adx = wilder(tr, ADX_LEN)
    plus_di = 100 * wilder(pd.Series(plus_dm, index=df.index), ADX_LEN) / atr_adx
    minus_di = 100 * wilder(pd.Series(minus_dm, index=df.index), ADX_LEN) / atr_adx
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = wilder(dx.fillna(0), ADX_LEN)

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    return df


# ---------------------------------------------------------------- signals
def add_signals(df):
    df = df.copy()
    c = df["Close"]

    long_regime = (c > df["ema_slow"]) & (df["ema_mid"] > df["ema_slow"])
    short_regime = (c < df["ema_slow"]) & (df["ema_mid"] < df["ema_slow"])
    trending = df["adx"] >= ADX_MIN

    cross_up_ema = (c > df["ema_fast"]) & (c.shift() <= df["ema_fast"].shift())
    cross_dn_ema = (c < df["ema_fast"]) & (c.shift() >= df["ema_fast"].shift())
    macd_up = (df["macd_hist"] > 0) & (df["macd_hist"].shift() <= 0)
    macd_dn = (df["macd_hist"] < 0) & (df["macd_hist"].shift() >= 0)

    df["long_signal"] = long_regime & trending & (cross_up_ema | macd_up) & (df["rsi"] < RSI_LONG_MAX)
    df["short_signal"] = short_regime & trending & (cross_dn_ema | macd_dn) & (df["rsi"] > RSI_SHORT_MIN)
    if not ALLOW_SHORTS:
        df["short_signal"] = False

    warm = df.index[:EMA_SLOW]                      # EMA200 needs ~200 bars to settle
    df.loc[warm, ["long_signal", "short_signal"]] = False
    df["signal"] = np.select([df["long_signal"], df["short_signal"]], [1, -1], 0)
    return df


# ---------------------------------------------------------------- backtest
def backtest(df):
    """One position at a time. Signal on close of day i -> entry at open of day i+1."""
    o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    atr, sig, dates = df["atr"].to_numpy(), df["signal"].to_numpy(), df.index

    trades, pos = [], None
    equity = np.ones(len(df))
    eq = 1.0

    def close_trade(i, price, reason):
        nonlocal eq, pos
        d = pos["dir"]
        gross = d * (price - pos["entry"]) / pos["entry"]
        ret = gross - 2 * COST_PCT
        eq *= 1 + ret
        trades.append({
            "side": "LONG" if d == 1 else "SHORT",
            "entry_date": pos["entry_date"], "entry": pos["entry"],
            "exit_date": dates[i], "exit": price, "reason": reason,
            "days": i - pos["entry_i"], "return_pct": ret * 100,
        })
        pos = None

    for i in range(1, len(df)):
        # 1) enter at today's open if yesterday closed with a signal (or reverse)
        if sig[i - 1] != 0 and (pos is None or pos["dir"] != sig[i - 1]):
            if pos is not None:
                close_trade(i, o[i], "Reverse")
            d = int(sig[i - 1])
            a = atr[i - 1]
            pos = {"dir": d, "entry": o[i], "entry_i": i, "entry_date": dates[i],
                   "stop": o[i] - d * ATR_STOP_MULT * a,
                   "target": o[i] + d * ATR_TARGET_MULT * a,
                   "best": o[i]}

        # 2) manage open position intraday (stop checked first = conservative)
        if pos is not None:
            d = pos["dir"]
            if d == 1:
                if l[i] <= pos["stop"]:
                    close_trade(i, min(o[i], pos["stop"]), "Stop")
                elif h[i] >= pos["target"]:
                    close_trade(i, max(o[i], pos["target"]), "Target")
            else:
                if h[i] >= pos["stop"]:
                    close_trade(i, max(o[i], pos["stop"]), "Stop")
                elif l[i] <= pos["target"]:
                    close_trade(i, min(o[i], pos["target"]), "Target")

        # 3) end-of-day: time exit, then ratchet the trailing stop
        if pos is not None:
            d = pos["dir"]
            if i - pos["entry_i"] >= MAX_HOLD_DAYS:
                close_trade(i, c[i], "Time")
            else:
                pos["best"] = max(pos["best"], c[i]) if d == 1 else min(pos["best"], c[i])
                trail = pos["best"] - d * ATR_TRAIL_MULT * atr[i]
                pos["stop"] = max(pos["stop"], trail) if d == 1 else min(pos["stop"], trail)

        open_pnl = 0.0 if pos is None else pos["dir"] * (c[i] - pos["entry"]) / pos["entry"]
        equity[i] = eq * (1 + open_pnl)

    trades = pd.DataFrame(trades)
    df = df.copy()
    df["equity"] = equity
    df["buy_hold"] = df["Close"] / df["Close"].iloc[0]
    return df, trades, pos


# ---------------------------------------------------------------- reporting
def equity_stats(curve):
    rets = curve.pct_change().dropna()
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1
    sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else np.nan
    max_dd = (curve / curve.cummax() - 1).min()
    return {"Total return %": (curve.iloc[-1] / curve.iloc[0] - 1) * 100,
            "CAGR %": cagr * 100, "Sharpe": sharpe, "Max drawdown %": max_dd * 100}


def trade_stats(trades):
    if trades.empty:
        return {"Trades": 0}
    r = trades["return_pct"]
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf
    return {"Trades": len(r), "Win rate %": 100 * len(wins) / len(r),
            "Avg win %": wins.mean() if len(wins) else 0, "Avg loss %": losses.mean() if len(losses) else 0,
            "Profit factor": pf, "Avg days held": trades["days"].mean()}


def fmt(d):
    return "  ".join(f"{k}: {v:,.2f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}"
                     for k, v in d.items())


def print_report(df, trades, open_pos, symbol=SYMBOL):
    print(f"\n=== {symbol} daily long/short  |  {df.index[0].date()} -> {df.index[-1].date()} ===")
    print("Strategy  ", fmt(equity_stats(df["equity"])))
    print("Buy & hold", fmt(equity_stats(df["buy_hold"])))
    print("\nAll trades", fmt(trade_stats(trades)))
    for side in ("LONG", "SHORT"):
        if not trades.empty:
            print(f"{side:<10}", fmt(trade_stats(trades[trades["side"] == side])))

    split = df.index[int(len(df) * (1 - OOS_FRACTION))]
    if not trades.empty:
        print(f"\nIn-sample  (< {split.date()})", fmt(trade_stats(trades[trades["entry_date"] < split])))
        print(f"Out-sample (>= {split.date()})", fmt(trade_stats(trades[trades["entry_date"] >= split])))
        print("\nExit reasons:", trades["reason"].value_counts().to_dict())
        print("\nLast 10 trades:")
        show = trades.tail(10).copy()
        for col in ("entry_date", "exit_date"):
            show[col] = pd.to_datetime(show[col]).dt.date
        print(show.round(2).to_string(index=False))

    last = df.iloc[-1]
    print(f"\n--- Status at close {df.index[-1].date()} ---")
    print(f"Close {last.Close:,.2f}  EMA20 {last.ema_fast:,.2f}  EMA50 {last.ema_mid:,.2f}  "
          f"EMA200 {last.ema_slow:,.2f}  RSI {last.rsi:.1f}  ADX {last.adx:.1f}  ATR {last.atr:.2f}")
    if open_pos:
        side = "LONG" if open_pos["dir"] == 1 else "SHORT"
        print(f"Open position: {side} since {open_pos['entry_date'].date()} @ {open_pos['entry']:,.2f}  "
              f"stop {open_pos['stop']:,.2f}  target {open_pos['target']:,.2f}")
    else:
        print("Open position: flat")
    action = {1: "BUY / go LONG at next open", -1: "SELL SHORT at next open", 0: "no new signal"}
    print("Signal for next session:", action[int(last.signal)])


def plot_chart(df, trades, symbol=SYMBOL, days=CHART_DAYS, html_path=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    v = df.iloc[-days:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.15, 0.25],
                        vertical_spacing=0.03, subplot_titles=(f"{symbol} daily", "RSI", "Equity"))
    fig.add_trace(go.Candlestick(x=v.index, open=v.Open, high=v.High, low=v.Low, close=v.Close,
                                 name="Price"), row=1, col=1)
    for col, name, color in (("ema_fast", f"EMA{EMA_FAST}", "#f2a900"),
                             ("ema_mid", f"EMA{EMA_MID}", "#2b7bff"),
                             ("ema_slow", f"EMA{EMA_SLOW}", "#9b59b6")):
        fig.add_trace(go.Scatter(x=v.index, y=v[col], name=name, line=dict(width=1.3, color=color)),
                      row=1, col=1)

    if not trades.empty:
        t = trades[trades["entry_date"] >= v.index[0]]
        for side, sym, color in (("LONG", "triangle-up", "#16a34a"), ("SHORT", "triangle-down", "#dc2626")):
            s = t[t["side"] == side]
            fig.add_trace(go.Scatter(
                x=s["entry_date"], y=s["entry"], mode="markers", name=f"{side} entry",
                marker=dict(symbol=sym, size=13, color=color, line=dict(width=1, color="black")),
                hovertext=[f"{side} @ {e:,.2f}<br>exit {x:,.2f} ({rs})<br>{r:+.2f}%"
                           for e, x, rs, r in zip(s.entry, s.exit, s.reason, s.return_pct)]),
                row=1, col=1)
        fig.add_trace(go.Scatter(x=t["exit_date"], y=t["exit"], mode="markers", name="Exit",
                                 marker=dict(symbol="x", size=9, color="#555")), row=1, col=1)

    fig.add_trace(go.Scatter(x=v.index, y=v.rsi, name="RSI", line=dict(color="#888")), row=2, col=1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line_dash="dot", line_color="#bbb", row=2, col=1)

    fig.add_trace(go.Scatter(x=v.index, y=v.equity / v.equity.iloc[0], name="Strategy",
                             line=dict(color="#16a34a")), row=3, col=1)
    fig.add_trace(go.Scatter(x=v.index, y=v.buy_hold / v.buy_hold.iloc[0], name="Buy & hold",
                             line=dict(color="#888", dash="dot")), row=3, col=1)

    fig.update_layout(height=900, xaxis_rangeslider_visible=False, template="plotly_white",
                      legend=dict(orientation="h", y=1.04))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    if html_path:
        fig.write_html(html_path)
        print(f"\nChart saved to {html_path}")
    else:
        fig.show()


def run(symbol=SYMBOL, start=START, csv_path=CSV_PATH, plot=True, html_path=None):
    df = fetch_data(symbol, start, csv_path)
    df = add_signals(add_indicators(df))
    df, trades, open_pos = backtest(df)
    print_report(df, trades, open_pos, symbol)
    if plot:
        plot_chart(df, trades, symbol, html_path=html_path)
    return df, trades


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="S&P 500 daily long/short signals + backtest")
    p.add_argument("--symbol", default=SYMBOL)
    p.add_argument("--start", default=START)
    p.add_argument("--csv", default=CSV_PATH, help="load OHLC from CSV instead of Yahoo Finance")
    p.add_argument("--html", default=None, help="save chart to this HTML file instead of opening it")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--long-only", action="store_true")
    args, _ = p.parse_known_args()          # parse_known_args so it also runs inside Colab/Jupyter
    if args.long_only:
        ALLOW_SHORTS = False
    run(args.symbol, args.start, args.csv, plot=not args.no_plot, html_path=args.html)
