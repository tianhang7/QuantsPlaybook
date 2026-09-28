"""可复现的鳄鱼线指数择时/轮动回测（不依赖 TA-Lib、vectorbt）。"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
CODES = ["000300.SH", "000001.SH", "000016.SH", "000905.SH", "000852.SH", "000985.CSI"]


def alligator(s: pd.Series):
    x = s.to_numpy(float)
    jaw = pd.Series(x).rolling(13).mean().shift(8)
    teeth = pd.Series(x).rolling(8).mean().shift(5)
    lip = pd.Series(x).rolling(5).mean().shift(3)
    a = np.column_stack([jaw, teeth, lip])
    bull = np.all(np.diff(a, axis=1) > 0, axis=1)
    bear = np.all(np.diff(a, axis=1) < 0, axis=1)
    return pd.Series(np.where(bull, 1, np.where(bear, -1, np.nan)), index=s.index).ffill().fillna(0)


def ao(s_high, s_low):
    x = (s_high - s_low) / 2
    v = x.rolling(5).mean() - x.rolling(34).mean()
    return v.diff().rolling(3).apply(lambda z: 1 if (z > 0).all() else -1 if (z < 0).all() else np.nan, raw=True).ffill().fillna(0)


def fractal(close, high, low):
    # 最近三根中的高低点形成五柱分形，收盘突破其确认位后触发。
    f = pd.Series(np.nan, index=close.index)
    up = (high.shift(1) > high.shift(3)) & (high.shift(1) > high.shift(2))
    down = (low.shift(1) < low.shift(3)) & (low.shift(1) < low.shift(2))
    f[up] = high.shift(1)[up]
    f[down] = low.shift(1)[down]
    sig = pd.Series(0.0, index=close.index)
    sig[(close > f) & (f > 0)] = 1
    sig[(close < f) & (f > 0)] = -1
    return sig.ffill().fillna(0)


def macd(s):
    d = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    e = d.ewm(span=9, adjust=False).mean()
    h = 2 * (d - e)
    bull = (d > e) & (d.shift(1) < e.shift(1)) & (h > 0) & (d >= 0) & (e >= 0)
    bear = (d < e) & (d.shift(1) > e.shift(1)) & (h < 0) & (d < 0) & (e < 0)
    return pd.Series(np.where(bull, 1, np.where(bear, -1, np.nan)), index=s.index).ffill().fillna(0)


def north_signal(n):
    x = n["north_money"]
    hi = x.rolling(60).quantile(.8)
    lo = x.rolling(60).quantile(.2)
    return pd.Series(np.where(x > hi, 1, np.where(x < lo, -1, np.nan)), index=n.index).ffill().fillna(0)


def stats(ret):
    nav = (1 + ret).cumprod()
    dd = nav / nav.cummax() - 1
    years = max(len(ret) / 252, 1 / 252)
    return {"累计收益": nav.iloc[-1]-1, "年化收益": nav.iloc[-1]**(1/years)-1,
            "年化波动": ret.std()*np.sqrt(252), "夏普": ret.mean()/ret.std()*np.sqrt(252),
            "最大回撤": dd.min(), "交易次数": int(ret.attrs.get("trades", 0))}


def run(start=None, end=None, init_cash=1e8, fee=.00015):
    daily = pd.read_parquet(DATA / "daily.parquet")
    nm = pd.read_parquet(DATA / "north_money.parquet")
    d = daily.pivot_table(index="trade_date", columns="code", values=["open", "high", "low", "close"])
    d.index = pd.to_datetime(d.index)
    nm.index = pd.to_datetime(nm.index)
    d = d.sort_index()
    codes = [c for c in CODES if c in d["close"].columns]
    begin = max(pd.Timestamp(start) if start else d.index.min(), pd.Timestamp(nm.index.min()))
    end = pd.Timestamp(end) if end else d.index.max()
    idx = d.loc[begin:end].index
    n = north_signal(nm).reindex(idx).ffill().fillna(0)
    signals = {}
    for c in codes:
        a, o, f, m = alligator(d["close"][c]), ao(d["high"][c], d["low"][c]), fractal(d["close"][c], d["high"][c], d["low"][c]), macd(d["close"][c])
        entry = (a == 1) & ((f == 1) | (m == 1))
        exit_ = (a == -1) | (o == -1) | (f == -1) | (m == -1)
        signals[c] = (entry, exit_)
    nav, rets, trades = {}, {}, {}
    for c in codes:
        entry, exit_ = signals[c]
        position = False; cash = init_cash; units = 0; value = []; trade_count = 0
        for i, dt in enumerate(idx):
            op = d["open"][c].loc[dt]; close = d["close"][c].loc[dt]
            if i and not np.isnan(op):
                if not position and bool(entry.iloc[i-1]): units = int((cash*(1-fee)) / op / 100)*100; cash -= units*op*(1+fee); position=True; trade_count+=1
                elif position and bool(exit_.iloc[i-1]) and units:
                    cash += units*op*(1-fee); units=0; position=False; trade_count+=1
            value.append(cash + units*close)
        s = pd.Series(value, index=idx, name=c); nav[c] = s / init_cash; rets[c] = s.pct_change().fillna(0); rets[c].attrs["trades"] = trade_count; trades[c] = trade_count
    out = pd.DataFrame(rets); OUT.mkdir(exist_ok=True)
    summary = pd.DataFrame({c: stats(rets[c]) for c in codes}).T
    out.to_csv(OUT / "daily_returns.csv", encoding="utf-8-sig")
    pd.DataFrame(nav).to_csv(OUT / "nav.csv", encoding="utf-8-sig")
    summary.to_csv(OUT / "metrics.csv", encoding="utf-8-sig")
    (OUT / "nav.png").write_bytes(b"")
    ax = pd.DataFrame(nav).plot(figsize=(12, 6), title="Alligator timing / rotation"); ax.figure.tight_layout(); ax.figure.savefig(OUT / "nav.png", dpi=140); plt.close(ax.figure)
    print(f"样本: {idx[0].date()} ~ {idx[-1].date()}, 指数: {len(codes)}")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print("输出:", OUT)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--start"); p.add_argument("--end"); p.add_argument("--cash", type=float, default=1e8); a=p.parse_args(); run(a.start, a.end, a.cash)

if __name__ == "__main__":
    main()
