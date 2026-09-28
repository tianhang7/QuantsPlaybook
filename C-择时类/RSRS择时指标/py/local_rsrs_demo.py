"""使用公开行情在本地体验 RSRS 择时策略。

默认标的为沪深300指数（Yahoo Finance 代码 000300.SS），策略参数采用
原研报常见的 N=18、M=600，并用修正后的右偏标准化处理负偏态。信号在
次日生效，以避免使用当日收盘后才能得到的信息交易当日收益。

运行：
    python local_rsrs_demo.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

TICKER = "000300.SS"
START = "2005-04-08"
N = 18
M = 600
POSITION_THRESHOLD = 0.8
COST = 0.001
HERE = Path(__file__).resolve().parent
CACHE = HERE / "rsrs_000300_daily.csv"
OUTPUT = HERE / "rsrs_local_backtest.png"


def load_ohlc() -> pd.DataFrame:
    """从东方财富读取沪深300日线；已有本地缓存则直接使用。"""
    if CACHE.exists():
        return prepare_cached_data(CACHE)

    url = "https://91.push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": "1.000300", "klt": "101", "fqt": "0",
        "beg": START.replace("-", ""), "end": "20500101",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56",
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
    response = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
    assert response is not None
    rows = response.json().get("data", {}).get("klines", [])
    if not rows:
        raise RuntimeError("东方财富行情下载失败")
    parsed = [row.split(",") for row in rows]
    data = pd.DataFrame(parsed, columns=["Date", "open", "close", "high", "low", "volume"])
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date").astype(float)
    data.to_csv(CACHE, index_label="Date")
    return data


def parse_github_csv(path: Path) -> pd.DataFrame:
    """解析第三方历史指数 CSV，忽略编码和损坏的代码/名称行。"""
    import re
    records = []
    for line in path.read_bytes().decode("latin1", errors="ignore").splitlines():
        match = re.match(r"^(\d{4}-\d{2}-\d{2}),", line)
        if not match:
            continue
        values = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line[match.end():])
        if len(values) < 6 or values[0] != "000300":
            continue
        close, high, low, open_ = map(float, values[2:6])
        if min(close, high, low, open_) <= 0 or high < max(close, open_, low):
            continue
        records.append((pd.Timestamp(match.group(1)), open_, high, low, close))
    result = pd.DataFrame(records, columns=["Date", "open", "high", "low", "close"])
    return result.set_index("Date").sort_index()


def prepare_cached_data(path: Path) -> pd.DataFrame:
    """兼容规范化缓存与第三方原始 CSV，并立即写回标准缓存。"""
    header = path.read_text(encoding="latin1").splitlines()[0].lower()
    if {"open", "high", "low", "close"}.issubset(header.split(",")):
        data = pd.read_csv(path, index_col="Date", parse_dates=True)
    else:
        data = parse_github_csv(path)
    if len(data) < 2:
        raise RuntimeError(f"缓存数据不足: {path}")
    data.to_csv(path, index_label="Date")
    return data


def right_tail_zscore(values: pd.Series, window: int) -> pd.Series:
    """估计正态参数，并计算原值减右尾 5% 均值后的 z-score。"""
    # 用滚动 5% 尾部分位数的均值估计右尾，降低极端值对 z-score 的影响。
    tail = values.rolling(window).apply(
        lambda x: x[x >= np.quantile(x, 0.95)].mean(), raw=True
    )
    right_tail_mean = tail
    adjusted = values - right_tail_mean
    return (adjusted - adjusted.rolling(window).mean()) / adjusted.rolling(window).std()


def rolling_slope(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """计算 high 对 low 的 N 日滚动 OLS 回归斜率。"""
    x = low.to_numpy()
    y = high.to_numpy()
    slopes = np.full(len(high), np.nan)
    for i in range(window - 1, len(high)):
        design = sm.add_constant(x[i - window + 1 : i + 1], has_constant="add")
        slopes[i] = sm.OLS(y[i - window + 1 : i + 1], design).fit().params[1]
    return pd.Series(slopes, index=high.index)


def backtest(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造信号并运行次日生效的全仓/空仓回测。"""
    close = data["close"]
    data = data[["high", "low"]].dropna()
    slope = rolling_slope(data["High"], data["Low"], N) if "High" in data else rolling_slope(data["high"], data["low"], N)
    score = right_tail_zscore(slope, M).reindex(close.index)
    desired = (score >= POSITION_THRESHOLD).astype(float)
    desired[desired.eq(0) & score.lt(-POSITION_THRESHOLD)] = -1.0
    # T-1 收盘信号用于 T 日收益。
    position = desired.shift(1).fillna(0.0)
    ret = close.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_ret = position * ret - turnover * COST
    frame = pd.DataFrame(
        {"close": close, "rsrs": score, "desired": desired, "position": position,
         "benchmark": ret, "strategy": strategy_ret}
    ).dropna(subset=["rsrs"])
    frame["benchmark_nav"] = (1 + frame["benchmark"]).cumprod()
    frame["strategy_nav"] = (1 + frame["strategy"]).cumprod()
    return frame, score


def metrics(nav: pd.Series, ret: pd.Series) -> dict[str, float]:
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    annual = nav.iloc[-1] ** (1 / years) - 1
    drawdown = nav / nav.cummax() - 1
    sharpe = ret.mean() / ret.std(ddof=1) * np.sqrt(244) if ret.std() else np.nan
    return {
        "累计收益": nav.iloc[-1] - 1,
        "年化收益": annual,
        "最大回撤": drawdown.min(),
        "夏普比率": sharpe,
        "交易日": len(nav),
    }


def main() -> None:
    ohlc = load_ohlc()
    frame, score = backtest(ohlc)
    results = pd.DataFrame(
        {name: metrics(frame[f"{name.lower()}_nav"], frame[name.lower()])
         for name in ("Benchmark", "Strategy")}
    ).T

    print(f"RSRS 有效样本: {frame.index[0].date()} 至 {frame.index[-1].date()}")
    print(f"参数: N={N}, M={M}, 开仓阈值={POSITION_THRESHOLD}, 单边成本={COST:.1%}")
    print(results.to_string(formatters={
        "累计收益": lambda x: f"{x:.2%}", "年化收益": lambda x: f"{x:.2%}",
        "最大回撤": lambda x: f"{x:.2%}", "夏普比率": lambda x: f"{x:.2f}",
        "交易日": lambda x: f"{int(x)}",
    }))

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(frame["close"], color="black", linewidth=1, label="沪深300")
    axes[0].set_title("沪深300指数")
    axes[0].legend()
    axes[1].plot(frame["rsrs"], color="gray", linewidth=1, label="右偏标准化 RSRS")
    axes[1].axhline(POSITION_THRESHOLD, color="red", linestyle="--", label="开仓阈值")
    axes[1].axhline(-POSITION_THRESHOLD, color="green", linestyle="--", label="平仓阈值")
    axes[1].legend()
    axes[1].set_title("RSRS 信号")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=160)
    plt.close(fig)

    nav_fig, ax = plt.subplots(figsize=(13, 5))
    frame[["benchmark_nav", "strategy_nav"]].plot(ax=ax)
    ax.set_title("RSRS 择时与买入持有沪深300净值对比")
    ax.grid(alpha=0.3)
    nav_fig.tight_layout()
    nav_fig.savefig(OUTPUT.with_name("rsrs_local_nav.png"), dpi=160)
    plt.close(nav_fig)
    results.to_csv(OUTPUT.with_name("rsrs_local_metrics.csv"), encoding="utf-8-sig")
    print(f"图表已保存: {OUTPUT}")
    print(f"净值图已保存: {OUTPUT.with_name('rsrs_local_nav.png')}")


if __name__ == "__main__":
    main()
