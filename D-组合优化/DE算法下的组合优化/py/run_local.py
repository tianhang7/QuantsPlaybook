"""本地运行 DE 差分进化算法组合优化 (复现 差分进化算法.ipynb).

复现内容:
    - 标的一: 7 只宽基/风格 ETF; 标的二: 15 只 (含行业/主题)
    - 调仓频率: 季度; 每期用观察日前 126 根日线优化权重
    - 观察日 = 换仓日前 2 个交易日 (避免未来数据, 与 Notebook 一致)
    - 目标函数: MD 最大回撤最小化 / CVaR 预期损失最小化
    - DE 参数: size_pop=150, max_iter=200, F=0.5, prob_mut=1
      (Notebook 网格搜索结论, 搜索本身耗时 1h+ 故跳过)
    - 回测: 每期持权重至下一换仓日, 收益滞后一期 (shift(-1)) 防未来数据

与 Notebook 的差异:
    - jqdata (聚宽) -> 本地 Data/etf_close.csv (Yahoo 复权价 + 腾讯指数)
    - tdaysoffset 基于本地交易日历 (000300 交易日并集)
    - 未上市标的: 权重在当日可得标的间重新归一化 (原版为 NaN 传染)
    - 输出绩效表/图表到 run_local_output/
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PY_DIR = Path(__file__).resolve().parent
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

import empyrical as ep
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DE_algorithm import DE

POOL1 = ["510050", "510500", "510300", "159915", "510180", "159901", "159902"]
POOL2 = POOL1 + ["510230", "159928", "510880", "159938", "512070", "159939",
                 "159905", "159910"]
BENCHMARK = "000300"

START_DT, END_DT = "2014-01-01", "2019-12-31"
LOOKBACK = 126  # get_price(..., count=126)


# ---------- 交易日历与区间 ----------

def load_close() -> pd.DataFrame:
    path = PY_DIR.parent / "Data" / "etf_close.csv"
    close = pd.read_csv(path, index_col=0, parse_dates=True)
    return close.sort_index()


CALENDAR: pd.DatetimeIndex = None  # 由 init_calendar 设置


def init_calendar(close: pd.DataFrame) -> None:
    global CALENDAR
    idx = close[BENCHMARK].dropna().index
    CALENDAR = pd.DatetimeIndex(sorted(idx))


def tdaysoffset(date, count: int):
    """交易日偏移, 语义与聚宽 get_trade_days/tdaysoffset 一致."""
    if isinstance(date, str):
        date = pd.Timestamp(date)
    pos = CALENDAR.searchsorted(date, side="right") - 1  # <= date 最近交易日
    if pos < 0:
        pos = 0
    if count > 0:
        return CALENDAR[min(pos + count, len(CALENDAR) - 1)]
    if count < 0:
        return CALENDAR[max(pos + count, 0)]
    raise ValueError("别闹！")


def build_periods(start: str, end: str) -> Tuple[List, List]:
    """季度区间: 与 Notebook freq='Q' (季末) 一致."""
    try:
        periods = pd.date_range(start, end, freq="QE")
    except ValueError:
        periods = pd.date_range(start, end, freq="Q")
    periods = [pd.Timestamp(p) for p in periods]
    return periods[:-1], periods[1:]


# ---------- 目标函数 (与 Notebook 相同) ----------

def maxDrawdown(pct_chg: pd.DataFrame, w: np.ndarray) -> float:
    return -ep.max_drawdown(pct_chg @ w)


def calcCVaR(returns: pd.DataFrame, w: np.ndarray, alpha: float = 0.05) -> float:
    N = len(returns)
    p_mean = (returns @ w).mean()
    p_std = np.sqrt((returns @ w).var())
    p_std_h = p_std * np.sqrt(N / 252)
    return float(alpha ** (-1) * st.norm.pdf(st.norm.ppf(alpha)) * p_std_h - p_mean)


# ---------- sklearn 风格封装 (与 Notebook 相同) ----------

class DEPortfolioOpt:
    def __init__(self, func, size_pop: int, max_iter: int, F: float,
                 proub_mut: float) -> None:
        self.func = func
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.F = F
        self.proub_mut = proub_mut

    def fit(self, returns: pd.DataFrame) -> pd.Series:
        self.de = DE(func=lambda w: self.func(returns, w),
                     n_dim=returns.shape[1],
                     size_pop=self.size_pop,
                     max_iter=self.max_iter,
                     F=self.F,
                     prob_mut=self.proub_mut,
                     lb=[0],
                     ub=[1],
                     constraint_eq=[lambda x: 1 - np.sum(x)])
        w, _ = self.de.run()
        return pd.Series(w, index=returns.columns)


def get_weight(opt: DEPortfolioOpt, codes: List[str], watch_date,
               N: int) -> pd.Series:
    """watch_date 末尾 N 根收盘 -> 收益率 -> DE 优化权重.

    未上市(全 NaN)列剔除; 零星停牌 NaN 以 0 填充.
    """
    close = CLOSE.loc[:watch_date, codes].iloc[-N:]
    pct_df = close.pct_change().iloc[1:]
    pct_df = pct_df.dropna(axis=1, how="all").fillna(0.0)
    if pct_df.empty:
        return pd.Series(1.0 / len(codes), index=codes)
    w = opt.fit(pct_df)
    full = pd.Series(0.0, index=codes)
    full[w.index] = w
    return full / full.sum()


# ---------- 回测引擎 ----------

def back_testing(time_range: Tuple[List, List],
                 weights: Dict,
                 close: pd.DataFrame) -> pd.DataFrame:
    begin, end = min(time_range[0]), max(time_range[1])
    idx = CALENDAR[(CALENDAR >= begin) & (CALENDAR <= tdaysoffset(end, 1))]
    ret_df = pd.DataFrame(0.0, index=idx, columns=["opt_ret", "equal_ret"])

    for begin_dt, next_dt in zip(*time_range):
        w = weights[begin_dt]
        end_dt = tdaysoffset(next_dt, 1)
        seg_idx = idx[(idx >= begin_dt) & (idx <= end_dt)]
        if len(seg_idx) < 2:
            continue
        seg = close.loc[seg_idx, w.index]
        pct_df = seg.pct_change().shift(-1).iloc[:-1]  # 收益滞后一期

        arr = pct_df.to_numpy()
        wv = w.to_numpy()
        avail = ~np.isnan(arr)
        weighted = np.where(avail, wv, 0.0)
        denom = weighted.sum(axis=1)
        denom[denom <= 0] = 1.0
        port = (np.nan_to_num(arr) * weighted / denom[:, None]).sum(axis=1)

        ret_df.loc[pct_df.index, "opt_ret"] = port
        ret_df.loc[pct_df.index, "equal_ret"] = pct_df.mean(axis=1).to_numpy()

    ret_df["benchmark"] = close[BENCHMARK].pct_change().reindex(ret_df.index)
    return ret_df



# ---------- 绩效与绘图 ----------

def strategy_performance(return_df: pd.DataFrame) -> pd.DataFrame:
    """复现 Notebook Strategy_performance (daily)."""
    ser = pd.DataFrame()
    ser["年化收益率"] = ep.annual_return(return_df, period="daily")
    ser["波动率"] = return_df.apply(lambda x: ep.annual_volatility(x, period="daily"))
    ser["夏普"] = return_df.apply(lambda x: ep.sharpe_ratio(x, period="daily"))
    ser["最大回撤"] = return_df.apply(ep.max_drawdown)
    if "benchmark" in return_df.columns:
        cols = [c for c in return_df.columns if c != "benchmark"]
        bench = return_df["benchmark"]
        ser["IR"] = return_df[cols].apply(lambda x: _information_ratio(x, bench))
        ser["Alpha"] = return_df[cols].apply(lambda x: ep.alpha(x, bench, period="daily"))
    return ser.T


def _information_ratio(returns: pd.Series, factor_returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    active = returns - factor_returns
    te = np.std(active, ddof=1)
    if np.isnan(te):
        return 0.0
    if te == 0:
        return np.nan
    return float(np.mean(active) / te)


def set_plot_style() -> None:
    for style in ("seaborn-v0_8", "ggplot"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue
    # style 会重置 font 配置, 必须在 style 之后设置
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


SCENARIOS = [("MD", "pool1", maxDrawdown, POOL1),
             ("MD", "pool2", maxDrawdown, POOL2),
             ("CVaR", "pool1", calcCVaR, POOL1),
             ("CVaR", "pool2", calcCVaR, POOL2)]


def main() -> None:
    parser = argparse.ArgumentParser(description="DE 组合优化本地回测")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size-pop", type=int, default=150)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--scenarios", nargs="+",
                        default=[f"{a}-{b}" for a, b, _, _ in SCENARIOS],
                        choices=[f"{a}-{b}" for a, b, _, _ in SCENARIOS])
    parser.add_argument("--out-dir", default=str(PY_DIR / "run_local_output"))
    args = parser.parse_args()

    set_plot_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global CLOSE
    CLOSE = load_close()
    init_calendar(CLOSE)
    time_range = build_periods(START_DT, END_DT)
    print(f"数据: {CLOSE.shape[0]} 天 x {CLOSE.shape[1]} 列 "
          f"({CLOSE.index[0].date()} ~ {CLOSE.index[-1].date()}) | "
          f"调仓期数: {len(time_range[0])} "
          f"({time_range[0][0].date()} ~ {time_range[1][-1].date()})", flush=True)

    all_metrics = []
    results: Dict[str, pd.DataFrame] = {}
    for obj_name, pool_name, obj_func, pool in SCENARIOS:
        key = f"{obj_name}-{pool_name}"
        if key not in args.scenarios:
            continue
        print(f"\n===== {obj_name} | {pool_name} ({len(pool)} ETFs) =====", flush=True)
        np.random.seed(args.seed)
        start = time.time()

        weights = {}
        for trade_dt in time_range[0]:
            watch = tdaysoffset(trade_dt, -2)
            opt = DEPortfolioOpt(obj_func, args.size_pop, args.max_iter, 0.5, 1)
            weights[trade_dt] = get_weight(opt, pool, watch, LOOKBACK)

        ret_df = back_testing(time_range, weights, CLOSE)
        elapsed = time.time() - start
        results[key] = ret_df
        ret_df.to_csv(out_dir / f"returns_{key}.csv", encoding="utf-8-sig")
        w_df = pd.DataFrame(weights).T
        w_df.index.name = "trade_dt"
        w_df.to_csv(out_dir / f"weights_{key}.csv", encoding="utf-8-sig")

        perf = strategy_performance(ret_df)
        all_metrics.append(perf)
        print(f"耗时 {elapsed:.1f}s | opt 累计 {ep.cum_returns_final(ret_df['opt_ret']):.2%} "
              f"| equal {ep.cum_returns_final(ret_df['equal_ret']):.2%} "
              f"| benchmark {ep.cum_returns_final(ret_df['benchmark'].dropna()):.2%}",
              flush=True)

        # 每期回撤对比 (Notebook cell 24)
        dd = pd.DataFrame(index=pd.to_datetime(time_range[0]),
                          columns=["opt_maxDrawDown", "equal_maxDrawDown"])
        for s, e in zip(*time_range):
            dd.loc[s, "opt_maxDrawDown"] = -ep.max_drawdown(ret_df.loc[s:e, "opt_ret"])
            dd.loc[s, "equal_maxDrawDown"] = -ep.max_drawdown(ret_df.loc[s:e, "equal_ret"])
        dd = dd.astype(float)
        ax = dd.plot.bar(y=["opt_maxDrawDown", "equal_maxDrawDown"],
                         color=["r", "darkgray"], title=f"优化效果 {key}",
                         figsize=(12, 5))
        ax.set_ylabel("max drawdown")
        ax.figure.tight_layout()
        ax.figure.savefig(out_dir / f"drawdown_bar_{key}.png", dpi=110)
        plt.close(ax.figure)

    # 绩效汇总 (跨场景按列拼接, 行与 Notebook 相同)
    metrics = pd.concat(all_metrics, axis=1)
    metrics.to_csv(out_dir / "metrics.csv", encoding="utf-8-sig")
    print("\n===== 绩效汇总 =====")
    print(metrics.to_string(float_format=lambda x: f"{x:,.4f}"))

    # 累计收益图: 每场景一子图
    fig, axes = plt.subplots(len(results), 1, figsize=(16, 4.5 * len(results)),
                             squeeze=False)
    for ax, (key, ret_df) in zip(axes[:, 0], results.items()):
        cum = 1 + ep.cum_returns(ret_df)
        cum["benchmark"].plot(ax=ax, color="darkgray", label="沪深300")
        cum["equal_ret"].plot(ax=ax, color="#6891BD", label="等权")
        cum["opt_ret"].plot(ax=ax, color="r", label=f"DE {key}")
        ax.set_title(f"DE组合优化 {key} (F=0.5, prob_mut=1, pop=150, iter=200)")
        ax.set_ylabel("cumulative return")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cum_returns.png", dpi=120)
    plt.close(fig)
    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()

