"""本地运行 Trader-Company 集成算法交易策略.

完整复现 Trader_Company.ipynb 的回测流程:
    - 数据: Data/data.csv, 6 个指数日收益率 (2010-05-31 ~ 2022-07-29)
    - 目标: 000300.SH, without_target=True (特征含全部 6 列)
    - Company: 100 traders, M=10, max_lag=9, l=1, time_window=100, Q=0.5
    - evaluation=ACC, aggregate=Q (前 50% 交易员预测均值)
    - 三种 generate 方法: BayesianGaussianMixture / GaussianMixture / Gaussian

与 Notebook 的差异 (已在输出中说明):
    - tqdm.notebook 进度条替换为标准 tqdm (非 Jupyter 环境)
    - 默认固定随机种子 seed=42 (Notebook 为 None), 可用 --no-seed 关闭;
      注意多进程 worker 各自持有 RNG, 结果仍有跨进程随机性
    - 输出绩效指标与图表到 run_local_output/

用法:
    python run_local.py                                  # 三种方法全量
    python run_local.py --methods Gaussian               # 单个方法
    python run_local.py --trader-num 20 --max-days 400   # 冒烟测试
"""

import argparse
import sys
import time
from pathlib import Path

# Windows 重定向到文件时默认 cp1252, 中文打印会崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import tqdm.notebook as _tqdm_notebook
import tqdm.std as _tqdm_std

# 非 Jupyter 环境下 tqdm.notebook 不可用, 先替换再导入 scr.TC
_tqdm_notebook.tqdm = _tqdm_std.tqdm

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import empyrical as ep
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scr.activation_funcs import ReLU, identity, sign, tanh
from scr.binary_operators import (
    get_x,
    get_y,
    operators_add,
    operators_diff,
    operators_max,
    operators_min,
    operators_multiple,
    x_is_greater_than_y,
)
from scr.TC import Company

ALL_METHODS = ["BayesianGaussianMixture", "GaussianMixture", "Gaussian"]

ACTIVATION_FUNCS = [identity, ReLU, sign, tanh]
BINARY_OPERATORS = [
    operators_max,
    operators_min,
    operators_add,
    operators_diff,
    get_x,
    get_y,
    operators_multiple,
    x_is_greater_than_y,
]

TARGET_NAME = "000300.SH"


def set_plot_font() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def load_price() -> pd.DataFrame:
    path = PROJECT_DIR / "Data" / "data.csv"
    price = pd.read_csv(path, index_col=[0], parse_dates=[0])
    return price


def get_backtesting(params: dict, data: pd.DataFrame, target_name: str,
                    without_target: bool = True) -> pd.DataFrame:
    """复现 Notebook 的 get_backtesting: 拟合并生成策略/基准日收益."""
    codes = data.columns.tolist()
    test_col = codes if without_target else [i for i in codes if i != target_name]

    train_data = data[test_col]
    target = data[target_name]

    company = Company(**params)
    company.fit(train_data.values, target.values)

    time_window = params["time_window"] - 1
    benchmark = target.iloc[time_window:]

    pred_sign = np.where(company.aggregate > 0, 1, 0)
    strategy_returns = benchmark.shift(-1) * pred_sign

    df = pd.concat((benchmark, strategy_returns), axis=1)
    df.columns = ["benchmark", target_name]
    return df

def calc_metrics(returns: pd.Series, name: str) -> dict:
    returns = returns.dropna()
    return {
        "name": name,
        "total_return": float(ep.cum_returns_final(returns)),
        "annual_return": float(ep.annual_return(returns)),
        "annual_volatility": float(ep.annual_volatility(returns)),
        "sharpe": float(ep.sharpe_ratio(returns)),
        "max_drawdown": float(ep.max_drawdown(returns)),
    }


def hit_rate(df: pd.DataFrame, col: str) -> float:
    """策略在持仓日的方向命中率 (相对次日基准收益)."""
    sub = df[[col, "benchmark"]].dropna()
    invested = sub[sub[col] != 0]
    if invested.empty:
        return float("nan")
    return float((np.sign(invested[col]) == np.sign(invested["benchmark"])).mean())


def build_params(args: argparse.Namespace, stock_num: int, method: str) -> dict:
    seed = None if args.no_seed else args.seed
    return {
        "trader_num": args.trader_num,
        "A": ACTIVATION_FUNCS,
        "O": BINARY_OPERATORS,
        "stock_num": stock_num - 1,  # 与 Notebook 一致: len(codes) - 1
        "M": 10,
        "max_lag": 9,
        "l": 1,
        "time_window": 100,
        "Q": 0.5,
        "generate_method": method,
        "evaluation_method": "ACC",
        "aggregate_method": "Q",
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trader-Company 本地回测")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--trader-num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-seed", action="store_true", help="与 Notebook 一致, 不设随机种子")
    parser.add_argument("--max-days", type=int, default=None, help="仅取前 N 行, 用于冒烟测试")
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "run_local_output"))
    args = parser.parse_args()

    set_plot_font()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    price = load_price()
    if args.max_days:
        price = price.iloc[: args.max_days]
    stock_num = len(price.columns)
    print(f"数据: {price.shape[0]} 天 x {stock_num} 指数 "
          f"({price.index[0].date()} ~ {price.index[-1].date()})", flush=True)

    results = {}
    metrics_rows = []
    for method in args.methods:
        params = build_params(args, stock_num, method)
        print(f"\n===== 运行 {method} =====", flush=True)
        start = time.time()
        df = get_backtesting(params, price, TARGET_NAME, without_target=True)
        elapsed = time.time() - start

        results[method] = df
        bench_m = calc_metrics(df["benchmark"], "benchmark")
        strat_m = calc_metrics(df[TARGET_NAME], method)
        strat_m["hit_rate"] = hit_rate(df, TARGET_NAME)
        strat_m["days_in_market"] = float((df[TARGET_NAME].dropna() != 0).mean())
        strat_m["elapsed_sec"] = round(elapsed, 1)
        metrics_rows.extend([bench_m, strat_m])

        print(f"{method}: 耗时 {elapsed:.1f}s | 策略累计 "
              f"{strat_m['total_return']:.2%} | 基准累计 {bench_m['total_return']:.2%} | "
              f"夏普 {strat_m['sharpe']:.2f} | 最大回撤 {strat_m['max_drawdown']:.2%} | "
              f"命中率 {strat_m['hit_rate']:.2%}", flush=True)

        df.to_csv(out_dir / f"returns_{method}.csv", encoding="utf-8-sig")

    # 绩效汇总
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    print("\n===== 绩效汇总 =====")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    # 累计收益图
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(16, 5 * n), squeeze=False)
    for ax, (method, df) in zip(axes[:, 0], results.items()):
        cum = ep.cum_returns(df)
        cum["benchmark"].plot(ax=ax, color="darkgray", label="benchmark 000300.SH")
        cum[TARGET_NAME].plot(ax=ax, color="red", label=f"TC ({method})")
        ax.set_title(f"Trader-Company {method} | seed={args.seed if not args.no_seed else 'None'} "
                     f"| traders={args.trader_num}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "cum_returns.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\n输出目录: {out_dir}\n图表: {fig_path}")


if __name__ == "__main__":
    main()

