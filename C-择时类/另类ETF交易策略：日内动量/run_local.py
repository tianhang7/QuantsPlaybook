"""å‘½ä»¤è¡Œä½“éªŒï¼šå¦ç±»ETFäº¤æ˜“ç­–ç•¥â€”â€”æ—¥å†…åŠ¨é‡ã€‚

ç¤ºä¾‹ï¼š
python run_local.py --code 510300.SH --start 2023-01-01 --end 2023-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import backtrader as bt
import empyrical as ep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.SignalMaker import NoiseArea
from src.strategy import NoiseRangeStrategy, NoiseRangeVWAPStrategy

DATA = ROOT / "dataset" / "etf" / "hfq_etf_minute_price.parquet"
STRATEGIES = {
    "noise": NoiseRangeStrategy,
    "vwap": NoiseRangeVWAPStrategy,
}


class MinuteData(bt.feeds.PandasData):
    lines = ("upperbound", "signal", "lowerbound", "vwap")
    params = (
        ("upperbound", -1),
        ("signal", -1),
        ("lowerbound", -1),
        ("vwap", -1),
    )


class TradeStats(bt.Analyzer):
    def start(self):
        self.trades = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append(
                {
                    "opened": bt.num2date(trade.dtopen),
                    "closed": bt.num2date(trade.dtclose),
                    "pnl": trade.pnl,
                    "pnlcomm": trade.pnlcomm,
                    "size": trade.size,
                }
            )

    def get_analysis(self):
        return self.trades


def load_data(code: str, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        DATA,
        filters=[
            ("code", "=", code),
            ("trade_time", ">=", pd.Timestamp(start)),
            ("trade_time", "<=", pd.Timestamp(end) + pd.Timedelta(hours=15)),
        ],
    )
    if frame.empty:
        raise ValueError(f"æ²¡æœ‰æ‰¾åˆ° {code} åœ¨ {start} è‡³ {end} çš„åˆ†é’Ÿæ•°æ®")
    frame = frame.sort_values("trade_time").set_index("trade_time")
    if frame.index.has_duplicates:
        frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def main():
    parser = argparse.ArgumentParser(description="è¿è¡Œæ—¥å†…åŠ¨é‡ ETF å›žæµ‹")
    parser.add_argument("--code", default="510300.SH")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--strategy", choices=STRATEGIES, default="vwap")
    parser.add_argument("--window", type=int, default=14)
    parser.add_argument("--cash", type=float, default=1e8)
    args = parser.parse_args()

    print("1/4 è¯»å–åˆ†é’Ÿæ•°æ®...")
    raw = load_data(args.code, args.start, args.end)
    print(f"    {len(raw):,} æ ¹åˆ†é’ŸKçº¿: {raw.index.min()} ~ {raw.index.max()}")

    print("2/4 è®¡ç®—å™ªå£°åŒºé—´å’Œ VWAP ä¿¡å·...")
    # è‡³å°‘ä¿ç•™ window+1 ä¸ªäº¤æ˜“æ—¥ç”¨äºŽæ»šåŠ¨åˆ†ä½æ•°å’Œé¦–æ¬¡ä¿¡å·ã€‚
    signal = NoiseArea(raw.reset_index()).fit(window=args.window)
    signal = signal.dropna(subset=["upperbound", "lowerbound", "vwap"])
    signal = signal.sort_values("trade_time").set_index("trade_time")
    signal = signal.loc[str(raw.index.min().date()):str(raw.index.max().date())]
    print(f"    æœ‰æ•ˆä¿¡å·: {len(signal):,} æ ¹")

    print("3/4 è¿è¡Œ Backtrader...")
    cerebro = bt.Cerebro(stdstats=False)
    feed_data = signal.reset_index()
    feed_data["trade_time"] = pd.to_datetime(feed_data["trade_time"])

    feed = MinuteData(
        dataname=feed_data,
        datetime="trade_time",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        upperbound="upperbound",
        signal="signal",
        lowerbound="lowerbound",
    )
    cerebro.adddata(feed, name=args.code)
    cerebro.addstrategy(STRATEGIES[args.strategy], commission=0.001, hold_num=1, verbose=False)
    cerebro.broker.setcash(args.cash)
    cerebro.broker.addcommissioninfo(bt.CommInfoBase(commission=0.00015, stocklike=True))
    cerebro.broker.set_slippage_perc(0.0001)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
    cerebro.addanalyzer(TradeStats, _name="trade_stats")
    result = cerebro.run()[0]

    print("4/4 æ±‡æ€»ç»©æ•ˆ...")
    daily = pd.Series(result.analyzers.time_return.get_analysis()).dropna()
    benchmark_close = signal["close"].resample("D").last().dropna()
    benchmark = benchmark_close.pct_change().dropna()
    common = daily.index.intersection(benchmark.index)
    daily, benchmark = daily.loc[common], benchmark.loc[common]
    value = args.cash * (1 + daily).cumprod()
    stats = pd.Series({
        "ç´¯è®¡æ”¶ç›Š": ep.cum_returns_final(daily),
        "å¹´åŒ–æ”¶ç›Š": ep.annual_return(daily),
        "å¹´åŒ–æ³¢åŠ¨": ep.annual_volatility(daily),
        "å¤æ™®æ¯”çŽ‡": ep.sharpe_ratio(daily),
        "æœ€å¤§å›žæ’¤": ep.max_drawdown(daily),
    })
    analysis = result.analyzers.trade_analyzer.get_analysis()
    trades = result.analyzers.trade_stats.get_analysis()
    trade_df = pd.DataFrame(trades)
    print("\nå›žæµ‹ç»©æ•ˆ")
    print(stats.to_string(float_format=lambda x: f"{x:.4%}"))
    print(f"\næ€»äº¤æ˜“: {analysis.get('total', {}).get('closed', len(trades))}")
    if not trade_df.empty:
        print(f"èƒœçŽ‡: {(trade_df.pnlcomm > 0).mean():.2%}")
        print(f"å¹³å‡å‡€ç›ˆäº: {trade_df.pnlcomm.mean():,.2f}")
        print(f"æœ€ä½³/æœ€å·®: {trade_df.pnlcomm.max():,.2f} / {trade_df.pnlcomm.min():,.2f}")

    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    suffix = f"{args.code}_{args.strategy}_{args.start}_{args.end}"
    stats.to_csv(output_dir / f"{suffix}_metrics.csv", header=["value"])
    if not trade_df.empty:
        trade_df.to_csv(output_dir / f"{suffix}_trades.csv", index=False, encoding="utf-8-sig")
    pd.concat([value.rename("strategy"), args.cash * (1 + benchmark).cumprod().rename("buy_hold")], axis=1).to_csv(output_dir / f"{suffix}_nav.csv", encoding="utf-8-sig")
    ax = value.div(args.cash).rename("Strategy").plot(figsize=(11, 5), legend=True)
    (args.cash * (1 + benchmark).cumprod()).div(args.cash).rename("Buy & Hold").plot(ax=ax, legend=True)
    ax.set(title=f"{args.code} {args.strategy} daily momentum", ylabel="NAV")
    plt.tight_layout()
    plt.savefig(output_dir / f"{suffix}_nav.png", dpi=150)
    plt.close()
    print(f"\nç»“æžœå·²ä¿å­˜åˆ°: {output_dir}")


if __name__ == "__main__":
    main()
