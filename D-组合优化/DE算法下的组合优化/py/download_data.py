"""下载 DE 组合优化所需的 ETF 日线数据 (Yahoo Finance, 复权收盘价).

标的池一 (Notebook etfList):    510050 510500 510300 159915 510180 159901 159902
标的池二 (Notebook etfList2):   池一 + 510230 159928 510880 159938 159902 512070
                                + 159939 159905 159910  (去掉重复)
基准: 000300.XSHG -> Yahoo 000300.SS

输出: Data/etf_close.csv (index=date, columns=代码, 前复权收盘价)
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"

POOL1 = ["510050", "510500", "510300", "159915", "510180", "159901", "159902"]
POOL2 = POOL1 + ["510230", "159928", "510880", "159938", "512070", "159939",
                 "159905", "159910"]
ALL_CODES = list(dict.fromkeys(POOL1 + POOL2))
BENCHMARK = "000300"

# 覆盖 2014-01 起 126 日回看窗口: 需要自 2012-06 起
PERIOD1 = 1338508800  # 2012-06-01 UTC
PERIOD2 = 1579142400  # 2020-01-16 UTC


def yahoo_symbol(code: str) -> str:
    if code.startswith(("51", "60", "000300")) and len(code) == 6:
        return f"{code}.SS"
    return f"{code}.SZ"


def fetch_yahoo(symbol: str) -> pd.Series:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?period1={PERIOD1}&period2={PERIOD2}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.load(resp)

    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"empty result for {symbol}")

    result = result[0]
    ts = result.get("timestamp", [])
    adj = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
    if adj is None:  # 个别标的无 adjclose, 回退 close
        adj = result.get("indicators", {}).get("quote", [{}])[0].get("close")
    if not ts or adj is None:
        raise ValueError(f"no data for {symbol}")

    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Shanghai").date
    ser = pd.Series(adj, index=pd.DatetimeIndex(idx), name=symbol)
    return ser[~ser.index.duplicated(keep="last")].dropna()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "etf_close.csv"

    existing = pd.read_csv(out_path, index_col=0, parse_dates=True) \
        if out_path.exists() else pd.DataFrame()

    series = []
    for code in ALL_CODES:
        symbol = yahoo_symbol(code)
        for attempt in range(3):
            try:
                ser = fetch_yahoo(symbol)
                print(f"{code:8s} -> {symbol:12s} {len(ser):5d} rows "
                      f"{ser.index[0].date()} ~ {ser.index[-1].date()}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    raise RuntimeError(f"下载失败: {code} ({symbol})") from exc
                time.sleep(2 * (attempt + 1))
        series.append(ser.rename(code))
        time.sleep(0.8)

    close = pd.concat(series, axis=1).sort_index()
    close.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n保存 {out_path} shape={close.shape} "
          f"{close.index[0].date()} ~ {close.index[-1].date()}", flush=True)
    missing = close.isna().sum()
    print("缺失统计:", dict(missing[missing > 0]) or "无缺失")
    print("注: 基准 000300 由 _fetch_benchmark.py 单独下载合并")



if __name__ == "__main__":
    sys.exit(main())
