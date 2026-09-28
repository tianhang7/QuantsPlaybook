"""下载基准 沪深300 指数日线 (腾讯 flashdata 历史文件) 并合并入 etf_close.csv.

格式: daily_data_YY="\\n\\nYYMMDD open close high low volume;";
"""

import re
import urllib.request
from pathlib import Path

import pandas as pd

YEARS = ["12", "13", "14", "15", "16", "17", "18", "19", "20"]


def fetch_year(yr: str) -> list:
    url = f"https://data.gtimg.cn/flashdata/hushen/daily/{yr}/sh000300.js"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        text = resp.read().decode("utf-8", "replace")

    match = re.search(r'daily_data_\d+="(.*)";', text, re.S)
    if not match:
        raise ValueError(f"unexpected payload for {yr}: {text[:80]!r}")

    rows = []
    for line in match.group(1).replace("\\n", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        yymmdd, _open, close = parts[0], parts[1], parts[2]
        rows.append((f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:]}", float(close)))
    return rows


def main() -> None:
    rows = []
    for yr in YEARS:
        year_rows = fetch_year(yr)
        print(f"20{yr}: {len(year_rows)} rows "
              f"{year_rows[0][0]} ~ {year_rows[-1][0]}", flush=True)
        rows.extend(year_rows)

    ser = pd.Series(dict(rows), name="000300")
    ser.index = pd.to_datetime(ser.index)
    ser = ser.sort_index()
    ser = ser[:"2020-01-16"]

    out = Path(__file__).resolve().parent.parent / "Data" / "etf_close.csv"
    px = pd.read_csv(out, index_col=0, parse_dates=True)
    px["000300"] = ser
    px = px.sort_index()
    px.to_csv(out, encoding="utf-8-sig")
    print(f"\n合并完成 {px.shape} {px.index[0].date()} ~ {px.index[-1].date()}")
    print("缺失统计:", {c: int(n) for c, n in px.isna().sum().items() if n})


if __name__ == "__main__":
    main()
