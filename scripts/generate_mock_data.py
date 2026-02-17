#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_series(n: int, seed: int, drift: float, vol: float, start_price: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = drift + vol * rng.standard_normal(n)
    close = start_price * np.exp(np.cumsum(ret))
    open_ = close * (1.0 + 0.002 * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1.0 + np.abs(0.006 * rng.standard_normal(n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(0.006 * rng.standard_normal(n)))
    volume = np.exp(13 + 0.7 * rng.standard_normal(n))
    amount = close * volume
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
        }
    )


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2018-01-01", "2026-02-17")
    n = len(dates)

    specs = {
        "000300.SH": (1, 0.00018, 0.012, 3800.0),
        "002772.SZ": (2, 0.00025, 0.020, 16.0),
        "603516.SH": (3, 0.00020, 0.019, 36.0),
        "600160.SH": (4, 0.00017, 0.017, 22.0),
        "000630.SZ": (5, 0.00016, 0.022, 4.0),
        "603619.SH": (6, 0.00023, 0.024, 28.0),
    }

    for symbol, (seed, drift, vol, price) in specs.items():
        df = make_series(n=n, seed=seed, drift=drift, vol=vol, start_price=price)
        df.insert(0, "date", dates)
        df.to_csv(out_dir / f"{symbol}.csv", index=False)

    print(f"Mock data written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

