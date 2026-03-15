from __future__ import annotations

import pandas as pd

from src.infrastructure.backtesting import _with_market_forward_return


def test_with_market_forward_return_keeps_single_market_forward_column() -> None:
    market_raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "open": [10.0, 10.2, 10.1, 10.4],
            "high": [10.3, 10.4, 10.5, 10.6],
            "low": [9.9, 10.0, 10.0, 10.2],
            "close": [10.1, 10.3, 10.4, 10.5],
            "volume": [1000, 1200, 1100, 1300],
            "amount": [10000, 12000, 11400, 13600],
        }
    )

    out = _with_market_forward_return(market_raw)

    assert "mkt_fwd_ret_1" in out.columns
    assert "mkt_fwd_ret_1_x" not in out.columns
    assert "mkt_fwd_ret_1_y" not in out.columns
