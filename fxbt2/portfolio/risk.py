from __future__ import annotations

"""
Portfolio risk metrics.
Ported from rcq_trading_library.roll_var, extended with risk attribution.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def rolling_var(
    pnl: pd.Series,
    window: int = 260,
    confidence: float = 0.05,
) -> pd.Series:
    """
    Rolling historical VaR (parametric, normal distribution assumption).
    Ported from rcq_trading_library.roll_var.

    Parameters
    ----------
    pnl        : pd.Series  daily PnL (returns or notional)
    window     : int        rolling window in trading days
    confidence : float      tail probability (0.05 = 95% VaR)

    Returns
    -------
    pd.Series of VaR values (negative = loss)
    """
    daily = pnl.diff()
    roll_mean = daily.rolling(window).mean()
    roll_std  = daily.rolling(window).std()
    return pd.Series(
        norm.ppf(confidence, roll_mean, roll_std),
        index=pnl.index,
        name="VaR",
    )


def risk_attribution(
    pair_returns: pd.DataFrame,
    freq: str = "1D",
) -> pd.DataFrame:
    """
    Decompose portfolio risk into per-pair contributions.

    Returns a DataFrame with:
      - ann_vol        : annualised volatility per pair
      - pct_vol_contrib: each pair's share of total portfolio variance
      - sharpe         : per-pair Sharpe ratio
      - total_return   : cumulative return contribution

    Parameters
    ----------
    pair_returns : pd.DataFrame  per-pair net returns (from BacktestResult.pair_returns)
    freq         : str           return frequency for annualisation
    """
    from ..metrics.stats import _ann, sharpe
    af = _ann(freq)

    portfolio = pair_returns.sum(axis=1)
    cov = pair_returns.cov() * af

    ann_vol = pair_returns.std() * np.sqrt(af)
    port_var = float(portfolio.var() * af)

    # Marginal contribution to variance = cov(pair, portfolio) / port_var
    cov_with_port = pair_returns.apply(lambda col: col.cov(portfolio) * af)
    pct_contrib = cov_with_port / port_var if port_var > 0 else cov_with_port * 0

    sharpes = pair_returns.apply(lambda col: sharpe(col, freq=freq))
    total_ret = pair_returns.sum()

    return pd.DataFrame({
        "ann_vol":         ann_vol,
        "pct_vol_contrib": pct_contrib,
        "sharpe":          sharpes,
        "total_return":    total_ret,
    }).round(4)
