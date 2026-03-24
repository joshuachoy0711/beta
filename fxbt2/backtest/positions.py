from __future__ import annotations

"""
Position sizing methods.

Each function takes a signal DataFrame (direction: +1/0/-1 or continuous)
and returns a sized position DataFrame.

Returns-mode  → values are portfolio weight fractions (e.g. 0.05 = 5%)
Notional-mode → values are USD notional (e.g. 5_000_000)
The Backtest engine handles the distinction; sizers just scale the signal.
"""

import numpy as np
import pandas as pd


def fixed_size(signals: pd.DataFrame, size: float = 1.0) -> pd.DataFrame:
    """Fixed notional weight per active signal."""
    return np.sign(signals) * size


def vol_target(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 20,
    freq: str = "1D",
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """
    Volatility-targeting sizing.

    Scales each pair so its expected contribution = target_vol / n_pairs.
    size_i = (target_vol / n) / realised_vol_i

    Parameters
    ----------
    target_vol   : annualised portfolio volatility target (e.g. 0.10 = 10%)
    lookback     : rolling window for realised vol (periods)
    max_leverage : cap on per-pair size
    """
    from ..metrics.stats import _ann
    af = _ann(freq)
    n = signals.shape[1]
    per_pair = target_vol / n
    rv = prices.pct_change().rolling(lookback).std() * np.sqrt(af)
    rv = rv.replace(0, np.nan).ffill()
    size = (per_pair / rv).clip(upper=max_leverage / n)
    return np.sign(signals) * size


def var_target(
    signals: pd.DataFrame,
    pnl_series: pd.Series,
    var_window: int = 260,
    var_target_usd: float = 20_000,
    confidence: float = 0.05,
) -> pd.DataFrame:
    """
    VaR-targeting sizing. Ported from rcq run_ccy_pair_backtest_with_var_adj.

    Scales the entire portfolio so its rolling historical VaR equals var_target_usd.
    Used in notional mode (PnL in USD).

    Parameters
    ----------
    pnl_series     : daily USD PnL of the unscaled portfolio
    var_window     : rolling window for VaR estimate (trading days)
    var_target_usd : target daily VaR in USD (e.g. 20_000 = $20k)
    confidence     : VaR confidence level tail (0.05 = 95% VaR)

    Returns
    -------
    pd.DataFrame  signals scaled by the VaR adjustment factor
    """
    from scipy.stats import norm
    daily_pnl = pnl_series.diff()
    roll_var = pd.Series(
        norm.ppf(confidence,
                 daily_pnl.rolling(var_window).mean(),
                 daily_pnl.rolling(var_window).std()),
        index=pnl_series.index,
    )
    adj = (var_target_usd / roll_var.abs()).round(4)
    # Apply scalar multiplier to all pairs, shifted by 1 day
    return signals.multiply(adj.shift(1), axis=0)


def inverse_vol_weight(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    lookback: int = 22,
    rebalance_day: int = 0,
) -> pd.DataFrame:
    """
    Inverse volatility weighting across pairs. Ported from
    rcq rolling_inverse_volatility_adjust_on_mondays.

    Each pair is weighted proportionally to 1 / realised_vol so that
    low-vol pairs get more weight and high-vol pairs get less —
    before any portfolio-level risk overlay (e.g. var_target).

    Parameters
    ----------
    lookback       : rolling window for vol estimate (trading days). Default 22.
    rebalance_day  : day of week to rebalance (0=Monday). Default 0.
                     Pass None to rebalance daily.
    """
    ann_vol = prices.pct_change().rolling(lookback).std() * np.sqrt(260)
    ann_vol = ann_vol.replace(0, np.nan).shift(1).fillna(0)
    inv_vol = 1 / ann_vol.replace(0, np.nan)
    total   = inv_vol.sum(axis=1)
    weights = inv_vol.div(total, axis=0).round(6)

    if rebalance_day is not None:
        # Only update weights on the specified weekday (freeze otherwise)
        is_rebal = pd.Series(
            weights.index.dayofweek == rebalance_day, index=weights.index
        )
        weights = weights.where(is_rebal).ffill()

    return np.sign(signals) * weights


def kelly(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    lookback: int = 60,
    fraction: float = 0.5,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """
    Fractional Kelly sizing.
    Kelly = mean_return / variance   × fraction
    """
    ret = prices.pct_change()
    k = (ret.rolling(lookback).mean() / ret.rolling(lookback).var().replace(0, np.nan) * fraction)
    k = k.clip(-max_leverage, max_leverage)
    return np.sign(signals) * k.abs()


def equal_weight(signals: pd.DataFrame) -> pd.DataFrame:
    """Equal exposure across all active positions each period."""
    n_active = (signals != 0).sum(axis=1).replace(0, np.nan)
    return signals.div(n_active, axis=0).fillna(0.0)
