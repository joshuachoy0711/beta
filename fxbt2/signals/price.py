from __future__ import annotations

"""
Price-based signal generators.

All functions:
  Input  → wide-format prices DataFrame (index=DatetimeIndex, columns=pairs)
  Output → same-shape DataFrame, values in {+1, 0, -1} or continuous
  Rule   → .shift(1) applied at the end of every function (no look-ahead)
"""

import numpy as np
import pandas as pd


def momentum(prices: pd.DataFrame, lookback: int = 20,
             signal_type: str = "sign") -> pd.DataFrame:
    """
    Time-series momentum.

    signal_type:
        'sign'   → +1/-1 based on sign of N-period return
        'zscore' → z-score of returns (continuous, normalised)
        'rank'   → cross-sectional percentile rank, centred at 0
    """
    ret = prices.pct_change(lookback)
    if signal_type == "sign":
        sig = np.sign(ret)
    elif signal_type == "zscore":
        sig = ret.sub(ret.rolling(lookback * 3).mean()).div(ret.rolling(lookback * 3).std())
    elif signal_type == "rank":
        sig = ret.rank(axis=1, pct=True) - 0.5
    else:
        raise ValueError(f"signal_type must be 'sign', 'zscore', or 'rank'. Got '{signal_type}'.")
    return sig.shift(1)


def crossover(prices: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.DataFrame:
    """Moving average crossover: +1 when fast MA > slow MA."""
    return np.sign(prices.rolling(fast).mean() - prices.rolling(slow).mean()).shift(1)


def macd_signal(prices: pd.DataFrame, fast: int = 12, slow: int = 26,
                signal_period: int = 9) -> pd.DataFrame:
    """MACD histogram crossover: +1 when MACD > signal line."""
    macd = prices.ewm(span=fast, adjust=False).mean() - prices.ewm(span=slow, adjust=False).mean()
    return np.sign(macd - macd.ewm(span=signal_period, adjust=False).mean()).shift(1)


def breakout(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Donchian channel breakout: +1 above rolling high, -1 below rolling low."""
    roll_high = prices.shift(1).rolling(lookback).max()
    roll_low  = prices.shift(1).rolling(lookback).min()
    sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    sig[prices > roll_high] = 1.0
    sig[prices < roll_low]  = -1.0
    return sig


def mean_reversion(prices: pd.DataFrame, lookback: int = 20,
                   z_entry: float = 1.5, z_exit: float = 0.5) -> pd.DataFrame:
    """
    Z-score mean reversion.
    +1 when z < -z_entry (below mean → expect move up)
    -1 when z >  z_entry (above mean → expect move down)
     0 when |z| < z_exit (near mean → flat)
    """
    roll_mean = prices.rolling(lookback).mean()
    roll_std  = prices.rolling(lookback).std()
    z = (prices - roll_mean) / roll_std
    sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    sig[z < -z_entry] =  1.0
    sig[z >  z_entry] = -1.0
    sig[z.abs() < z_exit] = 0.0
    return sig.shift(1)


def rsi_signal(prices: pd.DataFrame, period: int = 14,
               overbought: float = 70.0, oversold: float = 30.0) -> pd.DataFrame:
    """RSI mean-reversion: +1 when oversold, -1 when overbought."""
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    sig   = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    sig[rsi < oversold]   =  1.0
    sig[rsi > overbought] = -1.0
    return sig.shift(1)


def vol_regime(prices: pd.DataFrame, lookback: int = 20,
               high_vol_percentile: float = 75.0) -> pd.DataFrame:
    """
    Volatility regime filter.
    Returns 1 in normal vol regime (allow trades), 0 in high vol (suppress).
    Multiply against another signal: filtered = momentum_sig * vol_regime(prices)
    """
    rv = prices.pct_change().rolling(lookback).std()
    threshold = rv.quantile(high_vol_percentile / 100, axis=0)
    return (rv <= threshold).astype(float).shift(1)


def session_filter(prices: pd.DataFrame, session: str = "london") -> pd.DataFrame:
    """
    Trading session mask (1 = active, 0 = inactive). Intraday data only.
    Requires UTC-aware DatetimeIndex.

    Sessions (UTC): london 08-17, new_york 13-22, tokyo 23-08, overlap 13-17
    """
    _sessions = {"london": (8, 17), "new_york": (13, 22),
                 "overlap": (13, 17), "tokyo": None}
    if session not in _sessions:
        raise ValueError(f"session must be one of {list(_sessions.keys())}")
    hour = prices.index.hour
    mask = (hour >= 23) | (hour < 8) if session == "tokyo" else \
           (hour >= _sessions[session][0]) & (hour < _sessions[session][1])
    filt = pd.Series(mask.astype(float), index=prices.index)
    return pd.DataFrame(
        np.tile(filt.values.reshape(-1, 1), (1, len(prices.columns))),
        index=prices.index, columns=prices.columns,
    )
