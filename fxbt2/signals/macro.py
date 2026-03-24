from __future__ import annotations

"""
Macro / fundamental signal generators.
Ported and unified from rcq_trading_library (yield signals).

All functions return wide-format signal DataFrames (+1 / 0 / -1 or continuous).
All are forward-safe (.shift(1) applied).
"""

import numpy as np
import pandas as pd


def carry_fwd(
    fwd_points: pd.DataFrame,
    spot_prices: pd.DataFrame,
    signal_type: str = "sign",
) -> pd.DataFrame:
    """
    Carry signal from 1M forward points.

    carry ≈ -(fwd_points / spot) * 12   (annualised)

    Positive carry → base currency at forward discount → long base.

    Parameters
    ----------
    fwd_points   : wide DataFrame of 1M forward points
    spot_prices  : wide DataFrame of spot mid prices
    signal_type  : 'sign' → +1/-1,  'zscore' → continuous normalised
    """
    carry = -(fwd_points / spot_prices) * 12

    if signal_type == "sign":
        sig = np.sign(carry)
    elif signal_type == "zscore":
        roll_mean = carry.rolling(60).mean()
        roll_std  = carry.rolling(60).std()
        sig = (carry - roll_mean) / roll_std
    else:
        raise ValueError(f"signal_type must be 'sign' or 'zscore'. Got '{signal_type}'.")

    return sig.shift(1)


def carry_yield(
    impl_yield: pd.DataFrame,
    signal_type: str = "sign",
) -> pd.DataFrame:
    """
    Carry signal from implied yield differentials.

    Uses the annualised 1M implied yield differential directly
    (base_ccy_yield - quote_ccy_yield), computed by ForwardBuilder.build_implied_yield().

    This is more accurate than carry_fwd for EM / NDF pairs where
    forward points reflect credit risk as well as carry.

    Parameters
    ----------
    impl_yield   : wide DataFrame of annualised yield differentials per pair
    signal_type  : 'sign' → +1/-1,  'zscore' → continuous normalised
    """
    if signal_type == "sign":
        sig = np.sign(impl_yield)
    elif signal_type == "zscore":
        roll_mean = impl_yield.rolling(60).mean()
        roll_std  = impl_yield.rolling(60).std()
        sig = (impl_yield - roll_mean) / roll_std
    else:
        raise ValueError(f"signal_type must be 'sign' or 'zscore'. Got '{signal_type}'.")

    return sig.shift(1)


def yield_range(
    impl_yield: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Yield range / mean-reversion signal. Ported from rcq daily_nom_yield_range.

    Logic:
      - Compute rolling percentile rank of yield differential over lookback
      - Long  (+1) when rank > 51% AND yield > 0  (carry is high vs history AND positive)
      - Short (-1) when rank < 49% AND yield < 0  (carry is low vs history AND negative)
      - Flat  ( 0) otherwise (in the middle of range)

    Useful for EM carry: enter when yield differential is at a relative extreme.

    Parameters
    ----------
    impl_yield : wide DataFrame of annualised yield differentials
    lookback   : rolling window for rank calculation
    """
    def _rolling_rank(df, window):
        """Percentile rank within the rolling window (0 to 1)."""
        roll_min = df.rolling(window).min()
        roll_max = df.rolling(window).max()
        rng = roll_max - roll_min
        return (df - roll_min) / rng.replace(0, np.nan)

    rank = _rolling_rank(impl_yield, lookback)

    sig = pd.DataFrame(0.0, index=impl_yield.index, columns=impl_yield.columns)
    sig[(rank >= 0.51) & (impl_yield > 0.00001)] =  1.0
    sig[(rank <= 0.49) & (impl_yield < -0.00001)] = -1.0

    return sig.shift(1)


def yield_trend(
    impl_yield: pd.DataFrame,
    lookback: int = 252,
    ema_span: int = 5,
) -> pd.DataFrame:
    """
    Yield trend signal. Ported from rcq daily_nom_yield_trend.

    Logic:
      - Smooth yield differential with short EMA (captures recent trend direction)
      - Compute z-score of smoothed yield vs rolling history
      - Interpolate z-score to [-1, +1] (continuous sizing, not just sign)

    Useful for: trading rate-differential trends — e.g. one central bank hiking
    while another is cutting → persistent yield trend to ride.

    Parameters
    ----------
    impl_yield : wide DataFrame of annualised yield differentials
    lookback   : rolling window for z-score
    ema_span   : EMA span for smoothing yield before z-scoring
    """
    smoothed = impl_yield.ewm(span=ema_span, adjust=False).mean()
    roll_mean = smoothed.rolling(lookback).mean()
    roll_std  = smoothed.rolling(lookback).std()
    z = (smoothed - roll_mean) / roll_std

    # Interpolate: z clamped to [-1, 1], giving continuous position size
    sig = z.clip(-1, 1)

    return sig.shift(1)
