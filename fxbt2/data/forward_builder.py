from __future__ import annotations

"""
ForwardBuilder — construct FX forward outrights from spot + forward points.

Handles the three quoting conventions present across the currency universe:
  - Standard pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, etc.)
      outright = spot + fwd_pts / 10000
  - JPY pairs (USDJPY, EURJPY, GBPJPY, etc.)
      outright = spot + fwd_pts / 100
  - CLP
      outright = spot + fwd_pts  (fwd pts already in price units)
  - NDF pairs (USDBRL, USDKRW, USDINR, USDIDR, USDTWD, USDPHP, USDCNH, etc.)
      outright = NDF outright directly (no construction needed)

Also computes implied yield carry:
  carry = (outright / spot - 1) * annualisation_factor

with a holiday filter: if spot didn't move (public holiday), use a rolling
average of recent carry instead of the stale fixing — ported from rcq_trading_library.
"""

import numpy as np
import pandas as pd

from .ticker_dict import NDF_CCYS, FWD_PIP_SCALE


class ForwardBuilder:
    """
    Build forward outrights and implied yield differentials from raw data.

    Parameters
    ----------
    ann_factor : int
        Annualisation factor for implied yield. Default 12 (monthly tenors × 12).
    holiday_window : int
        Rolling window to average carry over on fixing holidays. Default 5.
    """

    def __init__(self, ann_factor: int = 12, holiday_window: int = 5):
        self.ann_factor = ann_factor
        self.holiday_window = holiday_window

    def build_outright(
        self,
        spot: pd.DataFrame,
        fwd_pts: pd.DataFrame,
        ccy_codes: list[str],
    ) -> pd.DataFrame:
        """
        Construct 1M forward outrights per currency (vs USD).

        Parameters
        ----------
        spot      : pd.DataFrame  wide, columns = ccy codes (e.g. 'EUR', 'JPY')
                                  values = USD per 1 unit of ccy (EURUSD convention)
        fwd_pts   : pd.DataFrame  wide, raw 1M forward points per ccy
        ccy_codes : list[str]     currencies to build (subset of spot.columns)

        Returns
        -------
        pd.DataFrame  wide, same shape as spot, values = 1M outright prices
        """
        outright = pd.DataFrame(index=spot.index)

        for ccy in ccy_codes:
            if ccy == "USD":
                outright["USD"] = 1.0
                continue

            if ccy in NDF_CCYS:
                # NDF: outright is pulled directly — just pass through fwd_pts
                outright[ccy] = fwd_pts[ccy]
                continue

            scale = FWD_PIP_SCALE.get(ccy, 10000)

            if ccy in {"EUR", "GBP", "AUD", "NZD"}:
                # These are quoted as USD per 1 CCY → fwd pts inverted
                outright[ccy] = spot[ccy] + (-fwd_pts[ccy] / scale)
            else:
                outright[ccy] = spot[ccy] + (fwd_pts[ccy] / scale)

        return outright

    def build_pair_outright(
        self,
        spot: pd.DataFrame,
        fwd_pts: pd.DataFrame,
        ccy_codes: list[str],
        pairs: list[str],
    ) -> pd.DataFrame:
        """
        Build outrights for specific FX pairs (e.g. EURUSD, USDJPY).

        Parameters
        ----------
        spot, fwd_pts : wide DataFrames with ccy codes as columns
        ccy_codes     : list of all ccy codes needed
        pairs         : list of FX pairs, e.g. ['EURUSD', 'USDJPY']

        Returns
        -------
        pd.DataFrame  wide, columns = pair names
        """
        ccy_outright = self.build_outright(spot, fwd_pts, ccy_codes)
        pair_df = pd.DataFrame(index=spot.index)

        for pair in pairs:
            base = pair[:3]
            quote = pair[3:]
            # pair outright = quote_outright / base_outright
            pair_df[pair] = ccy_outright[quote] / ccy_outright[base]

        return pair_df

    def build_implied_yield(
        self,
        spot: pd.DataFrame,
        fwd_pts: pd.DataFrame,
        ccy_codes: list[str],
        pairs: list[str],
    ) -> pd.DataFrame:
        """
        Compute implied yield differential per pair.

        carry_pair = carry_base - carry_quote
        carry_ccy  = (outright / spot - 1) * ann_factor

        Holiday filter: if spot[ccy] == spot[ccy].shift(1) (no move = holiday),
        substitute with rolling mean of recent carry to avoid stale fixing.

        Parameters
        ----------
        spot, fwd_pts : wide DataFrames with ccy codes as columns
        ccy_codes     : list of ccy codes
        pairs         : list of FX pairs

        Returns
        -------
        pd.DataFrame  wide, columns = pair names, values = annualised yield diff
        """
        ccy_outright = self.build_outright(spot, fwd_pts, ccy_codes)

        # Per-ccy implied yield
        ccy_yield = pd.DataFrame(index=spot.index)
        for ccy in ccy_codes:
            if ccy == "USD":
                ccy_yield["USD"] = 0.0
                continue
            if ccy in NDF_CCYS:
                raw = (ccy_outright[ccy] / spot[ccy] - 1) * self.ann_factor
            else:
                raw = (ccy_outright[ccy] / spot[ccy] - 1) * self.ann_factor

            # Holiday filter: where spot didn't move, use rolling mean
            holiday_mask = spot[ccy] == spot[ccy].shift(1)
            roll_avg = raw.rolling(window=self.holiday_window).mean()
            ccy_yield[ccy] = np.where(holiday_mask, roll_avg.shift(1), raw)

        # Pair yield = base yield - quote yield
        pair_yield = pd.DataFrame(index=spot.index)
        for pair in pairs:
            base = pair[:3]
            quote = pair[3:]
            pair_yield[pair] = ccy_yield[base] - ccy_yield[quote]

        return pair_yield
