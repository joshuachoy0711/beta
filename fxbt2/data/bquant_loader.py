from __future__ import annotations

"""
BQuantLoader — load FX data via Bloomberg BQL (BQuant JupyterLab environment).

Uses the ticker_dict for all ticker resolution so adding a new currency
only requires updating ticker_dict.py, not this file.
"""

import pandas as pd

from .base import DataLoader
from .ticker_dict import get_ticker, NDF_CCYS

_BQL_FREQ_MAP = {"1D": "D", "1W": "W", "1M": "M"}


class BQuantLoader(DataLoader):
    """
    Load FX data via Bloomberg BQL (BQuant environment only).

    Parameters
    ----------
    fixing        : str   Bloomberg fixing, e.g. 'CMPT', 'BGN', 'CMPN'
    load_forwards : bool  Fetch 1M forward outrights. Default True.
    load_vol      : bool  Fetch 1M ATM implied vol. Default True.
    load_yield    : bool  Fetch 1M implied yield. Default True.
    """

    def __init__(
        self,
        fixing: str = "CMPT",
        load_forwards: bool = True,
        load_vol: bool = True,
        load_yield: bool = True,
    ):
        self.fixing = fixing
        self.load_forwards = load_forwards
        self.load_vol = load_vol
        self.load_yield = load_yield
        self._bql = None

    def load(self, pairs: list[str], start: str, end: str, freq: str = "1D") -> pd.DataFrame:
        bql = self._get_bql()
        bbg_freq = _BQL_FREQ_MAP.get(freq)
        if not bbg_freq:
            raise ValueError(
                f"freq='{freq}' not supported by BDH. "
                "For intraday data export to CSV and use CSVLoader."
            )
        frames = [self._fetch_pair(bql, p.upper(), start, end, bbg_freq) for p in pairs]
        return self.validate(pd.concat(frames).sort_index())

    def _fetch_pair(self, bql, pair, start, end, freq):
        spot_ticker = get_ticker(pair[3:], "spot", self.fixing) if pair[:3] == "USD" \
                      else f"{pair} {self.fixing} Curncy"

        df = self._bdh(bql, f"{pair} {self.fixing} Curncy",
                       ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"],
                       start, end, freq)
        df = df.rename(columns={"PX_OPEN": "open", "PX_HIGH": "high",
                                 "PX_LOW": "low", "PX_LAST": "close"})

        base_ccy = pair[:3] if pair[:3] != "USD" else pair[3:]

        if self.load_forwards:
            fwd_ticker = get_ticker(base_ccy,
                                    "1m_ndf_outright" if base_ccy in NDF_CCYS else "1m_fwd_pts",
                                    self.fixing)
            if fwd_ticker:
                try:
                    fwd = self._bdh(bql, fwd_ticker, ["PX_LAST"], start, end, freq)
                    df["fwd_points"] = fwd["PX_LAST"]
                except Exception:
                    pass

        if self.load_vol:
            vol_ticker = get_ticker(base_ccy, "1m_impl_vol", self.fixing)
            if vol_ticker:
                try:
                    vol = self._bdh(bql, vol_ticker, ["PX_LAST"], start, end, freq)
                    df["impl_vol"] = vol["PX_LAST"]
                except Exception:
                    pass

        if self.load_yield:
            yld_ticker = get_ticker(base_ccy, "1m_impl_yield", self.fixing)
            if yld_ticker:
                try:
                    yld = self._bdh(bql, yld_ticker, ["PX_LAST"], start, end, freq)
                    df["impl_yield"] = yld["PX_LAST"]
                except Exception:
                    pass

        df["pair"] = pair
        return df

    def _bdh(self, bql, ticker, fields, start, end, freq):
        date_range = bql.func.range(start, end)
        # BQL data items are lowercase snake_case (e.g. px_last, px_open)
        items = {f: getattr(bql.data, f.lower())(dates=date_range, per=freq) for f in fields}
        response = bql.execute(ticker, items).get()
        frames = {}
        for field in fields:
            df_r = response[field].df()
            s = df_r["VALUE"] if "VALUE" in df_r.columns else df_r.select_dtypes("number").iloc[:, 0]
            # BQL often returns a MultiIndex of (ID, DATE) — extract the date level
            if isinstance(s.index, pd.MultiIndex):
                date_level = next(
                    (lvl for lvl in s.index.names if "DATE" in str(lvl).upper()), -1
                )
                s = s.copy()
                s.index = pd.to_datetime(s.index.get_level_values(date_level), utc=True)
            else:
                s.index = pd.to_datetime(s.index, utc=True)
            frames[field] = s
        return pd.DataFrame(frames)

    def _get_bql(self):
        if self._bql:
            return self._bql
        try:
            import bql
            self._bql = bql.Service()
            return self._bql
        except ImportError:
            raise ImportError(
                "bql not available. BQuantLoader only works inside Bloomberg BQuant. "
                "Use CSVLoader or PdblpLoader for local development."
            )
