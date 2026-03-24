from __future__ import annotations

"""
PdblpLoader — load FX data via pdblp (Bloomberg Terminal connection).

Requires a running Bloomberg Terminal session on the same machine.
Uses the shared ticker_dict and ForwardBuilder for all data construction
so it is fully consistent with BQuantLoader output.

Dependencies: pdblp  (pip install pdblp)
"""

import pandas as pd

from .base import DataLoader
from .ticker_dict import get_ticker, get_ccy_list, BBG_TICKER_DICT, NDF_CCYS
from .forward_builder import ForwardBuilder


class PdblpLoader(DataLoader):
    """
    Load FX data via Bloomberg Terminal using pdblp.

    Parameters
    ----------
    fixing        : str   Bloomberg fixing timestamp, e.g. 'CMPT', 'BGN', 'CMPN'
    port          : int   Bloomberg API port. Default 8194.
    timeout       : int   Connection timeout ms. Default 50000.
    load_forwards : bool  Build 1M forward outrights. Default True.
    load_vol      : bool  Fetch 1M ATM implied vol. Default True.
    load_yield    : bool  Compute implied yield differential. Default True.

    Usage
    -----
    loader = PdblpLoader(fixing='CMPT')
    df = loader.load(['EURUSD', 'USDJPY', 'USDMXN'],
                     start='20200101', end='20240101')
    """

    def __init__(
        self,
        fixing: str = "CMPT",
        port: int = 8194,
        timeout: int = 50000,
        load_forwards: bool = True,
        load_vol: bool = True,
        load_yield: bool = True,
    ):
        self.fixing = fixing
        self.port = port
        self.timeout = timeout
        self.load_forwards = load_forwards
        self.load_vol = load_vol
        self.load_yield = load_yield
        self._con = None
        self._fwd_builder = ForwardBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, pairs: list[str], start: str, end: str, freq: str = "1D") -> pd.DataFrame:
        """
        Load price data for one or more FX pairs via Bloomberg Terminal.

        Parameters
        ----------
        start / end : str  Date in YYYYMMDD or YYYY-MM-DD format.
        freq        : str  '1D' (daily) only for BDH. For intraday use load_intraday().
        """
        con = self._get_connection()
        start_fmt = start.replace("-", "")
        end_fmt = end.replace("-", "") if end else ""

        frames = []
        for pair in pairs:
            df = self._fetch_pair(con, pair.upper(), start_fmt, end_fmt)
            frames.append(df)

        return self.validate(pd.concat(frames).sort_index())

    def load_intraday(
        self,
        pairs: list[str],
        start_datetime: str,
        end_datetime: str,
        interval_min: int = 60,
    ) -> dict[str, pd.DataFrame]:
        """
        Load intraday bar data via Bloomberg BDIB.

        Parameters
        ----------
        start_datetime : str   ISO format '2024-01-01T00:00:00'
        end_datetime   : str   ISO format '2024-12-31T23:59:59'
        interval_min   : int   Bar interval in minutes. Default 60.

        Returns
        -------
        dict[pair_name, pd.DataFrame]  raw OHLCV DataFrames per pair
        """
        con = self._get_connection()
        result = {}
        for pair in pairs:
            ticker = f"{pair.upper()} {self.fixing} Curncy"
            try:
                df = con.bdib(ticker, start_datetime, end_datetime, "TRADE", interval_min)
                df.index = pd.to_datetime(df.index, utc=True)
                df.columns = [c.lower() for c in df.columns]
                result[pair.upper()] = df
            except Exception as e:
                print(f"Warning: could not load intraday data for {pair}: {e}")
        return result

    # ------------------------------------------------------------------
    # Internal: fetch a single pair
    # ------------------------------------------------------------------

    def _fetch_pair(self, con, pair: str, start: str, end: str) -> pd.DataFrame:
        spot_ticker = f"{pair} {self.fixing} Curncy"
        elms = [("calendarCodeOverride", "5D")]

        # Spot OHLCV
        raw = con.bdh([spot_ticker],
                      ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "BID", "ASK"],
                      start, end, elms=elms)

        # Flatten multi-level columns from pdblp
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[1].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        raw = raw.rename(columns={
            "px_open": "open", "px_high": "high",
            "px_low": "low", "px_last": "close",
        })
        raw.index = pd.to_datetime(raw.index, utc=True)

        base_ccy = pair[:3] if pair[:3] != "USD" else pair[3:]

        # Forward points
        if self.load_forwards:
            asset = "1m_ndf_outright" if base_ccy in NDF_CCYS else "1m_fwd_pts"
            fwd_ticker = get_ticker(base_ccy, asset, self.fixing)
            if fwd_ticker:
                try:
                    fwd = con.bdh([fwd_ticker], ["PX_LAST"], start, end, elms=elms)
                    fwd.index = pd.to_datetime(fwd.index, utc=True)
                    raw["fwd_points"] = fwd.iloc[:, -1].reindex(raw.index).ffill()
                except Exception:
                    pass

        # Implied vol
        if self.load_vol:
            vol_ticker = get_ticker(base_ccy, "1m_impl_vol", self.fixing)
            if vol_ticker:
                try:
                    vol = con.bdh([vol_ticker], ["PX_LAST"], start, end, elms=elms)
                    vol.index = pd.to_datetime(vol.index, utc=True)
                    raw["impl_vol"] = vol.iloc[:, -1].reindex(raw.index).ffill()
                except Exception:
                    pass

        # Implied yield
        if self.load_yield:
            yld_ticker = get_ticker(base_ccy, "1m_impl_yield", self.fixing)
            if yld_ticker:
                try:
                    yld = con.bdh([yld_ticker], ["PX_LAST"], start, end, elms=elms)
                    yld.index = pd.to_datetime(yld.index, utc=True)
                    raw["impl_yield"] = yld.iloc[:, -1].reindex(raw.index).ffill()
                except Exception:
                    pass

        raw["pair"] = pair
        return raw

    # ------------------------------------------------------------------
    # Lazy Bloomberg connection
    # ------------------------------------------------------------------

    def _get_connection(self):
        if self._con is not None:
            return self._con
        try:
            import pdblp
        except ImportError:
            raise ImportError(
                "pdblp is not installed. Run: pip install pdblp\n"
                "Also requires a running Bloomberg Terminal on this machine."
            )
        self._con = pdblp.BCon(debug=False, port=self.port, timeout=self.timeout)
        self._con.start()
        return self._con

    def __del__(self):
        if self._con is not None:
            try:
                self._con.stop()
            except Exception:
                pass
