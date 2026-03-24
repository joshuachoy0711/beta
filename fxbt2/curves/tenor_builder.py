from __future__ import annotations

"""
CurveBuilder — fetch and construct NDF/forward curves across multiple tenors.

Given a Bloomberg connection (pdblp.BCon) or a pre-loaded DataFrame, builds
a term structure of outright rates and implied yields for any FX pair.

Supported tenors (Bloomberg standard):
    ON, 1W, 2W, 1M, 2M, 3M, 6M, 9M, 12M, 2Y

Ticker convention (Bloomberg):
    - Deliverable fwd pts  : {CCY}{TENOR}  e.g. EUR1M, JPY3M, EUR12M
    - NDF outright         : {CCY}{TENOR}  e.g. KWN3M, IRN1M, KWN12M  (no + prefix)
    - Implied yield (1M)   : {CCY}I{TENOR} e.g. EURI1M, KWNI1M

NDF pairs use outright tickers directly.
Deliverable pairs construct outright = spot + fwd_pts / pip_scale.
"""

import datetime as dt
import pandas as pd

from ..data.ticker_dict import NDF_CCYS, FWD_PIP_SCALE

# ---------------------------------------------------------------------------
# Tenor metadata
# ---------------------------------------------------------------------------

#: Ordered list of standard FX tenors
TENORS: list[str] = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "12M", "2Y"]

#: Approximate business-day length of each tenor (used for annualisation)
TENOR_DAYS: dict[str, float] = {
    "ON": 1, "1W": 5, "2W": 10, "1M": 21, "2M": 42, "3M": 63,
    "6M": 126, "9M": 189, "12M": 252, "2Y": 504,
}

#: Bloomberg ticker suffix for each tenor
_TENOR_BBG: dict[str, str] = {
    "ON": "ON", "1W": "1W", "2W": "2W", "1M": "1M",
    "2M": "2M", "3M": "3M", "6M": "6M", "9M": "9M",
    "12M": "12M", "2Y": "2Y",
}

#: NDF currency root codes (Bloomberg prefix before the tenor suffix)
_NDF_ROOT: dict[str, str] = {
    "BRL": "BCN", "CLP": "CHN", "COP": "CON", "KRW": "KWN",
    "INR": "IRN", "IDR": "IHN", "TWD": "NTN", "PHP": "PPN",
    "CNH": "CNH", "SGD": "SGD", "THB": "THB",
    "MXN": "MXN", "ZAR": "ZAR",
}

#: Deliverable fwd pts root codes
_FWD_ROOT: dict[str, str] = {
    "EUR": "EUR", "GBP": "GBP", "JPY": "JPY", "AUD": "AUD",
    "NZD": "NZD", "CAD": "CAD", "CHF": "CHF", "NOK": "NOK",
    "SEK": "SEK", "HUF": "HUF", "CZK": "CZK", "PLN": "PLN",
    "ILS": "ILS",
}


def _ccy_from_pair(pair: str) -> str:
    """Return the non-USD currency code from a pair string."""
    pair = pair.upper()
    return pair[3:] if pair[:3] == "USD" else pair[:3]


def build_tenor_tickers(pair: str, fixing: str = "CMPT") -> dict[str, str]:
    """
    Return a dict of {tenor: bloomberg_ticker} for all tenors for a given pair.

    For NDF pairs  → uses outright tickers: ``{ROOT}{TENOR} {FIX} Curncy``  e.g. ``KWN3M BGN Curncy``
    For deliverable → uses fwd pts tickers: ``{ROOT}{TENOR} {FIX} Curncy``  e.g. ``EUR3M BGN Curncy``

    Parameters
    ----------
    pair   : str  FX pair e.g. 'USDKRW', 'EURUSD', 'USDBRL'
    fixing : str  Bloomberg fixing code. Default 'CMPT'.

    Returns
    -------
    dict[tenor, ticker_string]
    """
    ccy = _ccy_from_pair(pair)
    tickers = {}

    if ccy in NDF_CCYS or ccy in _NDF_ROOT:
        root = _NDF_ROOT.get(ccy, ccy)
        for tenor in TENORS:
            sfx = _TENOR_BBG[tenor]
            tickers[tenor] = f"{root}{sfx} {fixing} Curncy"
    else:
        root = _FWD_ROOT.get(ccy, ccy)
        for tenor in TENORS:
            sfx = _TENOR_BBG[tenor]
            tickers[tenor] = f"{root}{sfx} {fixing} Curncy"

    return tickers


# ---------------------------------------------------------------------------
# CurveBuilder
# ---------------------------------------------------------------------------

class CurveBuilder:
    """
    Build FX forward / NDF outright curves across multiple tenors.

    Can operate in two modes:
    - **Live** (Bloomberg): pass a ``pdblp.BCon`` connection
    - **Offline** (pre-loaded data): pass DataFrames directly to ``from_dataframes()``

    Parameters
    ----------
    fixing : str
        Bloomberg fixing code. Default ``'CMPT'``.

    Examples
    --------
    Live mode::

        import pdblp
        con = pdblp.BCon(debug=False, port=8194, timeout=50000)
        con.start()

        cb = CurveBuilder(fixing='CMPT')
        curve_today = cb.fetch_curve('USDKRW', con)
        history     = cb.fetch_history('USDKRW', con, start='20230101', end='20240101')

    Offline mode::

        cb = CurveBuilder()
        history = cb.from_dataframes(outright_wide, spot_series)
    """

    def __init__(self, fixing: str = "CMPT"):
        self.fixing = fixing

    # ------------------------------------------------------------------
    # Live Bloomberg fetch
    # ------------------------------------------------------------------

    def fetch_curve(
        self,
        pair: str,
        con,
        date: str | None = None,
        tenors: list[str] | None = None,
    ) -> pd.Series:
        """
        Fetch today's (or a specific date's) forward curve for one pair.

        Parameters
        ----------
        pair   : str         FX pair e.g. 'USDKRW'
        con    : pdblp.BCon  Live Bloomberg connection
        date   : str         'YYYYMMDD'. Defaults to today.
        tenors : list[str]   Subset of TENORS to fetch. Default = all.

        Returns
        -------
        pd.Series  index = tenor labels, values = outright rates
        """
        date = date or dt.date.today().strftime("%Y%m%d")
        tenors = tenors or TENORS
        tickers = build_tenor_tickers(pair, self.fixing)

        spot_ticker = self._spot_ticker(pair)
        try:
            spot_raw = con.bdh([spot_ticker], ["PX_LAST"], date, date)
            spot = float(spot_raw.iloc[-1, -1])
        except Exception:
            spot = float("nan")

        outrights = {}
        ccy = _ccy_from_pair(pair)
        is_ndf = ccy in NDF_CCYS or ccy in _NDF_ROOT

        for tenor in tenors:
            ticker = tickers[tenor]
            try:
                raw = con.bdh([ticker], ["PX_LAST"], date, date)
                val = float(raw.iloc[-1, -1])
            except Exception:
                outrights[tenor] = float("nan")
                continue

            if is_ndf:
                outrights[tenor] = val  # already an outright
            else:
                scale = FWD_PIP_SCALE.get(ccy, 10000)
                if ccy in {"EUR", "GBP", "AUD", "NZD"}:
                    outrights[tenor] = spot + (-val / scale)
                else:
                    outrights[tenor] = spot + (val / scale)

        curve = pd.Series(outrights, name=pair)
        curve.index.name = "tenor"
        return curve

    def fetch_history(
        self,
        pair: str,
        con,
        start: str,
        end: str,
        tenors: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical forward curve data for one pair.

        Parameters
        ----------
        pair        : str         FX pair e.g. 'USDKRW'
        con         : pdblp.BCon
        start / end : str         'YYYYMMDD' or 'YYYY-MM-DD'
        tenors      : list[str]   Subset of TENORS. Default = all.

        Returns
        -------
        pd.DataFrame
            MultiIndex columns = (tenor, 'outright')
            DatetimeIndex rows
            Plus a 'spot' column.
        """
        start = start.replace("-", "")
        end   = end.replace("-", "")
        tenors = tenors or TENORS
        tickers = build_tenor_tickers(pair, self.fixing)
        ccy = _ccy_from_pair(pair)
        is_ndf = ccy in NDF_CCYS or ccy in _NDF_ROOT
        scale = FWD_PIP_SCALE.get(ccy, 10000)
        elms = [("calendarCodeOverride", "5D")]

        # Spot
        spot_ticker = self._spot_ticker(pair)
        try:
            spot_raw = con.bdh([spot_ticker], ["PX_LAST"], start, end, elms=elms)
            spot = spot_raw.iloc[:, -1].rename("spot")
            spot.index = pd.to_datetime(spot.index)
        except Exception:
            spot = pd.Series(dtype=float, name="spot")

        frames = {"spot": spot}

        for tenor in tenors:
            ticker = tickers[tenor]
            try:
                raw = con.bdh([ticker], ["PX_LAST"], start, end, elms=elms)
                raw.index = pd.to_datetime(raw.index)
                vals = raw.iloc[:, -1]
            except Exception:
                continue

            if is_ndf:
                frames[tenor] = vals
            else:
                if ccy in {"EUR", "GBP", "AUD", "NZD"}:
                    frames[tenor] = spot.reindex(vals.index).ffill() + (-vals / scale)
                else:
                    frames[tenor] = spot.reindex(vals.index).ffill() + (vals / scale)

        df = pd.DataFrame(frames)
        df.index.name = "date"
        return df

    # ------------------------------------------------------------------
    # Offline: build from pre-loaded DataFrames
    # ------------------------------------------------------------------

    def from_dataframes(
        self,
        fwd_pts_or_outrights: pd.DataFrame,
        spot: pd.Series | pd.DataFrame | None = None,
        is_ndf: bool = False,
        pair: str = "",
    ) -> pd.DataFrame:
        """
        Build a history DataFrame from pre-loaded fwd pts / outright data.

        Parameters
        ----------
        fwd_pts_or_outrights : pd.DataFrame
            Columns = tenors (e.g. '1M', '3M', '1Y'), rows = dates.
            For deliverable pairs: forward points.
            For NDF pairs: outright rates directly (set ``is_ndf=True``).
        spot : pd.Series or DataFrame, optional
            Spot rates. Required for deliverable pairs.
        is_ndf : bool
            If True, treats input as outright rates (no construction needed).
        pair : str
            Pair name, used for column labels only.

        Returns
        -------
        pd.DataFrame  columns = tenor names (+ 'spot' if provided)
        """
        df = fwd_pts_or_outrights.copy()

        if is_ndf or spot is None:
            return df

        if isinstance(spot, pd.DataFrame):
            spot = spot.iloc[:, 0]

        spot_al = spot.reindex(df.index).ffill()
        ccy = _ccy_from_pair(pair) if pair else ""
        scale = FWD_PIP_SCALE.get(ccy, 10000)

        for col in df.columns:
            if ccy in {"EUR", "GBP", "AUD", "NZD"}:
                df[col] = spot_al + (-df[col] / scale)
            else:
                df[col] = spot_al + (df[col] / scale)

        df.insert(0, "spot", spot_al)
        return df

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def implied_yield_curve(
        self,
        history: pd.DataFrame,
        tenors: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute annualised implied yield at each tenor from outright rates.

        Formula::

            implied_yield(tenor) = (outright / spot - 1) * (252 / tenor_days)

        Parameters
        ----------
        history : pd.DataFrame
            Output of ``fetch_history()`` or ``from_dataframes()``.
            Must contain a 'spot' column and one column per tenor.
        tenors  : list[str]  Tenors to include. Default = all present.

        Returns
        -------
        pd.DataFrame  same shape as history (excluding spot), values = ann. yield
        """
        if "spot" not in history.columns:
            raise ValueError("history must contain a 'spot' column")

        spot = history["spot"]
        cols = tenors or [c for c in history.columns if c != "spot"]
        result = pd.DataFrame(index=history.index)

        for tenor in cols:
            if tenor not in history.columns:
                continue
            days = TENOR_DAYS.get(tenor, 21)
            ann_factor = 252 / days
            result[tenor] = (history[tenor] / spot - 1) * ann_factor

        result.index.name = "date"
        return result

    def percentile_bands(
        self,
        history: pd.DataFrame,
        tenors: list[str] | None = None,
        pcts: tuple[float, ...] = (10, 25, 50, 75, 90),
    ) -> pd.DataFrame:
        """
        Compute historical percentile levels for each tenor.

        Parameters
        ----------
        history : pd.DataFrame  Historical outright or yield DataFrame.
        tenors  : list[str]     Tenors to include.
        pcts    : tuple[float]  Percentiles to compute (0–100).

        Returns
        -------
        pd.DataFrame  index = percentile labels, columns = tenors
        """
        cols = tenors or [c for c in history.columns if c != "spot"]
        data = history[[c for c in cols if c in history.columns]].dropna(how="all")
        rows = {}
        for p in pcts:
            rows[f"p{int(p)}"] = data.quantile(p / 100)
        return pd.DataFrame(rows).T

    # ------------------------------------------------------------------
    # BQL (BQuant) fetch methods
    # ------------------------------------------------------------------

    def fetch_curve_bql(
        self,
        pair: str,
        bql,
        date: str | None = None,
        tenors: list[str] | None = None,
    ) -> pd.Series:
        """
        Fetch today's (or a specific date's) forward curve using BQL (BQuant).

        Parameters
        ----------
        pair   : str          FX pair e.g. 'USDKRW'
        bql    : bql.Service  BQL service instance (``bql.Service()``)
        date   : str          'YYYY-MM-DD'. Defaults to today.
        tenors : list[str]    Subset of TENORS to fetch. Default = all.

        Returns
        -------
        pd.Series  index = tenor labels, values = outright rates
        """
        date = date or dt.date.today().strftime("%Y-%m-%d")
        tenors = tenors or TENORS
        tickers = build_tenor_tickers(pair, self.fixing)
        ccy = _ccy_from_pair(pair)
        is_ndf = ccy in NDF_CCYS or ccy in _NDF_ROOT
        scale = FWD_PIP_SCALE.get(ccy, 10000)

        # Spot
        spot_ticker = self._spot_ticker(pair)
        try:
            spot = self._bql_ref(bql, spot_ticker, date)
        except Exception as e:
            print(f"[CurveBuilder] spot fetch failed ({spot_ticker}): {e}")
            spot = float("nan")

        outrights = {}
        for tenor in tenors:
            ticker = tickers[tenor]
            try:
                val = self._bql_ref(bql, ticker, date)
            except Exception as e:
                print(f"[CurveBuilder] {tenor} fetch failed ({ticker}): {e}")
                outrights[tenor] = float("nan")
                continue

            if is_ndf:
                outrights[tenor] = val
            else:
                if ccy in {"EUR", "GBP", "AUD", "NZD"}:
                    outrights[tenor] = spot + (-val / scale)
                else:
                    outrights[tenor] = spot + (val / scale)

        curve = pd.Series(outrights, name=pair)
        curve.index.name = "tenor"
        return curve

    def fetch_history_bql(
        self,
        pair: str,
        bql,
        start: str,
        end: str,
        tenors: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical forward curve data using BQL (BQuant).

        Parameters
        ----------
        pair        : str          FX pair e.g. 'USDKRW'
        bql         : bql.Service  BQL service instance
        start / end : str          'YYYY-MM-DD'
        tenors      : list[str]    Subset of TENORS. Default = all.

        Returns
        -------
        pd.DataFrame
            Columns = 'spot' + tenor names, DatetimeIndex rows.
        """
        tenors = tenors or TENORS
        tickers = build_tenor_tickers(pair, self.fixing)
        ccy = _ccy_from_pair(pair)
        is_ndf = ccy in NDF_CCYS or ccy in _NDF_ROOT
        scale = FWD_PIP_SCALE.get(ccy, 10000)

        # Spot
        spot_ticker = self._spot_ticker(pair)
        try:
            spot = self._bql_bdh(bql, spot_ticker, start, end)
            spot.name = "spot"
        except Exception as e:
            print(f"[CurveBuilder] spot history fetch failed ({spot_ticker}): {e}")
            spot = pd.Series(dtype=float, name="spot")

        frames = {"spot": spot}

        for tenor in tenors:
            ticker = tickers[tenor]
            try:
                vals = self._bql_bdh(bql, ticker, start, end)
            except Exception as e:
                print(f"[CurveBuilder] {tenor} history fetch failed ({ticker}): {e}")
                continue

            if is_ndf:
                frames[tenor] = vals
            else:
                sp = spot.reindex(vals.index).ffill()
                if ccy in {"EUR", "GBP", "AUD", "NZD"}:
                    frames[tenor] = sp + (-vals / scale)
                else:
                    frames[tenor] = sp + (vals / scale)

        df = pd.DataFrame(frames)
        df.index.name = "date"
        return df

    # ------------------------------------------------------------------
    # BQL internal helpers
    # ------------------------------------------------------------------

    def _bql_ref(self, bql_svc, ticker: str, date: str) -> float:
        """Fetch a single reference value via BQL for a given date."""
        date_range = bql_svc.func.range(date, date)
        item = {"PX_LAST": bql_svc.data.px_last(dates=date_range)}
        res = bql_svc.execute(ticker, item).get()
        return self._parse_bql_scalar(res["PX_LAST"].df())

    def _bql_bdh(self, bql_svc, ticker: str, start: str, end: str) -> pd.Series:
        """Fetch a daily price history via BQL and return as a Series."""
        date_range = bql_svc.func.range(start, end)
        item = {"PX_LAST": bql_svc.data.px_last(dates=date_range, per="D")}
        res = bql_svc.execute(ticker, item).get()
        return self._parse_bql_series(res["PX_LAST"].df(), ticker)

    @staticmethod
    def _parse_bql_scalar(df: pd.DataFrame) -> float:
        """Extract a single float value from a BQL response DataFrame."""
        if "VALUE" in df.columns:
            return float(df["VALUE"].dropna().iloc[-1])
        numeric = df.select_dtypes("number")
        return float(numeric.dropna().iloc[-1, 0])

    @staticmethod
    def _parse_bql_series(df: pd.DataFrame, name: str = "") -> pd.Series:
        """Extract a DatetimeIndex Series from a BQL response DataFrame."""
        if "VALUE" in df.columns:
            s = df["VALUE"].dropna()
        else:
            s = df.select_dtypes("number").iloc[:, 0].dropna()
        # BQL often returns a MultiIndex of (ID, DATE) — extract the date level
        if isinstance(s.index, pd.MultiIndex):
            date_level = next(
                (lvl for lvl in s.index.names if "DATE" in str(lvl).upper()), -1
            )
            s.index = pd.to_datetime(s.index.get_level_values(date_level))
        else:
            s.index = pd.to_datetime(s.index)
        s.name = name
        return s.sort_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spot_ticker(self, pair: str) -> str:
        pair = pair.upper()
        return f"{pair} {self.fixing} Curncy"
