from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd

REQUIRED_COLS = {"close"}
OPTIONAL_COLS = {"open", "high", "low", "bid", "ask", "volume",
                 "fwd_points", "fwd_outright", "impl_vol", "impl_yield"}
ALL_COLS = REQUIRED_COLS | OPTIONAL_COLS


class DataLoader(ABC):
    """
    Abstract base for all data loaders.

    Every loader returns a long-format DataFrame:
        DatetimeIndex (UTC) | pair | close | [open, high, low, bid, ask,
                                              volume, fwd_points, fwd_outright,
                                              impl_vol, impl_yield]

    Use DataLoader.to_wide(df, field) to pivot to wide format for signal generation.
    """

    @abstractmethod
    def load(
        self,
        pairs: list[str],
        start: str,
        end: str,
        freq: str = "1D",
    ) -> pd.DataFrame:
        """
        Load price data for one or more FX pairs.

        Parameters
        ----------
        pairs  : list[str]   e.g. ['EURUSD', 'USDJPY']
        start  : str         ISO date 'YYYY-MM-DD'
        end    : str         ISO date 'YYYY-MM-DD'
        freq   : str         Pandas offset alias: '1D', '1H', '15min', etc.

        Returns
        -------
        pd.DataFrame  long format, DatetimeIndex UTC
        """
        ...

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        """Enforce schema, coerce types, synthesise bid/ask if absent."""
        if "pair" not in df.columns:
            raise ValueError("DataFrame must have a 'pair' column.")
        if "close" not in df.columns:
            raise ValueError("DataFrame must have a 'close' column.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        numeric = ALL_COLS & set(df.columns)
        df[list(numeric)] = df[list(numeric)].apply(pd.to_numeric, errors="coerce")
        if "bid" not in df.columns and "ask" not in df.columns:
            df["bid"] = df["close"]
            df["ask"] = df["close"]
        return df

    @staticmethod
    def to_wide(df: pd.DataFrame, field: str = "close") -> pd.DataFrame:
        """Pivot long-format df → wide: index=timestamp, columns=pair."""
        if field not in df.columns:
            raise ValueError(f"Field '{field}' not found. Available: {df.columns.tolist()}")
        return df.pivot(columns="pair", values=field)
