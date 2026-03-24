from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from .base import DataLoader


class CSVLoader(DataLoader):
    """
    Load FX data from CSV files.

    Supports two layouts:

    1. One file per pair  (default)
       Directory:  <data_dir>/EURUSD.csv, USDJPY.csv, ...
       Columns:    Date, Close [, Open, High, Low, Bid, Ask, Volume,
                               fwd_points, fwd_outright, impl_vol, impl_yield]

    2. Single combined file
       Columns: Date, pair, Close [, ...]
       Pass combined=True.

    Bloomberg terminal CSV exports work out of the box.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        timestamp_col: str = "Date",
        combined: bool = False,
        pair_col: str = "pair",
    ):
        self.data_dir = Path(data_dir)
        self.timestamp_col = timestamp_col
        self.combined = combined
        self.pair_col = pair_col

    def load(self, pairs: list[str], start: str, end: str, freq: str = "1D") -> pd.DataFrame:
        if self.combined:
            df = self._load_combined(pairs, start, end, freq)
        else:
            df = self._load_per_pair(pairs, start, end, freq)
        return self.validate(df)

    def _load_per_pair(self, pairs, start, end, freq):
        frames = []
        for pair in pairs:
            path = self.data_dir / f"{pair.upper()}.csv"
            if not path.exists():
                candidates = (list(self.data_dir.glob(f"{pair.upper()}.csv")) +
                              list(self.data_dir.glob(f"{pair.lower()}.csv")))
                if not candidates:
                    raise FileNotFoundError(f"No CSV for '{pair}' in {self.data_dir}")
                path = candidates[0]
            raw = pd.read_csv(path)
            raw = self._parse_timestamps(raw)
            raw["pair"] = pair.upper()
            frames.append(self._filter_resample(raw, start, end, freq))
        return pd.concat(frames).sort_index()

    def _load_combined(self, pairs, start, end, freq):
        raw = pd.read_csv(self.data_dir)
        raw = self._parse_timestamps(raw)
        if self.pair_col not in raw.columns:
            raise ValueError(f"Combined CSV needs a '{self.pair_col}' column.")
        raw = raw.rename(columns={self.pair_col: "pair"})
        raw["pair"] = raw["pair"].str.upper()
        if pairs:
            raw = raw[raw["pair"].isin([p.upper() for p in pairs])]
        frames = [self._filter_resample(g, start, end, freq)
                  for _, g in raw.groupby("pair")]
        return pd.concat(frames).sort_index()

    def _parse_timestamps(self, df):
        col = self.timestamp_col
        if col not in df.columns:
            for c in ["Date", "date", "Datetime", "datetime", "timestamp", "Time"]:
                if c in df.columns:
                    col = c
                    break
            else:
                raise ValueError(f"Cannot find timestamp column. Columns: {df.columns.tolist()}")
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True, utc=True)
        df = df.set_index(col)
        df.index.name = "timestamp"
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    def _filter_resample(self, df, start, end, freq):
        df = df.sort_index().loc[start:end]
        if freq == "raw":
            return df
        agg = {}
        for col in df.columns:
            if col == "pair":
                continue
            if col == "open":       agg[col] = "first"
            elif col == "high":     agg[col] = "max"
            elif col == "low":      agg[col] = "min"
            elif col == "volume":   agg[col] = "sum"
            else:                   agg[col] = "last"
        pair_val = df["pair"].iloc[0] if "pair" in df.columns else None
        df = df.drop(columns=["pair"], errors="ignore").resample(freq).agg(agg).dropna(how="all")
        if pair_val:
            df["pair"] = pair_val
        return df
