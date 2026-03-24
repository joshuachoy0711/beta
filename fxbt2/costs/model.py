from __future__ import annotations

"""
Transaction cost models.

Three components:
  1. Bid/ask spread  — paid on every entry and exit
  2. Slippage        — additional market impact cost
  3. Rollover        — daily overnight financing via fwd points or impl yield
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class CostModel(ABC):
    @abstractmethod
    def entry_cost(self, pair: str, price: float, size: float, **kwargs) -> float: ...
    @abstractmethod
    def exit_cost(self, pair: str, price: float, size: float, **kwargs) -> float: ...

    def rollover_cost(self, pair: str, size: float,
                      fwd_points: float | None = None,
                      price: float | None = None,
                      impl_yield: float | None = None) -> float:
        """Daily overnight carry/financing cost. Override in subclasses."""
        return 0.0


class SpreadCostModel(CostModel):
    """
    Uses actual bid/ask data for entry/exit costs.
    Cost = 0.5 * spread / mid + slippage_bps (one-way, as fraction of price).
    Rollover from impl_yield (preferred) or fwd_points / spot.
    """

    def __init__(self, slippage_bps: float = 0.0):
        self.slippage = slippage_bps / 10_000

    def _half_spread(self, bid, ask):
        mid = (bid + ask) / 2
        return (ask - bid) / 2 / mid if mid != 0 else 0.0

    def entry_cost(self, pair, price, size, bid=None, ask=None, **kwargs):
        hs = self._half_spread(bid, ask) if bid is not None and ask is not None else 0.0
        return (hs + self.slippage) * abs(size)

    def exit_cost(self, pair, price, size, bid=None, ask=None, **kwargs):
        return self.entry_cost(pair, price, size, bid=bid, ask=ask)

    def rollover_cost(self, pair, size, fwd_points=None, price=None, impl_yield=None):
        if impl_yield is not None:
            return (impl_yield / 260) * size
        if fwd_points is not None and price and price != 0:
            return (fwd_points / price / 365) * size
        return 0.0


class FixedSpreadModel(CostModel):
    """
    Fixed pip-spread model. Useful when bid/ask data is unavailable.

    Default spreads cover 25+ pairs. Override per-pair via spreads_pips.
    Rollover from impl_yield or fwd_points.
    """

    DEFAULT_SPREADS: dict[str, float] = {
        # G10 majors
        "EURUSD": 0.5,  "GBPUSD": 0.8,  "USDJPY": 0.5,  "USDCHF": 0.8,
        "USDCAD": 0.8,  "AUDUSD": 0.8,  "NZDUSD": 1.0,
        # G10 crosses
        "EURGBP": 1.0,  "EURJPY": 1.0,  "GBPJPY": 1.5,  "AUDJPY": 1.5,
        "EURCAD": 1.5,  "EURCHF": 1.0,  "GBPCAD": 2.0,  "CADJPY": 1.5,
        "NOKSEK": 2.0,  "EURNOK": 2.0,  "EURSEK": 2.0,
        # EM
        "USDMXN": 8.0,  "USDBRL": 12.0, "USDZAR": 8.0,  "USDTRY": 15.0,
        "USDPHP": 8.0,  "USDCNH": 5.0,  "USDINR": 5.0,  "USDIDR": 10.0,
        "USDKRW": 5.0,  "USDTHB": 5.0,  "USDSGD": 2.0,  "USDTWD": 5.0,
        "USDCLP": 15.0, "USDCOP": 15.0, "USDHUF": 5.0,  "USDCZK": 5.0,
        "USDPLN": 5.0,  "USDILS": 5.0,
    }
    JPY_PAIRS = {p for p in DEFAULT_SPREADS if "JPY" in p}

    def __init__(self, spreads_pips: dict[str, float] | None = None,
                 default_spread_pips: float = 3.0, slippage_bps: float = 0.0):
        self.spreads = {**self.DEFAULT_SPREADS, **(spreads_pips or {})}
        self.default_spread = default_spread_pips
        self.slippage = slippage_bps / 10_000

    def _half_spread_frac(self, pair, price):
        pips = self.spreads.get(pair.upper(), self.default_spread)
        pip_size = 0.01 if pair.upper() in self.JPY_PAIRS else 0.0001
        return (pips * pip_size / 2) / price if price else 0.0

    def entry_cost(self, pair, price, size, **kwargs):
        return (self._half_spread_frac(pair, price) + self.slippage) * abs(size)

    def exit_cost(self, pair, price, size, **kwargs):
        return self.entry_cost(pair, price, size)

    def rollover_cost(self, pair, size, fwd_points=None, price=None, impl_yield=None):
        if impl_yield is not None:
            return (impl_yield / 260) * size
        if fwd_points is not None and price and price != 0:
            return (fwd_points / price / 365) * size
        return 0.0
