from __future__ import annotations

"""
Core backtesting engine — unified returns and notional modes.

Returns mode  (pnl_mode='returns')  : positions are portfolio weights,
                                       PnL in % of portfolio.
Notional mode (pnl_mode='notional'): positions are USD notional,
                                       PnL in absolute USD dollars.
                                       Requires notional_size parameter.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..costs.model import CostModel, FixedSpreadModel
from .positions import fixed_size


@dataclass
class BacktestResult:
    """
    Container for all backtest outputs.

    Key attributes
    --------------
    returns       : pd.Series  net portfolio return per period (returns mode)
                               or net USD PnL per period (notional mode)
    pair_returns  : pd.DataFrame  per-pair breakdown
    gross_returns : pd.Series  before costs
    costs         : pd.Series  cost drag per period
    positions     : pd.DataFrame  sized positions (weights or USD notional)
    equity_curve  : pd.Series  cumulative equity (starts at 1.0 in returns mode,
                               starts at 0 in notional mode = cumulative PnL)
    pnl_mode      : str  'returns' or 'notional'
    freq          : str  return frequency
    metadata      : dict  strategy name, params, etc.
    """
    returns: pd.Series
    pair_returns: pd.DataFrame
    gross_returns: pd.Series
    costs: pd.Series
    positions: pd.DataFrame
    signals: pd.DataFrame
    prices: pd.DataFrame
    equity_curve: pd.Series
    pnl_mode: str = "returns"
    freq: str = "1D"
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self, risk_free: float = 0.0) -> pd.DataFrame:
        from ..metrics.stats import summary
        return summary(
            self.returns,
            freq=self.freq,
            risk_free=risk_free,
            name=self.metadata.get("name", "Strategy"),
            pnl_mode=self.pnl_mode,
        )

    def tearsheet(self, **kwargs):
        from ..report.tearsheet import plot_tearsheet
        plot_tearsheet(self, **kwargs)

    def compare_to(self, other: "BacktestResult") -> pd.DataFrame:
        from ..metrics.stats import compare
        return compare(
            (self.returns,  self.metadata.get("name", "Strategy A")),
            (other.returns, other.metadata.get("name", "Strategy B")),
            freq=self.freq,
        )

    def rolling_sharpe(self, window: int = 52) -> pd.Series:
        from ..metrics.stats import _ann
        af = _ann(self.freq)
        r = self.returns.dropna()
        return r.rolling(window).apply(
            lambda x: x.mean() / x.std() * np.sqrt(af) if x.std() > 0 else np.nan
        )

    def monthly_returns(self) -> pd.DataFrame:
        r = self.returns.copy()
        r.index = pd.to_datetime(r.index)
        if self.pnl_mode == "notional":
            monthly = r.resample("M").sum()
        else:
            monthly = r.resample("M").apply(lambda x: (1 + x).prod() - 1)
        df = monthly.to_frame("ret")
        df["year"]  = df.index.year
        df["month"] = df.index.month
        return df.pivot(index="year", columns="month", values="ret")


class Backtest:
    """
    Unified FX backtesting engine.

    Supports:
    - Single pair and multi-pair portfolio
    - Returns mode (% of portfolio) or Notional mode (absolute USD)
    - Pluggable cost models and position sizers
    - Walk-forward analysis
    - VaR-targeted sizing (notional mode)

    Parameters
    ----------
    data         : pd.DataFrame  wide prices (index=DatetimeIndex, cols=pairs)
                                 or long-format with 'pair'/'close' columns
    signals      : pd.DataFrame  direction signals, same shape as data
    cost_model   : CostModel     default: FixedSpreadModel()
    sizer        : callable      sizer(signals, prices) → positions DataFrame
    freq         : str           '1D', '1H', '15min', etc.
    name         : str           strategy label
    pnl_mode     : str           'returns' (default) or 'notional'
    notional_size: float         base USD notional per unit signal (notional mode only)
                                 e.g. 10_000_000 = $10M per signal
    bids/asks    : pd.DataFrame  optional bid/ask for SpreadCostModel
    fwd_points   : pd.DataFrame  optional 1M forward points for rollover
    impl_yield   : pd.DataFrame  optional implied yield differential for rollover
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        cost_model: CostModel | None = None,
        sizer=None,
        freq: str = "1D",
        name: str = "Strategy",
        pnl_mode: str = "returns",
        notional_size: float = 10_000_000,
        bids: pd.DataFrame | None = None,
        asks: pd.DataFrame | None = None,
        fwd_points: pd.DataFrame | None = None,
        impl_yield: pd.DataFrame | None = None,
    ):
        self.prices = self._to_wide(data)
        self.signals = signals.reindex(self.prices.index).fillna(0.0)
        self.cost_model = cost_model or FixedSpreadModel()
        self.sizer = sizer or (lambda s, p: fixed_size(s, size=1.0))
        self.freq = freq
        self.name = name
        self.pnl_mode = pnl_mode
        self.notional_size = notional_size
        self.bids = bids
        self.asks = asks
        self.fwd_points = fwd_points
        self.impl_yield = impl_yield

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        prices  = self.prices
        signals = self.signals.reindex(columns=prices.columns).fillna(0.0)

        # 1. Size positions
        positions = self.sizer(signals, prices)

        # In notional mode, scale up to real dollar amounts
        if self.pnl_mode == "notional":
            positions = positions * self.notional_size

        # 2. Gross returns
        price_ret = prices.pct_change()  # period return
        if self.pnl_mode == "notional":
            # PnL = position_usd[t-1] × price_return[t]
            gross_pair = positions.shift(1) * price_ret
        else:
            gross_pair = positions.shift(1) * price_ret

        # 3. Costs
        cost_df = self._compute_costs(positions, prices)

        # 4. Net returns
        net_pair = gross_pair - cost_df

        # 5. Portfolio aggregation
        portfolio_gross = gross_pair.sum(axis=1)
        portfolio_costs = cost_df.sum(axis=1)
        portfolio_net   = net_pair.sum(axis=1)

        # 6. Equity curve
        if self.pnl_mode == "notional":
            equity = portfolio_net.cumsum()   # cumulative USD PnL
        else:
            equity = (1 + portfolio_net).cumprod()

        return BacktestResult(
            returns=portfolio_net,
            pair_returns=net_pair,
            gross_returns=portfolio_gross,
            costs=portfolio_costs,
            positions=positions,
            signals=signals,
            prices=prices,
            equity_curve=equity,
            pnl_mode=self.pnl_mode,
            freq=self.freq,
            metadata={"name": self.name, "pnl_mode": self.pnl_mode},
        )

    # ------------------------------------------------------------------
    # Walk-forward
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        train_periods: int,
        test_periods: int,
        signal_fn,
        sizer=None,
    ) -> BacktestResult:
        """
        Rolling walk-forward backtest.

        signal_fn(train_prices, test_prices) → test_signals DataFrame
        """
        sizer = sizer or self.sizer
        prices = self.prices
        n = len(prices)

        all_r, all_pos, all_sig, all_cost = [], [], [], []

        start = 0
        while start + train_periods + test_periods <= n:
            train = prices.iloc[start: start + train_periods]
            test  = prices.iloc[start + train_periods: start + train_periods + test_periods]

            test_signals = signal_fn(train, test)
            fold = Backtest(
                data=test, signals=test_signals,
                cost_model=self.cost_model, sizer=sizer,
                freq=self.freq, name=self.name,
                pnl_mode=self.pnl_mode, notional_size=self.notional_size,
            )
            r = fold.run()
            all_r.append(r.returns)
            all_pos.append(r.positions)
            all_sig.append(r.signals)
            all_cost.append(r.costs)
            start += test_periods

        rets  = pd.concat(all_r).sort_index()
        pos   = pd.concat(all_pos).sort_index()
        sigs  = pd.concat(all_sig).sort_index()
        costs = pd.concat(all_cost).sort_index()
        equity = rets.cumsum() if self.pnl_mode == "notional" else (1 + rets).cumprod()

        return BacktestResult(
            returns=rets,
            pair_returns=pos * prices.reindex(rets.index).pct_change(),
            gross_returns=rets + costs,
            costs=costs,
            positions=pos,
            signals=sigs,
            prices=prices.reindex(rets.index),
            equity_curve=equity,
            pnl_mode=self.pnl_mode,
            freq=self.freq,
            metadata={"name": f"{self.name} (WF)", "walk_forward": True},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_wide(self, data: pd.DataFrame) -> pd.DataFrame:
        if "pair" in data.columns and "close" in data.columns:
            wide = data.pivot(columns="pair", values="close")
            wide.index.name = "timestamp"
            return wide
        return data

    def _compute_costs(self, positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        trades = positions.diff().fillna(positions)
        cost_df = pd.DataFrame(0.0, index=positions.index, columns=positions.columns)

        for pair in positions.columns:
            p_trade = trades[pair]
            p_pos   = positions[pair]
            p_price = prices[pair]

            bid_s = self.bids[pair]        if self.bids       is not None and pair in self.bids.columns       else None
            ask_s = self.asks[pair]        if self.asks       is not None and pair in self.asks.columns       else None
            fp_s  = self.fwd_points[pair]  if self.fwd_points is not None and pair in self.fwd_points.columns else None
            iy_s  = self.impl_yield[pair]  if self.impl_yield is not None and pair in self.impl_yield.columns else None

            for t in positions.index:
                trade = p_trade.loc[t]
                pos   = p_pos.loc[t]
                price = p_price.loc[t]
                if pd.isna(price):
                    continue

                bid = bid_s.loc[t] if bid_s is not None else None
                ask = ask_s.loc[t] if ask_s is not None else None
                fp  = fp_s.loc[t]  if fp_s  is not None else None
                iy  = iy_s.loc[t]  if iy_s  is not None else None

                tc = self.cost_model.entry_cost(pair, price, trade, bid=bid, ask=ask) if trade != 0 else 0.0
                rc = self.cost_model.rollover_cost(pair, pos, fwd_points=fp, price=price, impl_yield=iy) if pos != 0 else 0.0
                cost_df.loc[t, pair] = tc + rc

        return cost_df
