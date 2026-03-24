"""
fxbt2 — Comprehensive FX Quantitative Backtesting Library
==========================================================
Merges fxbt (research/backtesting) with rcq_trading_library (production/execution).

Modules
-------
data        — Data loaders: CSV, BQuant (BQL), pdblp (Terminal), forward builder
signals     — Signal generators: price-based, macro/yield-based
costs       — Transaction cost models: spread, slippage, rollover
backtest    — Core engine: returns mode and notional mode, walk-forward
portfolio   — Basket construction, net currency exposure, risk attribution
metrics     — Performance statistics: Sharpe, Sortino, VaR, drawdown, etc.
report      — Tear sheet: equity curve, drawdown, heatmap, rolling stats
execution   — Trade table, IMM dates (requires Bloomberg Terminal)
"""

from .backtest.engine import Backtest, BacktestResult
from .data.csv_loader import CSVLoader
from .data.bquant_loader import BQuantLoader
from .data.pdblp_loader import PdblpLoader
from . import signals, costs, metrics, report, portfolio, execution

__version__ = "2.0.0"

__all__ = [
    "Backtest",
    "BacktestResult",
    "CSVLoader",
    "BQuantLoader",
    "PdblpLoader",
    "signals",
    "costs",
    "metrics",
    "report",
    "portfolio",
    "execution",
]
