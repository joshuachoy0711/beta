from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

if TYPE_CHECKING:
    from ..backtest.engine import BacktestResult

_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def plot_tearsheet(result: "BacktestResult", figsize: tuple = (16, 20), risk_free: float = 0.0):
    """Full 6-panel tear sheet."""
    notional = result.pnl_mode == "notional"
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        result.metadata.get("name", "Strategy") +
        f" — {'Notional PnL' if notional else 'Returns'} Tear Sheet",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_eq   = fig.add_subplot(gs[0, :])
    ax_dd   = fig.add_subplot(gs[1, :])
    ax_rs   = fig.add_subplot(gs[2, 0])
    ax_hm   = fig.add_subplot(gs[2, 1])
    ax_pair = fig.add_subplot(gs[3, 0])
    ax_tbl  = fig.add_subplot(gs[3, 1])

    _plot_equity(ax_eq, result)
    _plot_drawdown(ax_dd, result)
    _plot_rolling_sharpe(ax_rs, result)
    _plot_heatmap(ax_hm, result)
    _plot_pair_contribution(ax_pair, result)
    _plot_stats_table(ax_tbl, result, risk_free=risk_free)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    return fig


def plot_equity_curve(result, ax=None, **kwargs):
    fig, ax = _get_ax(ax)
    _plot_equity(ax, result)
    if fig: plt.tight_layout(); plt.show()


def plot_drawdown(result, ax=None, **kwargs):
    fig, ax = _get_ax(ax)
    _plot_drawdown(ax, result)
    if fig: plt.tight_layout(); plt.show()


def plot_rolling_sharpe(result, window: int = 52, ax=None):
    fig, ax = _get_ax(ax)
    _plot_rolling_sharpe(ax, result, window=window)
    if fig: plt.tight_layout(); plt.show()


def plot_monthly_heatmap(result, ax=None):
    fig, ax = _get_ax(ax)
    _plot_heatmap(ax, result)
    if fig: plt.tight_layout(); plt.show()


# -----------------------------------------------------------------------
# Internal
# -----------------------------------------------------------------------

def _plot_equity(ax, result):
    eq = result.equity_curve.dropna()
    gross_eq = result.gross_returns.dropna().cumsum() if result.pnl_mode == "notional" \
               else (1 + result.gross_returns.dropna()).cumprod()
    notional = result.pnl_mode == "notional"

    ax.plot(eq.index, eq.values, lw=2, color="steelblue", label="Net")
    ax.plot(gross_eq.index, gross_eq.values, lw=1, color="steelblue",
            alpha=0.4, linestyle="--", label="Gross")
    ax.axhline(0 if notional else 1, color="black", lw=0.8, linestyle=":")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Cumulative PnL (USD)" if notional else "Equity (1 = start)")
    if notional:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
    ax.legend(fontsize=8)
    _style(ax)


def _plot_drawdown(ax, result):
    if result.pnl_mode == "notional":
        cum = result.returns.dropna().cumsum()
        dd = cum - cum.cummax()
        ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_ylabel("Drawdown (USD)")
    else:
        from ..metrics.stats import drawdown_series
        dd = drawdown_series(result.returns.dropna())
        ax.fill_between(dd.index, dd.values * 100, 0, color="crimson", alpha=0.6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.set_ylabel("Drawdown (%)")
    ax.plot(dd.index, dd.values if result.pnl_mode == "notional" else dd.values * 100,
            color="crimson", lw=1)
    ax.set_title("Drawdown")
    _style(ax)


def _plot_rolling_sharpe(ax, result, window: int = 52):
    rs = result.rolling_sharpe(window=window).dropna()
    ax.plot(rs.index, rs.values, lw=1.5, color="darkorange")
    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.axhline(1, color="green", lw=0.8, linestyle="--", alpha=0.6)
    ax.set_title(f"Rolling Sharpe ({window}-period)")
    ax.set_ylabel("Sharpe")
    _style(ax)


def _plot_heatmap(ax, result):
    monthly = result.monthly_returns().reindex(columns=range(1, 13))
    vals = monthly.values * (1 if result.pnl_mode == "notional" else 100)
    vmin = -5 if result.pnl_mode == "returns" else None
    vmax = 5  if result.pnl_mode == "returns" else None
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(_MONTHS, fontsize=7)
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index.tolist(), fontsize=7)
    ax.set_title("Monthly Returns" + (" (USD)" if result.pnl_mode == "notional" else " (%)"))
    for i in range(len(monthly.index)):
        for j in range(12):
            v = vals[i, j]
            if not np.isnan(v):
                fmt = f"${v:,.0f}" if result.pnl_mode == "notional" else f"{v:.1f}"
                ax.text(j, i, fmt, ha="center", va="center", fontsize=5,
                        color="black" if abs(v) < (50000 if result.pnl_mode == "notional" else 3) else "white")
    plt.colorbar(im, ax=ax, fraction=0.03)


def _plot_pair_contribution(ax, result):
    pair_total = result.pair_returns.sum().sort_values()
    mult = 1 if result.pnl_mode == "notional" else 100
    colors = ["crimson" if v < 0 else "steelblue" for v in pair_total.values]
    ax.barh(pair_total.index, pair_total.values * mult, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Per-Pair Total Contribution")
    if result.pnl_mode == "notional":
        ax.set_xlabel("Total PnL (USD)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    else:
        ax.set_xlabel("Total Return (%)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    _style(ax)


def _plot_stats_table(ax, result, risk_free=0.0):
    s = result.summary(risk_free=risk_free)
    ax.axis("off")
    t = ax.table(
        cellText=[[v] for v in s.iloc[:, 0].values],
        rowLabels=s.index.tolist(),
        colLabels=[s.columns[0]],
        cellLoc="center", loc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.4)
    ax.set_title("Performance Summary", pad=10)


def _get_ax(ax):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
        return fig, ax
    return None, ax


def _style(ax):
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
