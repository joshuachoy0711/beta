from __future__ import annotations

"""
Forward curve plotting functions.

All functions accept the output of CurveBuilder and produce publication-quality
matplotlib charts. No Bloomberg connection required at plot time.

Functions
---------
plot_curve_today        — Single snapshot: today's forward curve
plot_curve_history      — Overlay of multiple historical snapshots on one chart
plot_curve_heatmap      — Heatmap of outright levels over time × tenor
plot_percentile_bands   — Today's curve vs historical percentile bands (fan chart)
plot_implied_yields     — Implied yield term structure (today + history)
plot_curve_dashboard    — 4-panel summary: all of the above in one figure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from .tenor_builder import TENORS, TENOR_DAYS, CurveBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tenor_positions(tenors: list[str]) -> list[float]:
    """Convert tenors to approximate log-day positions for x-axis spacing."""
    return [np.log1p(TENOR_DAYS.get(t, 21)) for t in tenors]


def _style(ax):
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _tenor_cols(df: pd.DataFrame, tenors: list[str] | None = None) -> list[str]:
    """Return tenor columns present in df, in TENORS order."""
    available = [c for c in TENORS if c in df.columns]
    if tenors:
        available = [t for t in tenors if t in available]
    return available


# ---------------------------------------------------------------------------
# 1. Today's curve snapshot
# ---------------------------------------------------------------------------

def plot_curve_today(
    curve: pd.Series,
    pair: str = "",
    ax=None,
    title: str | None = None,
    color: str = "steelblue",
    show_labels: bool = True,
) -> plt.Figure:
    """
    Plot a single forward curve snapshot (one date).

    Parameters
    ----------
    curve      : pd.Series  index = tenors, values = outright rates.
                            Output of ``CurveBuilder.fetch_curve()``.
    pair       : str        Pair name for the chart title.
    ax         : Axes       Existing axes to draw on. Creates new figure if None.
    title      : str        Custom title.
    color      : str        Line colour.
    show_labels: bool       Annotate each data point with its value.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _get_ax(ax, figsize=(10, 4))

    tenors = [t for t in TENORS if t in curve.index]
    vals   = curve[tenors].values
    xpos   = _tenor_positions(tenors)

    ax.plot(xpos, vals, "o-", color=color, lw=2, ms=6, zorder=3)

    if show_labels:
        for x, v, t in zip(xpos, vals, tenors):
            if not np.isnan(v):
                ax.annotate(f"{v:.4f}", (x, v),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=7.5, color=color)

    ax.set_xticks(xpos)
    ax.set_xticklabels(tenors, fontsize=9)
    ax.set_ylabel("Outright Rate")
    ax.set_title(title or f"{pair} Forward Curve", fontweight="bold")
    _style(ax)

    if fig:
        plt.tight_layout()
        plt.show()
    return fig or ax.get_figure()


# ---------------------------------------------------------------------------
# 2. Historical curve overlay
# ---------------------------------------------------------------------------

def plot_curve_history(
    history: pd.DataFrame,
    pair: str = "",
    snapshot_dates: list[str] | None = None,
    tenors: list[str] | None = None,
    ax=None,
    title: str | None = None,
) -> plt.Figure:
    """
    Overlay multiple historical snapshots of the forward curve on one chart.

    Parameters
    ----------
    history        : pd.DataFrame
        Output of ``CurveBuilder.fetch_history()``.
        Columns include 'spot' and tenor names. Index = DatetimeIndex.
    pair           : str
        Pair name for the chart title.
    snapshot_dates : list[str], optional
        Specific dates to plot, e.g. ['2024-01-01', '2024-06-01'].
        If None, plots today + 1M ago + 3M ago + 6M ago + 1Y ago.
    tenors         : list[str], optional
        Tenors to include. Default = all present.
    ax             : Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _get_ax(ax, figsize=(11, 5))

    cols = _tenor_cols(history, tenors)
    data = history[cols].dropna(how="all")

    if snapshot_dates:
        dates = [pd.Timestamp(d) for d in snapshot_dates]
        dates = [data.index[data.index.get_indexer([d], method="nearest")[0]] for d in dates]
    else:
        last = data.index[-1]
        candidates = {
            "Today":  last,
            "1M ago": last - pd.offsets.BDay(21),
            "3M ago": last - pd.offsets.BDay(63),
            "6M ago": last - pd.offsets.BDay(126),
            "1Y ago": last - pd.offsets.BDay(252),
        }
        dates = []
        labels_ordered = []
        for lbl, d in candidates.items():
            idx = data.index.get_indexer([d], method="nearest")[0]
            if 0 <= idx < len(data):
                dates.append(data.index[idx])
                labels_ordered.append(lbl)

    cmap   = cm.Blues
    colors = [cmap(0.4 + 0.6 * i / max(len(dates) - 1, 1)) for i in range(len(dates))]
    xpos   = _tenor_positions(cols)

    for i, d in enumerate(dates):
        if d not in data.index:
            continue
        row    = data.loc[d]
        label  = labels_ordered[i] if snapshot_dates is None else d.strftime("%d %b %Y")
        lw     = 2.5 if i == len(dates) - 1 else 1.3
        zorder = 3 if i == len(dates) - 1 else 2
        ax.plot(xpos, row[cols].values, "o-", color=colors[i],
                lw=lw, ms=5, label=label, zorder=zorder)

    ax.set_xticks(xpos)
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_ylabel("Outright Rate")
    ax.set_title(title or f"{pair} Forward Curve — Historical Snapshots", fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    _style(ax)

    if fig:
        plt.tight_layout()
        plt.show()
    return fig or ax.get_figure()


# ---------------------------------------------------------------------------
# 3. Percentile bands (fan chart)
# ---------------------------------------------------------------------------

def plot_percentile_bands(
    history: pd.DataFrame,
    pair: str = "",
    tenors: list[str] | None = None,
    ax=None,
    title: str | None = None,
    hist_window: int | None = None,
) -> plt.Figure:
    """
    Today's forward curve vs historical percentile bands.

    Shades the 10th–90th and 25th–75th percentile ranges of historical
    outright rates, with the median and today's curve overlaid.

    Parameters
    ----------
    history      : pd.DataFrame  Output of ``CurveBuilder.fetch_history()``.
    pair         : str
    tenors       : list[str]     Tenors to include.
    ax           : Axes, optional
    hist_window  : int, optional Rolling window (business days) for percentile
                                 calculation. None = full history.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _get_ax(ax, figsize=(11, 5))

    cols = _tenor_cols(history, tenors)
    data = history[cols].dropna(how="all")

    if hist_window:
        data_hist = data.iloc[-(hist_window):]
    else:
        data_hist = data

    cb = CurveBuilder()
    bands = cb.percentile_bands(data_hist, tenors=cols, pcts=(10, 25, 50, 75, 90))

    xpos  = _tenor_positions(cols)
    today = data.iloc[-1][cols].values

    ax.fill_between(xpos, bands.loc["p10"].values, bands.loc["p90"].values,
                    color="steelblue", alpha=0.15, label="10th–90th pct")
    ax.fill_between(xpos, bands.loc["p25"].values, bands.loc["p75"].values,
                    color="steelblue", alpha=0.30, label="25th–75th pct")
    ax.plot(xpos, bands.loc["p50"].values, "--", color="steelblue",
            lw=1.5, label="Median")
    ax.plot(xpos, today, "o-", color="crimson", lw=2.5, ms=6,
            label=f"Today ({data.index[-1].strftime('%d %b %Y')})", zorder=5)

    ax.set_xticks(xpos)
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_ylabel("Outright Rate")
    win_lbl = f"last {hist_window}d" if hist_window else "full history"
    ax.set_title(title or f"{pair} Forward Curve vs Historical Percentiles ({win_lbl})",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    _style(ax)

    if fig:
        plt.tight_layout()
        plt.show()
    return fig or ax.get_figure()


# ---------------------------------------------------------------------------
# 4. Implied yield term structure
# ---------------------------------------------------------------------------

def plot_implied_yields(
    history: pd.DataFrame,
    pair: str = "",
    tenors: list[str] | None = None,
    snapshot_dates: list[str] | None = None,
    ax=None,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot the implied yield term structure across tenors.

    Converts outright rates to annualised implied yields and overlays
    historical snapshots, identical in structure to ``plot_curve_history``
    but expressed as yield % rather than outright rate.

    Parameters
    ----------
    history        : pd.DataFrame  Output of ``CurveBuilder.fetch_history()``.
                                   Must include 'spot' column.
    pair           : str
    tenors         : list[str]
    snapshot_dates : list[str]     Specific dates to overlay.
    ax             : Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    cb = CurveBuilder()
    yield_history = cb.implied_yield_curve(history, tenors=tenors)

    # Delegate to history overlay, just with yield data
    yield_with_spot = yield_history.copy()

    fig, ax = _get_ax(ax, figsize=(11, 5))

    cols = _tenor_cols(yield_history, tenors)
    data = yield_history[cols].dropna(how="all")

    if snapshot_dates:
        dates = [pd.Timestamp(d) for d in snapshot_dates]
        dates = [data.index[data.index.get_indexer([d], method="nearest")[0]] for d in dates]
        labels_ordered = [pd.Timestamp(d).strftime("%d %b %Y") for d in snapshot_dates]
    else:
        last = data.index[-1]
        candidates = {
            "Today":  last,
            "1M ago": last - pd.offsets.BDay(21),
            "3M ago": last - pd.offsets.BDay(63),
            "6M ago": last - pd.offsets.BDay(126),
            "1Y ago": last - pd.offsets.BDay(252),
        }
        dates = []
        labels_ordered = []
        for lbl, d in candidates.items():
            idx = data.index.get_indexer([d], method="nearest")[0]
            if 0 <= idx < len(data):
                dates.append(data.index[idx])
                labels_ordered.append(lbl)

    cmap   = cm.Oranges
    colors = [cmap(0.4 + 0.6 * i / max(len(dates) - 1, 1)) for i in range(len(dates))]
    xpos   = _tenor_positions(cols)

    for i, d in enumerate(dates):
        if d not in data.index:
            continue
        row   = data.loc[d]
        lw    = 2.5 if i == len(dates) - 1 else 1.3
        label = labels_ordered[i]
        ax.plot(xpos, row[cols].values * 100, "o-", color=colors[i],
                lw=lw, ms=5, label=label,
                zorder=3 if i == len(dates) - 1 else 2)

    ax.axhline(0, color="black", lw=0.7, linestyle=":")
    ax.set_xticks(xpos)
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_ylabel("Implied Yield (% ann.)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))
    ax.set_title(title or f"{pair} Implied Yield Term Structure", fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    _style(ax)

    if fig:
        plt.tight_layout()
        plt.show()
    return fig or ax.get_figure()


# ---------------------------------------------------------------------------
# 5. Heatmap: outright level over time × tenor
# ---------------------------------------------------------------------------

def plot_curve_heatmap(
    history: pd.DataFrame,
    pair: str = "",
    tenors: list[str] | None = None,
    resample: str = "1W",
    ax=None,
    title: str | None = None,
) -> plt.Figure:
    """
    Heatmap of forward outright levels over time vs tenor.

    X-axis = dates, Y-axis = tenors, colour = outright level.
    Useful for seeing how the curve has shifted across the full history.

    Parameters
    ----------
    history  : pd.DataFrame  Output of ``CurveBuilder.fetch_history()``.
    pair     : str
    tenors   : list[str]
    resample : str           Resample frequency for x-axis density. Default '1W'.
    ax       : Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _get_ax(ax, figsize=(14, 5))

    cols = _tenor_cols(history, tenors)
    data = history[cols].dropna(how="all")

    if resample:
        data = data.resample(resample).last().dropna(how="all")

    mat = data[cols].T.values.astype(float)

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                   extent=[0, mat.shape[1], 0, mat.shape[0]])

    ax.set_yticks(np.arange(len(cols)) + 0.5)
    ax.set_yticklabels(cols[::-1], fontsize=8)

    n_ticks = min(12, mat.shape[1])
    tick_idx = np.linspace(0, mat.shape[1] - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx + 0.5)
    ax.set_xticklabels(
        [data.index[i].strftime("%b %y") for i in tick_idx],
        rotation=45, ha="right", fontsize=8,
    )

    plt.colorbar(im, ax=ax, fraction=0.015, label="Outright Rate")
    ax.set_title(title or f"{pair} Forward Curve — Level Heatmap Over Time",
                 fontweight="bold")

    if fig:
        plt.tight_layout()
        plt.show()
    return fig or ax.get_figure()


# ---------------------------------------------------------------------------
# 6. Dashboard: 4-panel summary
# ---------------------------------------------------------------------------

def plot_curve_dashboard(
    history: pd.DataFrame,
    pair: str = "",
    tenors: list[str] | None = None,
    hist_window: int | None = None,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """
    4-panel forward curve dashboard for a single pair.

    Panels:
    - Top-left:  Today's curve vs historical snapshots
    - Top-right: Percentile fan chart
    - Bottom-left: Implied yield term structure
    - Bottom-right: Level heatmap over time

    Parameters
    ----------
    history      : pd.DataFrame  Output of ``CurveBuilder.fetch_history()``.
    pair         : str
    tenors       : list[str]
    hist_window  : int           Rolling window for percentile bands.
    figsize      : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"{pair} Forward Curve Dashboard", fontsize=14, fontweight="bold", y=1.01)

    plot_curve_history(    history, pair=pair, tenors=tenors, ax=axes[0, 0])
    plot_percentile_bands( history, pair=pair, tenors=tenors, ax=axes[0, 1],
                           hist_window=hist_window)
    plot_implied_yields(   history, pair=pair, tenors=tenors, ax=axes[1, 0])
    plot_curve_heatmap(    history, pair=pair, tenors=tenors, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 7. Single-tenor time series: track one tenor vs spot over time
# ---------------------------------------------------------------------------

def plot_tenor_vs_spot(
    history: pd.DataFrame,
    pair: str = "",
    tenor: str = "1M",
    ax=None,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot a single tenor's outright rate alongside spot over the full history.

    Useful for seeing whether the forward premium/discount is widening or
    narrowing, and comparing current level vs historical range.

    Parameters
    ----------
    history : pd.DataFrame  Must contain 'spot' and ``tenor`` columns.
    pair    : str
    tenor   : str           Tenor to plot. Default '1M'.
    ax      : Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if "spot" not in history.columns:
        raise ValueError("history must contain a 'spot' column")
    if tenor not in history.columns:
        raise ValueError(f"tenor '{tenor}' not in history columns: {list(history.columns)}")

    fig, axes = _get_ax(ax, figsize=(14, 6), n=2, sharex=True)
    if ax is not None:
        axes = [ax, ax]

    data = history[["spot", tenor]].dropna()

    axes[0].plot(data.index, data["spot"],  lw=1.5, color="steelblue", label="Spot")
    axes[0].plot(data.index, data[tenor],   lw=1.5, color="darkorange", label=f"{tenor} Fwd", alpha=0.8)
    axes[0].set_title(title or f"{pair} Spot vs {tenor} Forward", fontweight="bold")
    axes[0].set_ylabel("Rate")
    axes[0].legend(fontsize=8)
    _style(axes[0])

    # Forward premium / discount (in pips equivalent)
    spread = data[tenor] - data["spot"]
    axes[1].fill_between(spread.index, spread.values, 0,
                         where=spread >= 0, color="steelblue", alpha=0.4, label="Fwd premium")
    axes[1].fill_between(spread.index, spread.values, 0,
                         where=spread < 0,  color="crimson",   alpha=0.4, label="Fwd discount")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title(f"{tenor} Forward Premium / Discount", fontweight="bold")
    axes[1].set_ylabel("Fwd – Spot")
    axes[1].legend(fontsize=8)
    _style(axes[1])

    if fig:
        plt.tight_layout()
        plt.show()
    return fig or axes[0].get_figure()


# ---------------------------------------------------------------------------
# Axis helper
# ---------------------------------------------------------------------------

def _get_ax(ax, figsize=(12, 5), n=1, sharex=False):
    if ax is not None:
        return None, ax
    if n == 1:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=sharex)
    return fig, list(axes)
