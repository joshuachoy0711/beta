from __future__ import annotations

"""
Performance metrics. Works in both returns mode (fractions) and notional mode (USD PnL).
"""

import numpy as np
import pandas as pd

_ANN_FACTORS = {
    "1D": 252, "D": 252, "1W": 52, "W": 52, "1M": 12, "M": 12,
    "1H": 252 * 24, "H": 252 * 24, "1h": 252 * 24,
    "60min": 252 * 24, "15min": 252 * 24 * 4,
    "5min": 252 * 24 * 12, "1min": 252 * 24 * 60,
}


def _ann(freq: str) -> int:
    if freq in _ANN_FACTORS:
        return _ANN_FACTORS[freq]
    raise ValueError(f"Unknown freq '{freq}'. Supported: {list(_ANN_FACTORS.keys())}")


def _clean(r: pd.Series) -> pd.Series:
    return r.dropna().replace([np.inf, -np.inf], np.nan).dropna()


def annualised_return(returns: pd.Series, freq: str = "1D", ann_factor: int = None) -> float:
    af = ann_factor or _ann(freq)
    return _clean(returns).mean() * af


def annualised_vol(returns: pd.Series, freq: str = "1D", ann_factor: int = None) -> float:
    af = ann_factor or _ann(freq)
    return _clean(returns).std() * np.sqrt(af)


def sharpe(returns: pd.Series, freq: str = "1D", risk_free: float = 0.0,
           ann_factor: int = None) -> float:
    af = ann_factor or _ann(freq)
    r = _clean(returns)
    excess = r - risk_free / af
    return excess.mean() / excess.std() * np.sqrt(af) if excess.std() > 0 else np.nan


def sortino(returns: pd.Series, freq: str = "1D", risk_free: float = 0.0,
            ann_factor: int = None) -> float:
    af = ann_factor or _ann(freq)
    r = _clean(returns)
    excess = r - risk_free / af
    down = excess[excess < 0]
    if len(down) == 0 or down.std() == 0:
        return np.nan
    return excess.mean() * af / (down.std() * np.sqrt(af))


def calmar(returns: pd.Series, freq: str = "1D", ann_factor: int = None) -> float:
    af = ann_factor or _ann(freq)
    mdd = abs(max_drawdown(returns))
    return annualised_return(returns, ann_factor=af) / mdd if mdd != 0 else np.nan


def drawdown_series(returns: pd.Series) -> pd.Series:
    r = _clean(returns)
    cum = (1 + r).cumprod()
    return (cum - cum.cummax()) / cum.cummax()


def max_drawdown(returns: pd.Series) -> float:
    return drawdown_series(_clean(returns)).min()


def hit_rate(returns: pd.Series) -> float:
    r = _clean(returns)
    r = r[r != 0]
    return (r > 0).sum() / len(r) if len(r) > 0 else np.nan


def profit_factor(returns: pd.Series) -> float:
    r = _clean(returns)
    gains  = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    return gains / losses if losses != 0 else np.inf


def avg_win_loss_ratio(returns: pd.Series) -> float:
    r = _clean(returns)
    avg_win  = r[r > 0].mean()
    avg_loss = abs(r[r < 0].mean())
    return avg_win / avg_loss if avg_loss != 0 else np.inf


def var(returns: pd.Series, level: float = 0.05) -> float:
    return _clean(returns).quantile(level)


def cvar(returns: pd.Series, level: float = 0.05) -> float:
    r = _clean(returns)
    return r[r <= var(r, level)].mean()


def summary(
    returns: pd.Series,
    freq: str = "1D",
    risk_free: float = 0.0,
    ann_factor: int = None,
    name: str = "Strategy",
    pnl_mode: str = "returns",
) -> pd.DataFrame:
    """Full performance summary table."""
    af = ann_factor or _ann(freq)
    r = _clean(returns)

    if pnl_mode == "notional":
        # Notional mode: format in USD
        stats = {
            "Total PnL (USD)":     f"${r.sum():,.0f}",
            "Ann. PnL (USD)":      f"${r.mean() * af:,.0f}",
            "Ann. Volatility (USD)":f"${r.std() * np.sqrt(af):,.0f}",
            "Sharpe Ratio":        f"{sharpe(r, ann_factor=af, risk_free=risk_free):.2f}",
            "Sortino Ratio":       f"{sortino(r, ann_factor=af, risk_free=risk_free):.2f}",
            "Max Drawdown (USD)":  f"${r.cumsum().sub(r.cumsum().cummax()).min():,.0f}",
            "Hit Rate":            f"{hit_rate(r):.2%}",
            "Profit Factor":       f"{profit_factor(r):.2f}",
            "Avg Win/Loss":        f"{avg_win_loss_ratio(r):.2f}",
            "VaR 95% (USD/day)":   f"${var(r, 0.05):,.0f}",
            "CVaR 95% (USD/day)":  f"${cvar(r, 0.05):,.0f}",
            "Obs.":                str(len(r)),
            "Start":               str(r.index[0].date()) if hasattr(r.index[0], "date") else str(r.index[0]),
            "End":                 str(r.index[-1].date()) if hasattr(r.index[-1], "date") else str(r.index[-1]),
        }
    else:
        stats = {
            "Ann. Return":      f"{annualised_return(r, ann_factor=af):.2%}",
            "Ann. Volatility":  f"{annualised_vol(r, ann_factor=af):.2%}",
            "Sharpe Ratio":     f"{sharpe(r, ann_factor=af, risk_free=risk_free):.2f}",
            "Sortino Ratio":    f"{sortino(r, ann_factor=af, risk_free=risk_free):.2f}",
            "Calmar Ratio":     f"{calmar(r, ann_factor=af):.2f}",
            "Max Drawdown":     f"{max_drawdown(r):.2%}",
            "Hit Rate":         f"{hit_rate(r):.2%}",
            "Profit Factor":    f"{profit_factor(r):.2f}",
            "Avg Win/Loss":     f"{avg_win_loss_ratio(r):.2f}",
            "VaR (95%)":        f"{var(r, 0.05):.2%}",
            "CVaR (95%)":       f"{cvar(r, 0.05):.2%}",
            "Obs.":             str(len(r)),
            "Start":            str(r.index[0].date()) if hasattr(r.index[0], "date") else str(r.index[0]),
            "End":              str(r.index[-1].date()) if hasattr(r.index[-1], "date") else str(r.index[-1]),
        }

    return pd.DataFrame.from_dict(stats, orient="index", columns=[name])


def compare(*results: tuple[pd.Series, str], freq: str = "1D",
            risk_free: float = 0.0) -> pd.DataFrame:
    """Compare multiple strategies side by side."""
    frames = [summary(r, freq=freq, risk_free=risk_free, name=name) for r, name in results]
    return pd.concat(frames, axis=1)
