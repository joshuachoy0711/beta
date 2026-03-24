from __future__ import annotations

"""
Portfolio construction utilities.
Ported from rcq_trading_library:
  - generate_currency_pairs  → generate_pairs
  - rolling_net_currency_pairs → net_ccy_exposure
  - rolling_net_currency_pairs_to_usd_cross → net_ccy_exposure_usd
"""

import pandas as pd


def generate_pairs(ccy_codes: list[str]) -> list[str]:
    """
    Generate all unique FX pairs from a list of currency codes.

    Parameters
    ----------
    ccy_codes : list[str]   e.g. ['USD', 'EUR', 'MXN', 'ZAR']

    Returns
    -------
    list[str]   e.g. ['USDEUR', 'USDMXN', 'USDZAR', 'EURMXN', 'EURZAR', 'MXNZAR']

    Example
    -------
    generate_pairs(['USD', 'MXN', 'ZAR'])
    # → ['USDMXN', 'USDZAR', 'MXNZAR']
    """
    pairs = []
    for i in range(len(ccy_codes)):
        for j in range(i + 1, len(ccy_codes)):
            pairs.append(f"{ccy_codes[i]}{ccy_codes[j]}")
    return pairs


def net_ccy_exposure(tradesize: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pair-level trade sizes into single-currency net exposures.

    For each pair (e.g. EURUSD): base (EUR) gets +tradesize, quote (USD) gets -tradesize.
    Summing across all pairs gives the net exposure per currency.

    Useful for:
    - Risk monitoring: "what is my net EUR exposure across the book?"
    - Execution: computing which individual currency crosses to hedge

    Parameters
    ----------
    tradesize : pd.DataFrame
        Wide-format DataFrame of trade sizes per pair.
        index = DatetimeIndex, columns = pair names (e.g. 'EURUSD')

    Returns
    -------
    pd.DataFrame
        index = DatetimeIndex, columns = currency codes (e.g. 'EUR', 'USD')
    """
    basket = pd.DataFrame(index=tradesize.index)

    for pair in tradesize.columns:
        base  = pair[:3]
        quote = pair[3:]
        size  = tradesize[pair]

        basket[base]  = basket.get(base,  pd.Series(0.0, index=tradesize.index)) + size
        basket[quote] = basket.get(quote, pd.Series(0.0, index=tradesize.index)) - size

    return basket


def net_ccy_exposure_usd(tradesize: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pair-level trade sizes into net USD-cross exposures.

    Same as net_ccy_exposure but converts each netted currency position
    to its USD cross format: e.g. net EUR exposure → USDEUR column.

    Useful for computing the actual USD-denominated hedges needed.

    Parameters
    ----------
    tradesize : pd.DataFrame  wide-format trade sizes per pair

    Returns
    -------
    pd.DataFrame
        columns = 'USD{CCY}' format (e.g. 'USDEUR', 'USDJPY'),
        values  = net trade size in that USD cross
    """
    ccy_net = net_ccy_exposure(tradesize)
    usd_cross = pd.DataFrame(index=tradesize.index)

    for ccy in ccy_net.columns:
        if ccy == "USD":
            continue
        usd_cross[f"USD{ccy}"] = -ccy_net[ccy]

    return usd_cross
