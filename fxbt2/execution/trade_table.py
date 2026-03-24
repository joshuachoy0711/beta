from __future__ import annotations

"""
Trade execution table builder.
Ported from rcq_trading_library.ccy_pair_trade_table.

Computes the delta-to-trade (today's target vs yesterday's position),
attaches reference rates, and resolves the IMM settlement date.

Requires Bloomberg Terminal connection (via PdblpLoader).
"""

import datetime as dt
import pandas as pd


def build_trade_table(
    tradesize: pd.DataFrame,
    ref_rates: pd.DataFrame | None = None,
    ndf_pairs: list[str] | None = None,
    pdblp_connection=None,
    fixing: str = "CMPT",
) -> pd.DataFrame:
    """
    Build an execution trade table showing what needs to be traded today.

    Parameters
    ----------
    tradesize         : pd.DataFrame
        Wide-format target position sizes (USD notional) per pair.
        index = DatetimeIndex, columns = pair names.
        Uses the last two rows (today and yesterday).
    ref_rates         : pd.DataFrame, optional
        Reference rates for each pair (spot or NDF outright as appropriate).
        If None and pdblp_connection is provided, rates are fetched live.
    ndf_pairs         : list[str], optional
        Pairs that trade as NDFs (use forward outright as reference rate).
        Defaults to ['USDBRL','USDCLP','USDCOP','USDKRW','USDINR',
                     'USDIDR','USDTWD','USDPHP'].
    pdblp_connection  : pdblp.BCon, optional
        Live Bloomberg connection. Required for live rate fetch and IMM dates.
    fixing            : str
        Bloomberg fixing code for live rate fetch. Default 'CMPT'.

    Returns
    -------
    pd.DataFrame with columns:
        target_rate       current spot or NDF rate
        {today}_pos       today's target position (USD)
        {yesterday}_pos   yesterday's position (USD)
        to_trade_usd      delta to execute (+ = buy base, - = sell base)
        val_date          IMM settlement date
    """
    _NDF_DEFAULT = {
        "USDBRL", "USDCLP", "USDCOP", "USDKRW",
        "USDINR", "USDIDR", "USDTWD", "USDPHP",
    }
    ndf_set = set(ndf_pairs or _NDF_DEFAULT)

    pairs       = tradesize.columns.tolist()
    today_ts    = tradesize.index[-1]
    ydy_ts      = tradesize.index[-2]
    today_str   = today_ts.strftime("%d-%m-%Y")
    ydy_str     = ydy_ts.strftime("%d-%m-%Y")

    # ---- Reference rates ----
    if ref_rates is not None:
        rate = ref_rates.iloc[-1]
    elif pdblp_connection is not None:
        rate = _fetch_live_rates(pdblp_connection, pairs, ndf_set, fixing, today_ts)
    else:
        rate = pd.Series(index=pairs, dtype=float)

    # ---- IMM date ----
    val_date = _get_imm_date(pdblp_connection) if pdblp_connection is not None else "T+2"

    # ---- Build table ----
    df = pd.DataFrame({"target_rate": rate[pairs]})
    df[f"{today_str}_pos"] = tradesize.iloc[-1]
    df[f"{ydy_str}_pos"]   = tradesize.iloc[-2]
    df["to_trade_usd"]     = tradesize.iloc[-1] - tradesize.iloc[-2]
    df["val_date"]         = val_date

    return df


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fetch_live_rates(con, pairs, ndf_set, fixing, date):
    """Pull today's spot or NDF outright rate for each pair."""
    from ..data.ticker_dict import get_ticker, NDF_CCYS
    date_str = date.strftime("%Y%m%d")
    rates = {}
    for pair in pairs:
        base_ccy = pair[:3] if pair[:3] != "USD" else pair[3:]
        is_ndf = pair.upper() in ndf_set or base_ccy in NDF_CCYS
        asset = "1m_ndf_outright" if is_ndf else "spot"
        ticker = get_ticker(base_ccy, asset, fixing)
        if ticker is None:
            ticker = f"{pair} {fixing} Curncy"
        try:
            raw = con.bdh([ticker], ["PX_LAST"], date_str, date_str)
            rates[pair] = float(raw.iloc[-1, -1])
        except Exception:
            rates[pair] = float("nan")
    return pd.Series(rates)


def _get_imm_date(con) -> str:
    """
    Return the next IMM settlement date.
    Roll to IMM2 if within 14 days of IMM1.
    """
    try:
        imm1 = con.ref("EUIM1 Curncy", "settle_dt")["value"].iloc[-1]
        imm2 = con.ref("EUIM2 Curncy", "settle_dt")["value"].iloc[-1]
        today = dt.date.today()
        if imm1 - dt.timedelta(days=14) <= today:
            return imm2.strftime("%d-%b-%Y")
        return imm1.strftime("%d-%b-%Y")
    except Exception:
        return "T+2"
