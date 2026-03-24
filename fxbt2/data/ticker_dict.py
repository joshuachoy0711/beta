from __future__ import annotations

"""
Bloomberg ticker reference dictionary.
Ported and extended from rcq_trading_library.bbg_ticker_dict().

Covers 25+ currencies with:
  spot, 1m_fwd_pts, 1m_ndf_outright, 1m_impl_yield, 1m_impl_vol,
  2y_irs, 5y_irs, 10y_irs, equity_index, bond_index, tot_index

Usage
-----
from fxbt2.data.ticker_dict import BBG_TICKER_DICT, get_ticker, get_ccy_list

# Get EURUSD spot ticker for CMPT fixing
get_ticker('EUR', 'spot', fixing='CMPT')
# → 'USDEUR CMPT Curncy'

# List all supported currencies
get_ccy_list()
"""

import pandas as pd
from typing import Optional

# ------------------------------------------------------------------
# Raw ticker templates  {ccy: {asset_type: ticker_template}}
# {FIX} is replaced with the fixing timestamp at query time.
# None = not available for that currency/asset combination.
# ------------------------------------------------------------------

_TICKER_TEMPLATES: dict[str, dict[str, Optional[str]]] = {
    "USD": {
        "spot":              "USD Curncy",
        "1m_fwd_pts":        None,
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "USOSFRA {FIX} Curncy",
        "1m_impl_vol":       None,
        "2y_irs":            "USOSFR2 {FIX} Curncy",
        "5y_irs":            "USOSFR5 {FIX} Curncy",
        "10y_irs":           "USOSFR10 {FIX} Curncy",
        "equity_index":      "IVV US Equity",
        "bond_index":        "AGG US Equity",
        "tot_index":         "CTOTUSD Index",
    },
    "EUR": {
        "spot":              "USDEUR {FIX} Curncy",
        "1m_fwd_pts":        "EUR1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "EURI1M {FIX} Curncy",
        "1m_impl_vol":       "EURUSDV1M {FIX} Curncy",
        "2y_irs":            "EUSA2 {FIX} Curncy",
        "5y_irs":            "EUSA5 {FIX} Curncy",
        "10y_irs":           "EUSA10 {FIX} Curncy",
        "equity_index":      "IEV US Equity",
        "bond_index":        "SPBDEGIT Index",
        "tot_index":         "CTOTEUR Index",
    },
    "GBP": {
        "spot":              "USDGBP {FIX} Curncy",
        "1m_fwd_pts":        "GBP1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "GBPI1M {FIX} Curncy",
        "1m_impl_vol":       "GBPUSDV1M {FIX} Curncy",
        "2y_irs":            "BPSWS2 {FIX} Curncy",
        "5y_irs":            "BPSWS5 {FIX} Curncy",
        "10y_irs":           "BPSWS10 {FIX} Curncy",
        "equity_index":      "EWU US Equity",
        "bond_index":        "SPFIGBT Index",
        "tot_index":         "CTOTGBP Index",
    },
    "JPY": {
        "spot":              "USDJPY {FIX} Curncy",
        "1m_fwd_pts":        "JPY1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "JPYI1M {FIX} Curncy",
        "1m_impl_vol":       "USDJPYV1M {FIX} Curncy",
        "2y_irs":            "JYSO2 {FIX} Curncy",
        "5y_irs":            "JYSO5 {FIX} Curncy",
        "10y_irs":           "JYSO10 {FIX} Curncy",
        "equity_index":      "EWJ US Equity",
        "bond_index":        "SPBJPCOT Index",
        "tot_index":         "CTOTJPY Index",
    },
    "AUD": {
        "spot":              "USDAUD {FIX} Curncy",
        "1m_fwd_pts":        "AUD1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "AUDI1M {FIX} Curncy",
        "1m_impl_vol":       "AUDUSDV1M {FIX} Curncy",
        "2y_irs":            "ADSW2 {FIX} Curncy",
        "5y_irs":            "ADSW5 {FIX} Curncy",
        "10y_irs":           "ADSW10 {FIX} Curncy",
        "equity_index":      "EWA US Equity",
        "bond_index":        "VGB AU Equity",
        "tot_index":         "CTOTAUD Index",
    },
    "NZD": {
        "spot":              "USDNZD {FIX} Curncy",
        "1m_fwd_pts":        "NZD1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "NZDI1M {FIX} Curncy",
        "1m_impl_vol":       "NZDUSDV1M {FIX} Curncy",
        "2y_irs":            "NDSW2 {FIX} Curncy",
        "5y_irs":            "NDSW5 {FIX} Curncy",
        "10y_irs":           "NDSW10 {FIX} Curncy",
        "equity_index":      "ENZL US Equity",
        "bond_index":        "SBNZL Index",
        "tot_index":         "CTOTNZD Index",
    },
    "CAD": {
        "spot":              "USDCAD {FIX} Curncy",
        "1m_fwd_pts":        "CAD1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "CADI1M {FIX} Curncy",
        "1m_impl_vol":       "USDCADV1M {FIX} Curncy",
        "2y_irs":            "CDSW2 {FIX} Curncy",
        "5y_irs":            "CDSW5 {FIX} Curncy",
        "10y_irs":           "CDSW10 {FIX} Curncy",
        "equity_index":      "EWC US Equity",
        "bond_index":        "XBB CN Equity",
        "tot_index":         "CTOTCAD Index",
    },
    "CHF": {
        "spot":              "USDCHF {FIX} Curncy",
        "1m_fwd_pts":        "CHF1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "CHFI1M {FIX} Curncy",
        "1m_impl_vol":       "USDCHFV1M {FIX} Curncy",
        "2y_irs":            "SFSNT2 {FIX} Curncy",
        "5y_irs":            "SFSNT5 {FIX} Curncy",
        "10y_irs":           "SFSNT10 {FIX} Curncy",
        "equity_index":      "EWL US Equity",
        "bond_index":        "SPFISWUT Index",
        "tot_index":         "CTOTCHF Index",
    },
    "NOK": {
        "spot":              "USDNOK {FIX} Curncy",
        "1m_fwd_pts":        "NOK1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "NOKI1M {FIX} Curncy",
        "1m_impl_vol":       "USDNOKV1M {FIX} Curncy",
        "2y_irs":            "NKSW2 {FIX} Curncy",
        "5y_irs":            "NKSW5 {FIX} Curncy",
        "10y_irs":           "NKSW10 {FIX} Curncy",
        "equity_index":      "ENOR US Equity",
        "bond_index":        "SPFINOT Index",
        "tot_index":         "CTOTNOK Index",
    },
    "SEK": {
        "spot":              "USDSEK {FIX} Curncy",
        "1m_fwd_pts":        "SEK1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "SEKI1M {FIX} Curncy",
        "1m_impl_vol":       "USDSEKV1M {FIX} Curncy",
        "2y_irs":            "SKSW2 {FIX} Curncy",
        "5y_irs":            "SKSW5 {FIX} Curncy",
        "10y_irs":           "SKSW10 {FIX} Curncy",
        "equity_index":      "EWD US Equity",
        "bond_index":        "SPFISSET Index",
        "tot_index":         "CTOTSEK Index",
    },
    "MXN": {
        "spot":              "USDMXN {FIX} Curncy",
        "1m_fwd_pts":        "MXN1M {FIX} Curncy",
        "1m_ndf_outright":   "MXN+1M {FIX} Curncy",
        "1m_impl_yield":     "MXNI1M {FIX} Curncy",
        "1m_impl_vol":       "USDMXNV1M {FIX} Curncy",
        "2y_irs":            "MPSW2B {FIX} Curncy",
        "5y_irs":            "MPSW5E {FIX} Curncy",
        "10y_irs":           "MPSW10J {FIX} Curncy",
        "equity_index":      "EWW US Equity",
        "bond_index":        "SPVSOVGT Index",
        "tot_index":         "CTOTMXN Index",
    },
    "ZAR": {
        "spot":              "USDZAR {FIX} Curncy",
        "1m_fwd_pts":        "ZAR1M {FIX} Curncy",
        "1m_ndf_outright":   "ZAR+1M {FIX} Curncy",
        "1m_impl_yield":     "ZARI1M {FIX} Curncy",
        "1m_impl_vol":       "USDZARV1M {FIX} Curncy",
        "2y_irs":            "SASW2 {FIX} Curncy",
        "5y_irs":            "SASW5 {FIX} Curncy",
        "10y_irs":           "SASW10 {FIX} Curncy",
        "equity_index":      "EZA US Equity",
        "bond_index":        "SPFIZAT Index",
        "tot_index":         "CTOTZAR Index",
    },
    "BRL": {
        "spot":              "USDBRL {FIX} Curncy",
        "1m_fwd_pts":        "BCN1M {FIX} Curncy",
        "1m_ndf_outright":   "BCN+1M {FIX} Curncy",
        "1m_impl_yield":     "BCNI1M {FIX} Curncy",
        "1m_impl_vol":       "USDBRLV1M {FIX} Curncy",
        "2y_irs":            "BCSFPPDV {FIX} Curncy",
        "5y_irs":            "BCSFSPDV {FIX} Curncy",
        "10y_irs":           "BCSFXPDV {FIX} Curncy",
        "equity_index":      "EWZ US Equity",
        "bond_index":        "SPFIBRAT Index",
        "tot_index":         "CTOTBRL Index",
    },
    "CLP": {
        "spot":              "USDCLP {FIX} Curncy",
        "1m_fwd_pts":        "CHN1M {FIX} Curncy",
        "1m_ndf_outright":   "CHN+1M {FIX} Curncy",
        "1m_impl_yield":     "CHNI1M {FIX} Curncy",
        "1m_impl_vol":       "USDCLPV1M {FIX} Curncy",
        "2y_irs":            None,
        "5y_irs":            None,
        "10y_irs":           None,
        "equity_index":      "ECH US Equity",
        "bond_index":        None,
        "tot_index":         "CTOTCLP Index",
    },
    "COP": {
        "spot":              "USDCOP {FIX} Curncy",
        "1m_fwd_pts":        None,
        "1m_ndf_outright":   "CON+1M {FIX} Curncy",
        "1m_impl_yield":     "CONI1M {FIX} Curncy",
        "1m_impl_vol":       "USDCOPV1M {FIX} Curncy",
        "2y_irs":            None,
        "5y_irs":            None,
        "10y_irs":           None,
        "equity_index":      "GXG US Equity",
        "bond_index":        None,
        "tot_index":         "CTOTCOP Index",
    },
    "HUF": {
        "spot":              "USDHUF {FIX} Curncy",
        "1m_fwd_pts":        "HUF1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "HUFI1M {FIX} Curncy",
        "1m_impl_vol":       "USDHUFV1M {FIX} Curncy",
        "2y_irs":            "HFSW2 {FIX} Curncy",
        "5y_irs":            "HFSW5 {FIX} Curncy",
        "10y_irs":           "HFSW10 {FIX} Curncy",
        "equity_index":      "M1HU Index",
        "bond_index":        "SPFIHUT Index",
        "tot_index":         "CTOTHUF Index",
    },
    "CZK": {
        "spot":              "USDCZK {FIX} Curncy",
        "1m_fwd_pts":        "CZK1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "CZKI1M {FIX} Curncy",
        "1m_impl_vol":       "USDCZKV1M {FIX} Curncy",
        "2y_irs":            "CKSW2 {FIX} Curncy",
        "5y_irs":            "CKSW5 {FIX} Curncy",
        "10y_irs":           "CKSW10 {FIX} Curncy",
        "equity_index":      "M1CZ Index",
        "bond_index":        "SPFICZT Index",
        "tot_index":         "CTOTCZK Index",
    },
    "PLN": {
        "spot":              "USDPLN {FIX} Curncy",
        "1m_fwd_pts":        "PLN1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "PLNI1M {FIX} Curncy",
        "1m_impl_vol":       "USDPLNV1M {FIX} Curncy",
        "2y_irs":            "PZSW2 {FIX} Curncy",
        "5y_irs":            "PZSW5 {FIX} Curncy",
        "10y_irs":           "PZSW10 {FIX} Curncy",
        "equity_index":      "EPOL US Equity",
        "bond_index":        None,
        "tot_index":         "CTOTPLN Index",
    },
    "ILS": {
        "spot":              "USDILS {FIX} Curncy",
        "1m_fwd_pts":        "ILS1M {FIX} Curncy",
        "1m_ndf_outright":   None,
        "1m_impl_yield":     "ILSI1M {FIX} Curncy",
        "1m_impl_vol":       "USDILSV1M {FIX} Curncy",
        "2y_irs":            "ISSW2 {FIX} Curncy",
        "5y_irs":            "ISSW5 {FIX} Curncy",
        "10y_irs":           "ISSW10 {FIX} Curncy",
        "equity_index":      "EIS US Equity",
        "bond_index":        None,
        "tot_index":         "CTOTILS Index",
    },
    "CNH": {
        "spot":              "USDCNH {FIX} Curncy",
        "1m_fwd_pts":        "CNH1M {FIX} Curncy",
        "1m_ndf_outright":   "CNH+1M {FIX} Curncy",
        "1m_impl_yield":     "CNHI1M {FIX} Curncy",
        "1m_impl_vol":       "USDCNHV1M {FIX} Curncy",
        "2y_irs":            "CCSWNI2 {FIX} Curncy",
        "5y_irs":            "CCSWNI5 {FIX} Curncy",
        "10y_irs":           "CCSWNI10 {FIX} Curncy",
        "equity_index":      "MCHI US Equity",
        "bond_index":        "SPBCNGOT Index",
        "tot_index":         "CTOTCNY Index",
    },
    "KRW": {
        "spot":              "USDKRW {FIX} Curncy",
        "1m_fwd_pts":        "KWN1M {FIX} Curncy",
        "1m_ndf_outright":   "KWN+1M {FIX} Curncy",
        "1m_impl_yield":     "KWNI1M {FIX} Curncy",
        "1m_impl_vol":       "USDKRWV1M {FIX} Curncy",
        "2y_irs":            "KWSWNI2 {FIX} Curncy",
        "5y_irs":            "KWSWNI5 {FIX} Curncy",
        "10y_irs":           "KWSWNI10 {FIX} Curncy",
        "equity_index":      "EWY US Equity",
        "bond_index":        "SPBKRGOT Index",
        "tot_index":         "CTOTKRW Index",
    },
    "INR": {
        "spot":              "USDINR {FIX} Curncy",
        "1m_fwd_pts":        "IRN1M {FIX} Curncy",
        "1m_ndf_outright":   "IRN+1M {FIX} Curncy",
        "1m_impl_yield":     "IRNI1M {FIX} Curncy",
        "1m_impl_vol":       "USDINRV1M {FIX} Curncy",
        "2y_irs":            "IRSWNI2 {FIX} Curncy",
        "5y_irs":            "IRSWNI5 {FIX} Curncy",
        "10y_irs":           "IRSWNI10 {FIX} Curncy",
        "equity_index":      "INDA US Equity",
        "bond_index":        "SPBINGOT Index",
        "tot_index":         "CTOTINR Index",
    },
    "IDR": {
        "spot":              "USDIDR {FIX} Curncy",
        "1m_fwd_pts":        "IHN1M {FIX} Curncy",
        "1m_ndf_outright":   "IHN+1M {FIX} Curncy",
        "1m_impl_yield":     "IHNI1M {FIX} Curncy",
        "1m_impl_vol":       "USDIDRV1M {FIX} Curncy",
        "2y_irs":            None,
        "5y_irs":            None,
        "10y_irs":           None,
        "equity_index":      "EIDO US Equity",
        "bond_index":        "SPBIDGOT Index",
        "tot_index":         "CTOTIDR Index",
    },
    "PHP": {
        "spot":              "USDPHP {FIX} Curncy",
        "1m_fwd_pts":        "PPN1M {FIX} Curncy",
        "1m_ndf_outright":   "PPN+1M {FIX} Curncy",
        "1m_impl_yield":     "PPNI1M {FIX} Curncy",
        "1m_impl_vol":       "USDPHPV1M {FIX} Curncy",
        "2y_irs":            None,
        "5y_irs":            None,
        "10y_irs":           None,
        "equity_index":      "EPHE US Equity",
        "bond_index":        "SPBPHGOT Index",
        "tot_index":         "CTOTPHP Index",
    },
    "TWD": {
        "spot":              "USDTWD {FIX} Curncy",
        "1m_fwd_pts":        "NTN1M {FIX} Curncy",
        "1m_ndf_outright":   "NTN+1M {FIX} Curncy",
        "1m_impl_yield":     "NTNI1M {FIX} Curncy",
        "1m_impl_vol":       "USDTWDV1M {FIX} Curncy",
        "2y_irs":            None,
        "5y_irs":            None,
        "10y_irs":           None,
        "equity_index":      "EWT US Equity",
        "bond_index":        "SPBTWGOT Index",
        "tot_index":         "CTOTTWD Index",
    },
    "THB": {
        "spot":              "USDTHB {FIX} Curncy",
        "1m_fwd_pts":        "THB1M {FIX} Curncy",
        "1m_ndf_outright":   "THB+1M {FIX} Curncy",
        "1m_impl_yield":     "THBI1M {FIX} Curncy",
        "1m_impl_vol":       "USDTHBV1M {FIX} Curncy",
        "2y_irs":            None,
        "5y_irs":            None,
        "10y_irs":           None,
        "equity_index":      "THD US Equity",
        "bond_index":        "SPBTHGOT Index",
        "tot_index":         "CTOTTHB Index",
    },
    "SGD": {
        "spot":              "USDSGD {FIX} Curncy",
        "1m_fwd_pts":        "SGD1M {FIX} Curncy",
        "1m_ndf_outright":   "SGD+1M {FIX} Curncy",
        "1m_impl_yield":     "SGDI1M {FIX} Curncy",
        "1m_impl_vol":       "USDSGDV1M {FIX} Curncy",
        "2y_irs":            None,
        "5y_irs":            None,
        "10y_irs":           None,
        "equity_index":      None,
        "bond_index":        None,
        "tot_index":         "CTOTSGD Index",
    },
}

# ------------------------------------------------------------------
# NDF pairs — use outright, not spot + fwd pts
# ------------------------------------------------------------------
NDF_CCYS = {"BRL", "CLP", "COP", "KRW", "INR", "IDR", "TWD", "PHP", "CNH"}

# ------------------------------------------------------------------
# Pip scaling for forward point construction
# Groups: (divisor to convert raw fwd pts to price units)
# ------------------------------------------------------------------
FWD_PIP_SCALE: dict[str, int] = {
    # divide by 10000 (standard)
    "EUR": 10000, "GBP": 10000, "AUD": 10000, "NZD": 10000,
    "CAD": 10000, "CHF": 10000, "NOK": 10000, "SEK": 10000,
    "MXN": 10000, "ZAR": 10000, "HUF": 10000, "CZK": 10000,
    "PLN": 10000, "ILS": 10000, "CNH": 10000, "SGD": 10000,
    # divide by 100
    "JPY": 100, "THB": 100,
    # raw (CLP uses large integer fwd pts)
    "CLP": 1,
}

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_ticker(ccy: str, asset_type: str, fixing: str = "CMPT") -> Optional[str]:
    """
    Return a Bloomberg ticker string for a given currency and asset type.

    Parameters
    ----------
    ccy        : str   3-letter currency code, e.g. 'EUR'
    asset_type : str   one of: spot, 1m_fwd_pts, 1m_ndf_outright,
                       1m_impl_yield, 1m_impl_vol, 2y_irs, 5y_irs, 10y_irs,
                       equity_index, bond_index, tot_index
    fixing     : str   Bloomberg fixing code, e.g. 'CMPT', 'CMPN', 'BGN', 'L083'

    Returns
    -------
    str or None
    """
    ccy = ccy.upper()
    if ccy not in _TICKER_TEMPLATES:
        raise KeyError(f"Currency '{ccy}' not in ticker dictionary. "
                       f"Supported: {list(_TICKER_TEMPLATES.keys())}")
    tmpl = _TICKER_TEMPLATES[ccy].get(asset_type)
    if tmpl is None:
        return None
    return tmpl.replace("{FIX}", fixing)


def get_ccy_list() -> list[str]:
    """Return all supported currency codes."""
    return list(_TICKER_TEMPLATES.keys())


def BBG_TICKER_DICT(fixing: str = "CMPT") -> pd.DataFrame:
    """
    Return the full ticker reference as a DataFrame.
    Rows = currencies, Columns = asset types.

    Parameters
    ----------
    fixing : str   Bloomberg fixing timestamp, e.g. 'CMPT', 'BGN'
    """
    rows = {}
    for ccy, assets in _TICKER_TEMPLATES.items():
        rows[ccy] = {
            k: (v.replace("{FIX}", fixing) if v else None)
            for k, v in assets.items()
        }
    return pd.DataFrame(rows).transpose()
