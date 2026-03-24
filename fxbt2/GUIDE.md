# fxbt2 — Comprehensive FX Quantitative Backtesting Library

`fxbt2` is a merged, production-grade FX backtesting library combining:
- **`fxbt`** — research-focused backtesting, signal generation, and tear sheets
- **`rcq_trading_library`** — desk-ready execution, Bloomberg data, VaR sizing, yield signals

It covers the full workflow from research idea to live trade execution.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Core Concepts](#2-core-concepts)
3. [What's New vs fxbt](#3-whats-new-vs-fxbt)
4. [Step-by-Step Walkthrough](#4-step-by-step-walkthrough)
5. [Data Module](#5-data-module)
6. [Signals Module](#6-signals-module)
7. [Costs Module](#7-costs-module)
8. [Backtest Module](#8-backtest-module)
9. [Portfolio Module](#9-portfolio-module)
10. [Metrics Module](#10-metrics-module)
11. [Report Module](#11-report-module)
12. [Execution Module](#12-execution-module)
13. [PnL Modes: Returns vs Notional](#13-pnl-modes-returns-vs-notional)
14. [Bloomberg Ticker Reference](#14-bloomberg-ticker-reference)
15. [Common Recipes](#15-common-recipes)

---

## 1. Project Structure

```
fxbt2/
├── data/
│   ├── base.py              — DataLoader ABC, validate(), to_wide()
│   ├── ticker_dict.py       — 27 currencies × 11 Bloomberg asset types
│   ├── forward_builder.py   — outright construction, implied yield, holiday filter
│   ├── csv_loader.py        — CSV backend (one file per pair or combined)
│   ├── bquant_loader.py     — Bloomberg BQL backend (BQuant JupyterLab)
│   └── pdblp_loader.py      — Bloomberg Terminal backend (pdblp)
├── signals/
│   ├── price.py             — momentum, crossover, MACD, breakout, mean_reversion,
│   │                          RSI, vol_regime, session_filter
│   └── macro.py             — carry_fwd, carry_yield, yield_range, yield_trend
├── costs/
│   └── model.py             — FixedSpreadModel, SpreadCostModel
├── backtest/
│   ├── engine.py            — Backtest, BacktestResult, walk_forward()
│   └── positions.py         — fixed_size, vol_target, var_target, inverse_vol_weight,
│                              kelly, equal_weight
├── portfolio/
│   ├── construction.py      — generate_pairs, net_ccy_exposure, net_ccy_exposure_usd
│   └── risk.py              — rolling_var, risk_attribution
├── metrics/
│   └── stats.py             — sharpe, sortino, calmar, drawdown, VaR, CVaR, summary
├── report/
│   └── tearsheet.py         — full tear sheet (returns or notional mode)
└── execution/
    └── trade_table.py       — build_trade_table, IMM dates (Bloomberg optional)
```

---

## 2. Core Concepts

Every backtest is built from four components:

| Component | What it is | Example |
|---|---|---|
| **Data** | Historical prices for one or more FX pairs | EURUSD + USDJPY daily 2018–2024 |
| **Signal** | Rule producing +1 (long), -1 (short), 0 (flat) | Momentum over 20 days |
| **Cost model** | Transaction costs per trade | 0.5 pip spread + 0.5bps slippage |
| **Position sizer** | How large each position is | Scale to 10% ann. vol or $20k VaR |

The `Backtest` engine combines all four and returns a `BacktestResult` containing returns, positions, costs, equity curve, and everything needed for analysis.

---

## 3. What's New vs fxbt

| Feature | fxbt | fxbt2 |
|---|---|---|
| Bloomberg Terminal loader | — | `PdblpLoader` via `pdblp` |
| Bloomberg ticker dictionary | — | 27 currencies × 11 asset types |
| Forward outright builder | — | `ForwardBuilder` (handles NDFs, pip conventions, holiday filter) |
| Yield-based carry signal | — | `carry_yield`, `yield_range`, `yield_trend` |
| Notional PnL mode | — | `pnl_mode='notional'`, `notional_size=$10M` |
| VaR-targeted sizing | — | `var_target()` |
| Inverse vol weighting | — | `inverse_vol_weight()` |
| Net currency exposure | — | `portfolio.net_ccy_exposure()` |
| Pair generator | — | `portfolio.generate_pairs()` |
| Risk attribution | — | `portfolio.risk_attribution()` |
| Trade execution table | — | `execution.build_trade_table()` |
| Rollover via impl yield | — | `CostModel.rollover_cost(impl_yield=...)` |
| Dual-mode tear sheet | — | Automatically adapts labels for % or USD |

---

## 4. Step-by-Step Walkthrough

### Setup

```python
import sys
sys.path.append('/path/to/fx_projects/')   # parent of fxbt2/

import pandas as pd
import numpy as np

from fxbt2 import Backtest, CSVLoader, BQuantLoader, PdblpLoader
from fxbt2 import signals, metrics, report, portfolio, execution
from fxbt2.backtest.positions import vol_target, var_target, inverse_vol_weight
from fxbt2.costs import FixedSpreadModel
from fxbt2.data.base import DataLoader
from fxbt2.data.forward_builder import ForwardBuilder
```

### Load Data

**CSV (local/offline):**
```python
loader = CSVLoader(data_dir='data/')
df = loader.load(['EURUSD', 'USDJPY', 'USDMXN'], start='2018-01-01', end='2024-12-31')
```

**BQuant (Bloomberg JupyterLab):**
```python
loader = BQuantLoader(fixing='CMPT', load_forwards=True, load_yield=True)
df = loader.load(['EURUSD', 'USDJPY', 'USDMXN'], start='2018-01-01', end='2024-12-31')
```

**Bloomberg Terminal (pdblp):**
```python
loader = PdblpLoader(fixing='CMPT', load_forwards=True, load_yield=True)
df = loader.load(['EURUSD', 'USDJPY', 'USDMXN'], start='20180101', end='20241231')

# Intraday:
intraday = loader.load_intraday(['EURUSD'], '2024-01-01T00:00:00', '2024-12-31T23:59:59', interval_min=60)
```

### Pivot to wide format

```python
prices     = DataLoader.to_wide(df, 'close')
fwd_pts    = DataLoader.to_wide(df, 'fwd_points')   # if loaded
impl_yield = DataLoader.to_wide(df, 'impl_yield')   # if loaded
```

### Build forward outrights and implied yield (optional)

```python
# If you want more accurate carry from raw spot + fwd points data
builder = ForwardBuilder()
ccy_codes = ['USD', 'EUR', 'JPY', 'MXN']
# outright per ccy vs USD
outrights = builder.build_pair_outright(spot_by_ccy, fwd_by_ccy, ccy_codes, pairs=['EURUSD','USDJPY'])
# implied yield differential per pair
yield_diff = builder.build_implied_yield(spot_by_ccy, fwd_by_ccy, ccy_codes, pairs=['EURUSD','USDJPY'])
```

### Generate signals

```python
# Price signals
mom_sig    = signals.momentum(prices, lookback=20)
cross_sig  = signals.crossover(prices, fast=10, slow=50)
mr_sig     = signals.mean_reversion(prices, lookback=20, z_entry=1.5)

# Macro/yield signals (require fwd_pts or impl_yield)
carry_sig  = signals.carry_fwd(fwd_pts, prices, signal_type='sign')
yield_sig  = signals.carry_yield(impl_yield, signal_type='zscore')
yr_sig     = signals.yield_range(impl_yield, lookback=252)
yt_sig     = signals.yield_trend(impl_yield, lookback=252)

# Filters (multiply against directional signals)
vol_filt   = signals.vol_regime(prices, lookback=20, high_vol_percentile=75)
filtered   = mom_sig * vol_filt

# Blend two signals
blend = (mom_sig + carry_sig) / 2
```

### Run the backtest

**Returns mode (default):**
```python
bt = Backtest(
    data=prices,
    signals=mom_sig,
    cost_model=FixedSpreadModel(slippage_bps=0.5),
    sizer=lambda s, p: vol_target(s, p, target_vol=0.10, freq='1D'),
    freq='1D',
    name='20D Momentum',
)
result = bt.run()
```

**Notional mode ($10M per signal):**
```python
bt = Backtest(
    data=prices,
    signals=mom_sig,
    cost_model=FixedSpreadModel(slippage_bps=0.5),
    sizer=lambda s, p: inverse_vol_weight(s, p),
    freq='1D',
    name='Momentum — $10M Notional',
    pnl_mode='notional',
    notional_size=10_000_000,
)
result = bt.run()
```

### Analyse results

```python
result.summary()          # performance table
result.tearsheet()        # 6-panel chart (auto-adapts for returns vs notional)
result.compare_to(other)  # side-by-side vs another strategy
result.rolling_sharpe(window=63)
result.monthly_returns()
```

---

## 5. Data Module

### DataLoader (abstract base)

```python
DataLoader.validate(df)            # enforce schema, coerce types, add synthetic bid/ask
DataLoader.to_wide(df, 'close')    # pivot long → wide
```

**Normalised schema:**

| Column | Required | Description |
|---|---|---|
| `close` | Yes | Mid price |
| `open`, `high`, `low` | No | OHLC |
| `bid`, `ask` | No | For SpreadCostModel |
| `fwd_points` | No | 1M forward points |
| `fwd_outright` | No | 1M forward outright price |
| `impl_vol` | No | 1M ATM implied vol |
| `impl_yield` | No | 1M implied yield differential |

### CSVLoader

```python
CSVLoader(
    data_dir='data/',         # directory (one CSV per pair) or path to combined file
    timestamp_col='Date',     # name of date column
    combined=False,           # True = single multi-pair file
    pair_col='pair',          # column name for pair (combined mode)
)
```

### BQuantLoader

```python
BQuantLoader(
    fixing='CMPT',            # Bloomberg fixing: 'CMPT', 'BGN', 'CMPN', 'L083'
    load_forwards=True,
    load_vol=True,
    load_yield=True,
)
```

### PdblpLoader

```python
PdblpLoader(
    fixing='CMPT',
    port=8194,
    timeout=50000,
    load_forwards=True,
    load_vol=True,
    load_yield=True,
)

# Also supports intraday:
loader.load_intraday(pairs, start_datetime, end_datetime, interval_min=60)
```

### ForwardBuilder

```python
builder = ForwardBuilder(ann_factor=12, holiday_window=5)

# Build 1M outright prices (handles NDF pairs, pip conventions)
outrights = builder.build_outright(spot_by_ccy, fwd_by_ccy, ccy_codes)
pair_outrights = builder.build_pair_outright(spot, fwd, ccy_codes, pairs)

# Build implied yield differential with holiday filter
yield_diff = builder.build_implied_yield(spot, fwd, ccy_codes, pairs)
```

**NDF pairs** (use outright directly, not spot + fwd pts):
`BRL, CLP, COP, KRW, INR, IDR, TWD, PHP, CNH`

---

## 6. Signals Module

All signals: **input** = wide prices DataFrame → **output** = same-shape signal DataFrame.
All are forward-safe (`.shift(1)` internally applied — signal on day T uses only data up to day T-1).

---

### `momentum(prices, lookback, signal_type)`

**Logic:** Measures the N-period return of each pair.

```
return_t = (price_t / price_{t-N}) - 1
```

- `signal_type='sign'` → `+1` if return > 0, `-1` if return < 0
- `signal_type='zscore'` → normalises the return: `(return - rolling_mean) / rolling_std` over a `lookback × 3` window. Returns a continuous value centred at 0.
- `signal_type='rank'` → cross-sectional percentile rank across all pairs on each day, centred at 0 (so 0.5 rank → 0.0 signal). Useful for relative-value baskets.

**Intuition:** If a currency has risen over the past N days, assume it continues rising (trend following).

```python
momentum(prices, lookback=20, signal_type='sign')
```

---

### `crossover(prices, fast, slow)`

**Logic:** Compares a short-term moving average to a long-term moving average.

```
fast_MA_t = mean(price_{t-F+1} ... price_t)
slow_MA_t = mean(price_{t-S+1} ... price_t)
signal    = +1 if fast_MA > slow_MA, else -1
```

- `+1` (long) when the fast MA is above the slow MA — short-term trend is above long-term trend
- `-1` (short) when fast MA crosses below slow MA

**Intuition:** A rising short-term average crossing above the long-term average signals upward momentum. A crossover below signals a downtrend.

```python
crossover(prices, fast=10, slow=50)
```

---

### `macd_signal(prices, fast, slow, signal_period)`

**Logic:** MACD uses exponential moving averages (EMA) which weight recent prices more heavily than a simple MA.

```
MACD_line   = EMA(price, fast) - EMA(price, slow)
signal_line = EMA(MACD_line, signal_period)
histogram   = MACD_line - signal_line

signal = +1 if histogram > 0, else -1
```

- `+1` when MACD line has crossed above its own signal line (momentum accelerating up)
- `-1` when MACD line has crossed below its signal line (momentum turning down)

**Intuition:** MACD captures the rate of change of trend. The histogram crossing zero means the trend is gaining or losing momentum — it tends to lead price turns earlier than a simple crossover.

```python
macd_signal(prices, fast=12, slow=26, signal_period=9)
```

---

### `breakout(prices, lookback)`

**Logic:** Uses Donchian channels — the rolling N-day high and low computed from **yesterday's** data (no look-ahead).

```
upper_band_t = max(price_{t-N} ... price_{t-1})
lower_band_t = min(price_{t-N} ... price_{t-1})

signal = +1 if price_t > upper_band  (breaks above recent high)
signal = -1 if price_t < lower_band  (breaks below recent low)
signal =  0 if price_t inside band   (no breakout)
```

**Intuition:** A price breaking above the recent N-day range signals that buyers have overcome resistance — the move has enough force to extend. Classic trend-following entry used in systematic CTA strategies.

```python
breakout(prices, lookback=20)
```

---

### `mean_reversion(prices, lookback, z_entry, z_exit)`

**Logic:** Computes the z-score of the current price relative to its recent history.

```
rolling_mean_t = mean(price_{t-N} ... price_t)
rolling_std_t  = std(price_{t-N}  ... price_t)
z_t            = (price_t - rolling_mean_t) / rolling_std_t

signal = +1  if z < -z_entry   (price far below mean → expect reversion up)
signal = -1  if z > +z_entry   (price far above mean → expect reversion down)
signal =  0  if |z| < z_exit   (price near mean → flat, avoid whipsaw)
```

**Intuition:** FX rates tend to mean-revert around fundamental value over short horizons. If a price has moved 1.5 standard deviations away from its recent average, it is statistically extended and likely to revert. The `z_exit` threshold prevents constantly flipping around the mean.

```python
mean_reversion(prices, lookback=20, z_entry=1.5, z_exit=0.5)
```

---

### `rsi_signal(prices, period, overbought, oversold)`

**Logic:** RSI (Relative Strength Index) measures the speed and magnitude of recent price changes on a 0–100 scale.

```
avg_gain_t = mean of positive daily changes over last N periods
avg_loss_t = mean of absolute negative daily changes over last N periods
RS_t       = avg_gain_t / avg_loss_t
RSI_t      = 100 - (100 / (1 + RS_t))

signal = +1  if RSI < oversold    (e.g. < 30 — pair sold off too hard)
signal = -1  if RSI > overbought  (e.g. > 70 — pair rallied too hard)
signal =  0  otherwise
```

**Intuition:** RSI above 70 means the pair has gained strongly in recent periods relative to losses — the move may be exhausted. RSI below 30 means the opposite. This is a mean-reversion signal based on momentum exhaustion rather than absolute price levels.

```python
rsi_signal(prices, period=14, overbought=70, oversold=30)
```

---

### `vol_regime(prices, lookback, high_vol_percentile)`

**Logic:** Measures current realised volatility and compares it to its own history.

```
rv_t     = std(daily_returns_{t-N} ... daily_returns_t)
threshold = Nth percentile of rv over entire history

filter = 1  if rv_t <= threshold   (normal vol — allow trading)
filter = 0  if rv_t >  threshold   (elevated vol — suppress signal)
```

**This is a filter, not a directional signal.** Multiply it against another signal:

```python
filtered_signal = momentum(prices) * vol_regime(prices, high_vol_percentile=75)
```

**Intuition:** Many signals break down in high-volatility regimes — noise overwhelms the edge. By suppressing trading when vol is in the top 25% of its history, you avoid getting chopped up during market stress.

---

### `session_filter(prices, session)`

**Logic:** Returns 1 during the active trading session, 0 outside it. Intraday data only.

```
london    → UTC 08:00–17:00
new_york  → UTC 13:00–22:00
tokyo     → UTC 23:00–08:00
overlap   → UTC 13:00–17:00  (London/NY overlap — highest liquidity)
```

**Use the same way as `vol_regime`** — multiply against an intraday signal to only trade during liquid hours.

---

### `carry_fwd(fwd_points, spot_prices, signal_type)`

**Logic:** Derives the annualised carry from 1-month forward points.

```
carry_t = -(fwd_points_t / spot_t) × 12
```

The forward points represent the premium or discount of the forward rate over spot. Dividing by spot converts to a rate differential fraction; multiplying by 12 annualises it from a 1M tenor.

- Negative forward points → base currency at a forward discount → holding the base earns carry
- Positive forward points → base currency at a forward premium → holding the base costs carry

```
signal = +1  if carry > 0  (earn positive carry being long base)
signal = -1  if carry < 0  (earn positive carry being short base)
```

`signal_type='zscore'` normalises carry against its own 60-day rolling mean and std — useful for comparing carry magnitude across pairs with different yield levels.

```python
carry_fwd(fwd_points, prices, signal_type='sign')
```

---

### `carry_yield(impl_yield, signal_type)`

**Logic:** Uses the annualised implied yield differential directly.

```
impl_yield_t = base_ccy_yield_t - quote_ccy_yield_t   (computed by ForwardBuilder)

signal = +1  if impl_yield > 0  (base yields more than quote)
signal = -1  if impl_yield < 0  (quote yields more than base)
```

This is more accurate than `carry_fwd` for EM and NDF pairs where forward points embed both carry and credit/liquidity risk components. The implied yield cleanly separates the pure rate differential.

```python
carry_yield(impl_yield, signal_type='sign')
```

---

### `yield_range(impl_yield, lookback)`

**Logic:** Enter carry only when the yield differential is at a **historically extreme** level relative to its own recent range.

```
rank_t = (impl_yield_t - rolling_min) / (rolling_max - rolling_min)
         over a lookback-day window   (0 = historical low, 1 = historical high)

signal = +1  if rank > 0.51  AND  impl_yield > 0    (carry high vs history AND positive)
signal = -1  if rank < 0.49  AND  impl_yield < 0    (carry low vs history AND negative)
signal =  0  otherwise                               (in the middle of the range — stay flat)
```

**Intuition:** A carry trade only makes sense when the yield differential is attractive. If a pair normally yields 5% but currently yields 8%, that's a historically wide differential worth fading back toward. The rank filter avoids entering when carry is thin or ambiguous.

Particularly useful for EM carry where yield differentials can compress sharply during risk-off episodes.

```python
yield_range(impl_yield, lookback=252)
```

---

### `yield_trend(impl_yield, lookback, ema_span)`

**Logic:** Follows the *trend* of the yield differential itself rather than its level.

```
smoothed_t  = EMA(impl_yield, span=ema_span)           (reduces noise in yield data)
z_t         = (smoothed_t - rolling_mean) / rolling_std  (over lookback-day window)
signal_t    = clip(z_t, -1, +1)                        (continuous, bounded [-1, +1])
```

Unlike `carry_yield` which is always-on, `yield_trend` only positions when the carry differential is **trending in a direction** relative to its own history. A z-score of +1 means the differential is as high as it has been in a year; -1 means as low.

**Intuition:** This captures rate divergence trades — e.g. one central bank hiking while another is cutting. The EMA smoothing prevents noise from daily fixing data triggering false signals.

```python
yield_trend(impl_yield, lookback=252, ema_span=5)
```

---

## 7. Costs Module

Every backtest deducts two types of cost on every bar:

1. **Entry/exit cost** — paid when a position changes (spread + slippage)
2. **Rollover cost** — paid daily on any open position (overnight carry/financing)

---

### Entry / Exit Cost

**`FixedSpreadModel`** — uses a pip spread table built into the library.

```
pip_size      = 0.0001  for most pairs,  0.01  for JPY pairs
half_spread   = (spread_pips × pip_size) / (2 × price)
entry_cost    = (half_spread + slippage_fraction) × |trade_size|
```

For example, EURUSD at 1.10 with a 0.5 pip spread:
```
half_spread = (0.5 × 0.0001) / (2 × 1.10) = 0.0000227  per unit
```

**`SpreadCostModel`** — uses the actual bid/ask you supply.

```
mid           = (bid + ask) / 2
half_spread   = (ask - bid) / (2 × mid)
entry_cost    = (half_spread + slippage_fraction) × |trade_size|
```

Both cost models charge this on every trade — i.e. every time `positions.diff() != 0`.

---

### Rollover Cost

Paid every day on any **open** position. Two methods, in order of preference:

**Method 1 — implied yield** (preferred, more accurate):
```
rollover_cost_t = (impl_yield_t / 260) × position_size
```
The implied yield is already annualised, so dividing by 260 gives the daily rate.

**Method 2 — forward points** (fallback):
```
rollover_cost_t = (fwd_points_t / price_t / 365) × position_size
```
Converts forward points to an annualised fraction of spot, then to a daily rate.

If neither is supplied, rollover cost is zero.

---

### Built-in Spread Defaults

Selected pairs from the built-in table:

| Pair | Spread (pips) | Pair | Spread (pips) |
|---|---|---|---|
| EURUSD | 0.5 | USDMXN | 8.0 |
| USDJPY | 0.5 | USDBRL | 12.0 |
| GBPUSD | 0.8 | USDZAR | 8.0 |
| AUDUSD | 0.8 | USDKRW | 5.0 |
| USDCAD | 0.8 | USDINR | 5.0 |
| EURGBP | 1.0 | USDCLP | 15.0 |
| GBPJPY | 1.5 | USDCOP | 15.0 |

---

```python
FixedSpreadModel(
    spreads_pips={'EURUSD': 0.4, 'USDMXN': 6.0},  # override defaults per pair
    default_spread_pips=3.0,                         # fallback for unlisted pairs
    slippage_bps=0.5,
)

SpreadCostModel(slippage_bps=0.5)
# Pass bids= and asks= DataFrames to Backtest() when using this
```

---

## 8. Backtest Module

### Backtest

```python
Backtest(
    data=prices,              # wide or long-format DataFrame
    signals=sig,              # signal DataFrame (same shape as data)
    cost_model=...,           # CostModel instance
    sizer=...,                # callable: sizer(signals, prices) → positions
    freq='1D',                # return frequency
    name='Strategy Name',
    pnl_mode='returns',       # 'returns' or 'notional'
    notional_size=10_000_000, # USD notional (notional mode only)
    bids=...,                 # optional bid prices
    asks=...,                 # optional ask prices
    fwd_points=...,           # optional, for rollover
    impl_yield=...,           # optional, for rollover (preferred over fwd_points)
)
result = bt.run()
```

### Walk-forward

```python
def signal_fn(train_prices, test_prices):
    combined = pd.concat([train_prices, test_prices])
    return signals.momentum(combined, lookback=20).loc[test_prices.index]

wf = bt.walk_forward(
    train_periods=252,   # 1 year in-sample
    test_periods=63,     # 1 quarter OOS
    signal_fn=signal_fn,
)
```

### Position sizers

Each sizer takes `(signals, prices)` and returns a positions DataFrame of the same shape. The signal direction (+1/-1) is preserved; the sizer controls the magnitude.

---

#### `fixed_size(signals, size)`

```
position_t = sign(signal_t) × size
```

Every active signal gets the same fixed weight. Simple but ignores differences in pair volatility — a USDJPY position and a USDMXN position will have very different risk despite the same nominal size.

```python
sizer = lambda s, p: fixed_size(s, size=0.2)
```

---

#### `equal_weight(signals)`

```
n_active_t  = number of pairs with non-zero signal on day t
position_t  = signal_t / n_active_t
```

Allocates 1.0 total weight equally across all active positions each day. If 3 pairs are active, each gets 0.333. If 5 are active, each gets 0.2. Automatically adjusts when pairs turn flat.

```python
sizer = lambda s, p: equal_weight(s)
```

---

#### `vol_target(signals, prices, target_vol, lookback, freq)`

**Goal:** Scale each pair so the portfolio hits a target annualised volatility.

```
rv_i_t       = rolling_std(daily_returns_i, lookback) × sqrt(ann_factor)
per_pair_vol = target_vol / n_pairs
size_i_t     = per_pair_vol / rv_i_t
position_i_t = sign(signal_i_t) × min(size_i_t, max_leverage / n_pairs)
```

A pair with 10% annualised vol gets a bigger weight than one with 20% vol, because we need a larger notional to generate the same risk contribution. The `max_leverage` cap prevents extreme weights on very low-vol pairs.

**Example:** target_vol=10%, 5 pairs → each pair targets 2% vol. If EURUSD realised vol is 8%, its weight = 2% / 8% = 0.25.

```python
sizer = lambda s, p: vol_target(s, p, target_vol=0.10, lookback=20, freq='1D')
```

---

#### `inverse_vol_weight(signals, prices, lookback, rebalance_day)`

**Goal:** Weight pairs by the inverse of their realised volatility, normalised to sum to 1.

```
ann_vol_i_t  = rolling_std(returns_i, lookback) × sqrt(260)
inv_vol_i_t  = 1 / ann_vol_i_t
weight_i_t   = inv_vol_i_t / sum(inv_vol_j_t for all j)
position_i_t = sign(signal_i_t) × weight_i_t
```

Weights are only updated on `rebalance_day` (default: Monday), then held constant for the week. This reduces turnover compared to daily rebalancing.

The result is that low-vol pairs (e.g. EURUSD) get relatively more weight than high-vol pairs (e.g. USDMXN), which roughly equalises the risk contribution per pair.

```python
sizer = lambda s, p: inverse_vol_weight(s, p, lookback=22, rebalance_day=0)
```

---

#### `kelly(signals, prices, lookback, fraction, max_leverage)`

**Goal:** Size each position in proportion to its expected edge divided by variance.

```
μ_i_t     = rolling_mean(returns_i, lookback)      (expected return)
σ²_i_t    = rolling_var(returns_i,  lookback)      (variance)
kelly_i_t = μ_i_t / σ²_i_t × fraction              (fractional Kelly)
position_i_t = sign(signal_i_t) × clip(|kelly_i_t|, 0, max_leverage)
```

Full Kelly is theoretically optimal but highly sensitive to estimation error — a bad mean estimate produces extreme sizing. `fraction=0.5` (half-Kelly) is standard; it halves the position size in exchange for much better robustness.

```python
sizer = lambda s, p: kelly(s, p, lookback=60, fraction=0.5, max_leverage=2.0)
```

---

#### `var_target(signals, pnl_series, var_window, var_target_usd, confidence)`

**Goal:** Scale the entire portfolio daily so its rolling VaR equals a fixed dollar target.

```
daily_pnl_t    = pnl_series.diff()
roll_mean      = rolling_mean(daily_pnl, var_window)
roll_std       = rolling_std(daily_pnl,  var_window)
roll_var_t     = norm.ppf(confidence, roll_mean, roll_std)   (parametric, normal dist)
adj_factor_t   = var_target_usd / |roll_var_t|
position_t     = signal_t × adj_factor_{t-1}                (lagged by 1 day)
```

In a high-vol period, `roll_var_t` is large so `adj_factor` shrinks — the book gets smaller. In a quiet period, `adj_factor` grows — the book scales up. This keeps daily dollar risk roughly constant regardless of market conditions.

`confidence=0.05` means the 95th percentile loss — VaR at the 5% tail.

```python
sig_adj = var_target(signals, pnl_series=r_base.returns.cumsum(),
                     var_window=260, var_target_usd=20_000, confidence=0.05)
```

---

**Combining sizers** — `inverse_vol_weight` and `var_target` are designed to be chained:
first weight pairs by inverse vol (cross-sectional), then scale the whole portfolio to hit the VaR target (time-series):

```python
# Step 1: cross-sectional weighting
inv_vol_sig = inverse_vol_weight(signals, prices)

# Step 2: scale whole book to $20k VaR
sig_final = var_target(inv_vol_sig, pnl_series, var_target_usd=20_000)
```

---

## 9. Portfolio Module

### Pair generation

```python
from fxbt2.portfolio import generate_pairs

# Generate all unique pairs from a basket of currencies
generate_pairs(['USD', 'EUR', 'MXN', 'ZAR'])
# → ['USDEUR', 'USDMXN', 'USDZAR', 'EURMXN', 'EURZAR', 'MXNZAR']
```

### Net currency exposure

```python
from fxbt2.portfolio import net_ccy_exposure, net_ccy_exposure_usd

# Decompose pair positions into single-currency net exposures
# e.g. long EURUSD + long EURGBP → net EUR = +2, net USD = -1, net GBP = -1
net = net_ccy_exposure(result.positions)

# Convert to USD-cross format for execution
usd_net = net_ccy_exposure_usd(result.positions)
# → columns: USDEUR, USDJPY, USDMXN, etc.
```

### Risk attribution

```python
from fxbt2.portfolio import risk_attribution

ra = risk_attribution(result.pair_returns, freq='1D')
```

**How it is calculated:**

```
ann_vol_i        = std(pair_returns_i) × sqrt(ann_factor)

portfolio        = sum of all pair_returns per day
cov(i, portfolio)= covariance of pair i returns with portfolio returns × ann_factor
port_variance    = variance(portfolio) × ann_factor

pct_vol_contrib_i = cov(i, portfolio) / port_variance
```

`pct_vol_contrib` answers: "what fraction of total portfolio variance does this pair explain?" — pairs that are highly correlated with the portfolio AND have high vol will dominate this number.

Returns a DataFrame with columns: `ann_vol`, `pct_vol_contrib`, `sharpe`, `total_return`.

---

### Rolling VaR

```python
from fxbt2.portfolio import rolling_var

var_series = rolling_var(result.returns, window=260, confidence=0.05)
```

**How it is calculated** (parametric, normal distribution assumption):

```
daily_pnl   = pnl_series.diff()
roll_mean_t = rolling_mean(daily_pnl, window)
roll_std_t  = rolling_std(daily_pnl,  window)
VaR_t       = norm.ppf(confidence, roll_mean_t, roll_std_t)
```

`norm.ppf(0.05, μ, σ)` gives the value below which 5% of daily PnL observations fall — i.e. the loss you expect to exceed only 1 in 20 days. The result is negative (it represents a loss).

`confidence=0.05` → 95% VaR. `confidence=0.01` → 99% VaR.

---

## 10. Metrics Module

All functions accept a `pd.Series` of period returns. `freq` controls the annualisation factor.

Supported `freq` values: `'1D'` (×252), `'1W'` (×52), `'1M'` (×12), `'1H'` (×252×24), `'15min'`, `'5min'`, `'1min'`

---

#### `annualised_return(r, freq)`
```
ann_return = mean(r) × ann_factor
```
Average period return scaled up to one year. Simple, not compounded.

---

#### `annualised_vol(r, freq)`
```
ann_vol = std(r) × sqrt(ann_factor)
```
Standard deviation of returns scaled to annual. The `sqrt` comes from the square-root-of-time rule for independent returns.

---

#### `sharpe(r, freq, risk_free)`
```
excess_r = r - risk_free / ann_factor
sharpe   = mean(excess_r) / std(excess_r) × sqrt(ann_factor)
```
Return per unit of total risk. A Sharpe above 1.0 is considered good for an FX strategy; above 1.5 is strong. The `risk_free` rate is annualised — divide by `ann_factor` to get the per-period equivalent before subtracting.

---

#### `sortino(r, freq, risk_free)`
```
excess_r    = r - risk_free / ann_factor
down_r      = excess_r[excess_r < 0]
sortino     = mean(excess_r) × ann_factor / (std(down_r) × sqrt(ann_factor))
```
Like Sharpe but only penalises **downside** volatility. A strategy that has smooth gains but occasional large drawdowns scores worse on Sharpe than Sortino. Sortino is generally a more relevant metric for systematic strategies.

---

#### `calmar(r, freq)`
```
calmar = annualised_return(r) / abs(max_drawdown(r))
```
Return per unit of maximum historical loss. Measures how efficiently the strategy earns return relative to its worst historical drawdown. Higher is better.

---

#### `max_drawdown(r)`
```
equity_t  = cumprod(1 + r)
drawdown_t = (equity_t - cummax(equity_t)) / cummax(equity_t)
max_dd    = min(drawdown_t)
```
The largest peak-to-trough decline in the equity curve as a fraction. A value of -0.15 means the strategy lost 15% from its highest point before recovering.

---

#### `drawdown_series(r)`
Returns the full time series of drawdown values using the same formula above. Useful for plotting or identifying when the worst periods occurred.

---

#### `hit_rate(r)`
```
hit_rate = count(r > 0) / count(r != 0)
```
Fraction of non-zero return periods that were positive. A hit rate of 50% means the strategy wins half the time. Note: hit rate alone is not sufficient — a strategy with 40% hit rate can still be profitable if winners are much larger than losers (see `avg_win_loss_ratio`).

---

#### `profit_factor(r)`
```
profit_factor = sum(r[r > 0]) / abs(sum(r[r < 0]))
```
Total gross profit divided by total gross loss. A value above 1.0 means the strategy makes more than it loses in aggregate. Values of 1.2–1.5 are typical for systematic FX strategies.

---

#### `avg_win_loss_ratio(r)`
```
avg_win_loss = mean(r[r > 0]) / abs(mean(r[r < 0]))
```
Average winning period return divided by average losing period return. Combines with hit rate to give the full picture: a 40% hit rate with a 2.0 win/loss ratio is profitable (0.4 × 2.0 > 0.6 × 1.0).

---

#### `var(r, level)`
```
VaR = quantile(r, level)
```
The return level below which `level` fraction of observations fall. `level=0.05` gives the 5th percentile of the return distribution — the loss exceeded on 1 in 20 periods. Expressed as a fraction (e.g. -0.012 = -1.2%).

---

#### `cvar(r, level)`
```
CVaR = mean(r[r <= VaR])
```
Conditional Value at Risk — the **average** return in the worst `level` fraction of periods. Also called Expected Shortfall. More informative than VaR because it tells you how bad the bad days actually are, not just the threshold.

---

```python
from fxbt2 import metrics

metrics.sharpe(r, freq='1D', risk_free=0.05)
metrics.sortino(r, freq='1D')
metrics.calmar(r, freq='1D')
metrics.max_drawdown(r)
metrics.drawdown_series(r)
metrics.hit_rate(r)
metrics.profit_factor(r)
metrics.avg_win_loss_ratio(r)
metrics.var(r, level=0.05)
metrics.cvar(r, level=0.05)
metrics.annualised_return(r)
metrics.annualised_vol(r)
metrics.summary(r, freq='1D', name='My Strategy', pnl_mode='returns')
metrics.compare((r1, 'Strat A'), (r2, 'Strat B'), freq='1D')
```

---

## 11. Report Module

```python
from fxbt2 import report

result.tearsheet()                          # full 6-panel tear sheet
report.plot_equity_curve(result)
report.plot_drawdown(result)
report.plot_rolling_sharpe(result, window=52)
report.plot_monthly_heatmap(result)
```

The tear sheet automatically adapts for `pnl_mode`:
- **Returns mode**: equity in `x` multiples, drawdown in `%`, heatmap in `%`
- **Notional mode**: equity in `$USD`, drawdown in `$USD`, heatmap in `$USD`

---

## 12. Execution Module

The execution module bridges research and live trading. It requires a Bloomberg Terminal
connection for live rate fetch and IMM date resolution, but works without it (pass `ref_rates`
manually or receive `'T+2'` as the settlement date).

```python
from fxbt2.execution import build_trade_table

# After running a notional-mode backtest:
trade_table = build_trade_table(
    tradesize=result.positions,          # today's and yesterday's positions
    ref_rates=None,                      # auto-fetch if pdblp_connection provided
    ndf_pairs=['USDBRL', 'USDKRW'],     # these pairs use NDF outright as ref rate
    pdblp_connection=con,                # optional: live Bloomberg connection
    fixing='CMPT',
)
```

Output columns:
| Column | Description |
|---|---|
| `target_rate` | Today's reference rate (spot or NDF outright) |
| `{today}_pos` | Target position in USD |
| `{yesterday}_pos` | Previous position in USD |
| `to_trade_usd` | Delta to execute today (+ = buy, - = sell) |
| `val_date` | IMM settlement date |

---

## 13. PnL Modes: Returns vs Notional

**Returns mode** (default) — portable, data-source agnostic, works without knowing AUM.
Everything is a fraction of portfolio.

```python
bt = Backtest(data=prices, signals=sig, pnl_mode='returns')
# result.returns     → daily % return, e.g. 0.0042 = +0.42%
# result.equity_curve → starts at 1.0, grows with compounding
```

**Notional mode** — desk-ready, PnL in real USD. Matches how a trading desk runs a book.

```python
bt = Backtest(data=prices, signals=sig,
              pnl_mode='notional', notional_size=10_000_000)
# result.returns     → daily USD PnL, e.g. 42000 = +$42k
# result.equity_curve → cumulative USD PnL (starts at 0)
```

**VaR sizing in notional mode** — the typical production workflow:

```python
# Step 1: run with equal_weight to get a PnL series
pilot = Backtest(data=prices, signals=sig,
                 sizer=lambda s,p: equal_weight(s),
                 pnl_mode='notional', notional_size=1_000_000).run()

# Step 2: rerun with VaR targeting scaled to $20k/day
from fxbt2.backtest.positions import var_target
bt = Backtest(
    data=prices, signals=sig,
    sizer=lambda s,p: var_target(equal_weight(s), pilot.returns,
                                 var_window=260, var_target_usd=20_000),
    pnl_mode='notional', notional_size=1_000_000,
).run()
```

---

## 14. Bloomberg Ticker Reference

The `ticker_dict` maps 27 currencies to Bloomberg tickers for 11 asset types.

```python
from fxbt2.data.ticker_dict import BBG_TICKER_DICT, get_ticker, get_ccy_list

# All supported currencies
get_ccy_list()
# → ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF', 'NOK', 'SEK',
#    'MXN', 'ZAR', 'BRL', 'CLP', 'COP', 'HUF', 'CZK', 'PLN', 'ILS',
#    'CNH', 'KRW', 'INR', 'IDR', 'PHP', 'TWD', 'THB', 'SGD']

# Get a specific ticker
get_ticker('EUR', 'spot',         fixing='CMPT')   # → 'USDEUR CMPT Curncy'
get_ticker('MXN', '1m_impl_vol',  fixing='BGN')    # → 'USDMXNV1M BGN Curncy'
get_ticker('BRL', '1m_ndf_outright', fixing='CMPT') # → 'BCN+1M CMPT Curncy'
get_ticker('JPY', '10y_irs',      fixing='CMPT')   # → 'JYSO10 CMPT Curncy'

# Full reference table
BBG_TICKER_DICT(fixing='CMPT')  # → pd.DataFrame, rows=ccy, cols=asset_type
```

**Asset types available per currency:**

| Asset type | Description |
|---|---|
| `spot` | Spot mid price |
| `1m_fwd_pts` | 1M forward points |
| `1m_ndf_outright` | 1M NDF outright (EM/restricted pairs) |
| `1m_impl_yield` | 1M implied yield |
| `1m_impl_vol` | 1M ATM implied volatility |
| `2y_irs` | 2-year interest rate swap |
| `5y_irs` | 5-year interest rate swap |
| `10y_irs` | 10-year interest rate swap |
| `equity_index` | Local iShares equity ETF |
| `bond_index` | Local government bond index |
| `tot_index` | Total return index |

---

## 15. Common Recipes

### EM basket: carry + momentum + inverse vol weight

```python
pairs = portfolio.generate_pairs(['USD', 'MXN', 'ZAR', 'BRL', 'CLP'])
df = loader.load(pairs, start='2018-01-01', end='2024-12-31')
prices = DataLoader.to_wide(df, 'close')
fwd    = DataLoader.to_wide(df, 'fwd_points')

carry_sig = signals.carry_fwd(fwd, prices, signal_type='zscore')
mom_sig   = signals.momentum(prices, lookback=20, signal_type='zscore')
blend     = (carry_sig + mom_sig) / 2

result = Backtest(
    data=prices, signals=blend,
    cost_model=FixedSpreadModel(slippage_bps=1.0),
    sizer=lambda s, p: inverse_vol_weight(s, p),
    pnl_mode='notional', notional_size=5_000_000,
    freq='1D', name='EM Carry+Momentum',
).run()
result.tearsheet()
```

### Yield trend strategy (rates-driven)

```python
# Requires impl_yield data from BQuantLoader or PdblpLoader
yt_sig = signals.yield_trend(impl_yield, lookback=252, ema_span=5)
vol_filter = signals.vol_regime(prices, lookback=20)
filtered = yt_sig * vol_filter

result = Backtest(data=prices, signals=filtered,
                  sizer=lambda s,p: vol_target(s,p,target_vol=0.08),
                  freq='1D', name='Yield Trend').run()
```

### Check net currency exposure before trading

```python
# Decompose positions into single-currency net notional
net = portfolio.net_ccy_exposure_usd(result.positions)
print("Today's net exposures:")
print(net.iloc[-1].sort_values())
```

### Risk attribution across pairs

```python
ra = portfolio.risk_attribution(result.pair_returns, freq='1D')
print(ra.sort_values('pct_vol_contrib', ascending=False))
```

### Compare strategies cleanly

```python
metrics.compare(
    (r_momentum.returns,  'Momentum'),
    (r_carry.returns,     'Carry'),
    (r_blend.returns,     'Blend'),
    (r_wf.returns,        'Walk-Forward OOS'),
    freq='1D',
)
```

### Use rolling VaR to monitor a live strategy

```python
var_ts = portfolio.rolling_var(result.returns, window=260, confidence=0.05)
var_ts.plot(title='Rolling 95% Daily VaR')
```
