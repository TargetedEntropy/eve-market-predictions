# Jita prediction platform: data and correctness contract

This is the station-scoped companion to the EveMarket prediction workstation.
It is not a claim that an untrained neural network can print ISK. A baseline
which refuses to invent evidence is preferable to a profitable-looking lie.

## Scope and data flow

Only Jita IV - Moon 4 - Caldari Navy Assembly Plant, station **60003760**, in
system **30000142**, region **10000002**, is in scope. The Forge is a region,
not an alternative spelling of Jita.

1. EveMarket's existing worker remains the sole owner of its ESI ingestion.
2. Authenticated `/api/predictions/source/*` routes export stored Jita facts.
3. A Nomad GPU batch fetches those facts with a dedicated read API key,
   accumulates station history, evaluates candidates and publishes a snapshot.
4. A separate CPU-only API reads that atomic snapshot. No web request trains a
   model, waits on ESI, or needs a GPU.
5. EveMarket's authenticated backend proxies typed, bounded prediction reads.
   Browser clients never receive the service token or source API key.
6. The workstation renders the same run across overview, risk, forecast,
   liquidity, validation and multi-monitor panels.

## What the dataset actually means

EveMarket has approximately 35 days of rolling five-minute market statistics.
The `hub_bid` and `hub_ask` fields are station prices. The similarly positioned
`bid_units` and `ask_units` fields are **region-wide stock** and must not be used
as station depth. The source therefore exports daily Jita midpoint observations
and snapshot counts, with **null historical station depth**. Current depth is
summed from live orders restricted to Jita's station and region.

Daily history is a quote-derived midpoint proxy, not traded VWAP or OHLCV.
Standing order stock is not executed volume. Neither unknown station execution
volume nor a probability of profit is synthesized. Both remain unavailable.

The new artifact volume accumulates up to 730 days per type, independently of
the source's 35-day retention. Only a bounded recent history is embedded in each
serving snapshot. Keep and back up the history directory: rebuilding the image
does not reconstruct station observations which the source has already expired.

## Forecast and evidence policy

- Direct 1-, 7- and 30-calendar-day forecast targets, not recursive fake OHLCV.
- Today's incomplete daily aggregate is excluded from training.
- Missing calendar days are not compressed into apparently adjacent days.
- Training, feature normalization and target normalization see each fold's
  historical prefix only.
- An actual PyTorch return-regression MLP is compared with price persistence
  using chronological, non-overlapping horizon folds.
- Neural selection requires improvement on earlier selection folds and later
  audit folds. The audit also participates in the fallback gate, so its metric
  is **not an untouched final test estimate**.
- If history or validation is insufficient, persistence is served by name.
  A 30-day forecast does not imply a 30-day backtest exists.
- Interval bands are empirical historical relative-residual quantiles, emitted
  only with enough horizon-specific residuals. They are not calibrated odds,
  guaranteed coverage, or a chance of making money.
- Fees, a spread assumption and explicit slippage reduce scenario returns.
  Quotes and total standing depth cannot establish executable position size.
  The platform recommends; it never places, edits or cancels EVE orders.

### Cold-start expectations

With the default lookback of 14 days, 32 training labels and 12 independent
selection/audit origins, a neural candidate first becomes eligible at roughly
**59 completed contiguous days for 1-day forecasts, 137 for 7-day forecasts,
and 436 for 30-day forecasts**. Passing the eligibility threshold does not
mean it will beat persistence. Gaps reset the usable contiguous suffix.

An initial source import of roughly 34 completed days therefore cannot prove
neural edge, especially at 30 days. The baseline and factual station screens
work immediately; history accumulation supplies future evaluation evidence.
This is a deliberate constraint, not a reason to relabel regional history as
station observations or loosen validation until a colorful number appears.

## Operational boundaries

Nomad configuration and hardware acceptance are in [JITA-NOMAD.md](JITA-NOMAD.md).
The GPU batch must execute a CUDA kernel before it accepts the GPU environment.
Blackwell-capable wheels are required for an RTX 5080; driver detection alone
does not prove kernel support. The deployment must fail rather than silently
running the GPU job on CPU.

The batch limits the selected universe and request concurrency. The initial
workstation is bounded to the most relevant selected station books, not a claim
to have trained every Jita type. Actual coverage and fallback status are data.
Do not label an initial baseline-only run a trained production model.

## Validation and rollout

Run `python -m pytest --noconftest tests/test_jita_*.py -q` in the isolated Jita
environment. The legacy root test fixtures import the legacy model stack, so
`--noconftest` is intentional. The CPU image must work without PyTorch.

Synthetic training tests establish mechanics, not market edge. The paired
EveMarket PR includes a cross-repository integration harness which uses a
disposable PostgreSQL database and the real source, engine, serving API and
authenticated proxy. Browser tests independently exercise the workstation.

Before production promotion, operators must separately verify the RTX 5080,
Nomad NVIDIA device plugin, shared volume, least-privilege secrets, real source
freshness, a completed batch, and same-run responses through EveMarket. No local
CPU test or HCL parse replaces those checks. Until the service is connected,
the site must show an explicit unavailable state rather than demonstration data.
