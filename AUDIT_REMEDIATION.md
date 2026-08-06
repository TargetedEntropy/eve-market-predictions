# Audit Remediation

What changed in response to the audit in `fixes.md`, section by section.
Branch: `fix/audit-remediation`.

Read this alongside `README.md`, which now describes the system as it actually
behaves.

---

## Headline changes

| Before | After |
|--------|-------|
| Hypertables silently never created (raw SQL never executed) | All DDL goes through `text()`; init fails loudly if TimescaleDB is missing |
| Feature job crashed on every run; no features ever stored | Non-numeric columns dropped, naive-UTC timestamps, bulk upserts |
| `/predict` 404'd on every request | Features are looked up at-or-before the target time |
| One shared `best_model.pth` overwritten by whichever item trained last | One checkpoint + JSON metadata per `(type_id, region_id)` |
| Scaler never persisted; raw 0-1 output returned as an ISK price | Scaler params stored as JSON and inverse-transformed at inference |
| Scheduler crash-looped on `RETRAINING_DAY=sunday` | Config normalizes day names and rejects nonsense at startup |
| Predictions written to JSON files only | Every prediction path writes to the `predictions` table |
| Model accuracy never measured | `actual_price` backfill + `model_performance` table + daily job |
| Error-limit handling was log-only; 420 just raised | Shared gate pauses all in-flight work; exponential backoff for 5xx |
| Zero tests | 185 tests, no external services required |

---

## 1. Broken / incorrect implementations

| Item | Fix |
|------|-----|
| 1.1 `init_db()` raw SQL | All statements wrapped in `sqlalchemy.text()`. Hypertable creation is verified afterwards and `init_db(require_timescaledb=True)` raises `TimescaleDBUnavailable` rather than warning. `scripts/init_database.py --allow-missing-timescaledb` opts out. |
| 1.2 Feature pipeline crash | `FeatureEngine._numeric_features` drops `time`, strings, datetimes, NaN and inf; timestamps are naive UTC throughout; features are bulk-upserted. Order-book features now read one complete snapshot instead of "the newest 1000 rows". |
| 1.3 `get_feature_vector` | Selects the newest feature set at-or-before `target_time`, with an optional `max_age` freshness limit. |
| 1.4 `/predict` inference | Rebuilt on `src/models/pipeline.py`: the last `lookback_window` prices are scaled with the model's persisted scaler, run through the model, and inverse-transformed. `confidence` is `1 - validation MAPE`, not `0.85`. The request takes `horizon_days` (the model is daily) instead of an ignored `horizon_hours`. |
| 1.5 Shared checkpoint | `src/models/artifacts.py` writes `model_{type_id}_{region_id}.pth` plus a JSON sidecar (scaler, feature columns, architecture, metrics, version). The API serves the model matching the request and returns 503 otherwise. |
| 1.6 `RETRAINING_DAY=sunday` | `src/config.py` normalizes full day names to APScheduler's form and rejects invalid values. `.env` and `.env.example` corrected. |
| 1.7 Dashboard `asyncio.run()` | `src/database/connection.run_async` runs every dashboard query on one long-lived event loop, so pooled asyncpg connections stay valid. |
| 1.8 Training leakage | Chronological split (`chronological_split`), scaler fit on the training portion only and persisted with the model. |
| 1.9 MAPE / squeeze | `regression_metrics` masks zero targets; `outputs.squeeze(-1)` everywhere. |

## 2. Scheduling gaps

All eight jobs live in `scripts/scheduler_daemon.py` with
`max_instances=1`, `coalesce=True` and a 300s misfire grace (2.5).

- **2.1** Daily market history, item metadata, liquidity tiers, and reference prices.
- **2.2** `retrain_models()` is real: per item, chronological training, per-item checkpoint, MLflow logging, and promotion only when validation improves.
- **2.3** Daily drift detection and daily prediction generation. Drift triggers retraining when `RETRAIN_ON_DRIFT=True`.
- **2.4** Daily prediction scoring (`src/monitoring/evaluation.py`) fills `actual_price` and writes `model_performance`.

## 3. Data consistency

- **3.1** One convention: naive UTC. `src/utils/timeutils` provides `utcnow()`, `to_naive_utc()` and `parse_esi_datetime()`; `datetime.now()` and `datetime.utcnow()` are gone from library code.
- **3.2** `src/constants.py` holds `TRACKED_ITEMS` and `REGIONS`; the collector, dashboard, drift script, predict scripts and the EVE Ref importer all import it. Labels corrected against ESI (40520 = Large Skill Injector, 40519 = Skill Extractor, 11399 = Morphite). A test fails if any module reintroduces its own list.
- **3.3** `Item.category` holds a resolved category *name*; `group_id`/`group_name` are separate columns. Metadata refreshes instead of skipping existing rows, with group lookups memoized per run.
- **3.4** `volume_remain`, `volume_total` and `min_volume` are `BigInteger`, with idempotent `ALTER ... TYPE BIGINT` migrations applied by `init_db`.
- **3.5** The "30-Day Change" card compares against the observation closest to 30 days back, and shows `N/A` when there is not enough history.
- **3.6** `src/features/technical_indicators_old.py` deleted.

## 4. Data gaps

- **4.1** Redis is used: `src/utils/cache.py` stores ESI payloads and ETags, degrading to a no-op when Redis is unreachable.
- **4.2** `generate_predictions()` writes one `Prediction` row per horizon day, with `target_time`, `model_version` and `confidence`. JSON export is opt-in (`--json-dir`).
- **4.3** The dashboard overlays the latest stored forecast on the price chart and marks realized prices once scored.
- **4.4** `order_snapshots` gets compression (2 days) and retention (30 days) policies, both configurable.
- **4.5** The dashboard's region dropdown is built from `COLLECTION_REGION_IDS`, so it can no longer offer a region that is never collected. The collector handles multiple regions; the EVE Ref importer has explicit `--region-id` / `--all-regions` flags.

## 5. ESI rate-limit safety

- **5.1** `ESIRateLimiter` is a shared gate. Below `ESI_ERROR_LIMIT_FLOOR` remaining errors, or on HTTP 420, *every* task blocks until the reset window elapses. 5xx and timeouts retry with exponential backoff plus jitter. A `User-Agent` identifying the app is now sent.
- **5.2** `ETag`/`If-None-Match` conditional requests backed by Redis; 304 reuses the cached payload and extends its TTL.
- **5.3** Pagination follows `X-Pages`; the extra pages go through the same aiometer pacing as everything else.
- **5.4** The EVE Ref importer paces requests, retries with backoff and skips unpublished days.

## 6. Performance

- **6.1 / 6.2 / 6.3** `src/database/upsert.py` provides chunked `INSERT ... ON CONFLICT DO UPDATE`, used by the collector, the feature engine and the EVE Ref importer (which also stream-filters the CSV before building rows). Duplicate keys within a batch are collapsed, so a retried run updates rather than failing.
- **6.4** `get_session()` no longer auto-commits; callers own their transactions.
- **6.5** Dashboard refresh is `DASHBOARD_REFRESH_SECONDS` (default 900) with a short-TTL query cache.

## 7. Security

- **7.1** `debug=False` unless `DASHBOARD_DEBUG` is set; the container runs gunicorn against `src.dashboard.app:server`.
- **7.2** CORS origins come from settings; wildcard-plus-credentials is rejected by a config validator.
- **7.3** `torch.load(..., weights_only=True)` everywhere; the scaler lives in JSON, not a pickle.
- **7.4** Postgres and Redis are no longer published to the host; MLflow, the API and the dashboard bind to `127.0.0.1` by default. `DB_PASSWORD` and `REDIS_PASSWORD` have no defaults and Redis requires a password. **Rotating the weak password in your local `.env` is still a manual step** -- changing it here would have broken your running database.
- **7.5** 500s return a generic message and log the detail; the Docker image runs as a non-root user.

## 8. Missing deliverables

- **8.1** `tests/` holds 185 tests covering config validation, timestamps, indicators, feature extraction, ESI pagination/backoff/caching, upsert SQL, artifacts and scalers, training, the API contract, the scheduler and prediction scoring. `pyproject.toml` configures pytest (`asyncio_mode=auto`), black and ruff.
- **8.2** Lifespan handler instead of `@app.on_event`, `sqlalchemy.orm.declarative_base`, unused `NullPool` import removed, `vectorbt`/`matplotlib`/`seaborn` dropped from requirements, gunicorn adopted.

## 9. Docs and deployment

- **9.1** `MLFLOW_TRACKING_URI=http://mlflow:5000` for all app services; the collector and dashboard mount `./data`; health checks on api/dashboard/mlflow; `/health` returns 503 when nothing is servable. The one-off MLflow workaround in `predict_item.py` is gone. Training also probes MLflow with a 2s TCP connect first, so an unreachable server costs two seconds instead of minutes of client retries.
- **9.2** Implemented: Redis caching, weekly retraining, prediction persistence, `actual_price` backfill, the `model_performance` table, prediction-error monitoring, an alerting hook, MLflow registered models and real `model_version` values, plus naive/moving-average/linear-trend baselines scored next to every model. `ModelTrainer.evaluate()` is wired into training. Still roadmap and now labelled as such: ARIMA/XGBoost, hyperparameter search, backtesting, and the missing feature families.
- **9.3** `QUICKSTART.md` rewritten (it pointed at a nonexistent module); `SETUP_GUIDE.md` no longer tells you to overwrite `scripts/train_model.py` and lists six containers; script inventories updated; `LICENSE` added; `TECHNOLOGY_STACK.md`'s project tree matches reality.
  *One audit item was incorrect:* the issue-tracker URL in `README.md` does match the configured remote (`TargetedEntropy/eve-market-predictions`), so it was left alone.
- **9.4** `predict_and_save.py` deleted (folded into `predict_item.py`); prediction scripts no longer retrain unless asked (`--train`); the autoregressive horizon and its error-compounding caveat are documented; `import_everef_data.py` takes real CLI arguments.

---

## Breaking changes

1. **`MODEL_PATH` is replaced by `MODEL_DIR`.** Existing `data/models/best_model.pth` files are ignored -- retrain with `python scripts/train_model.py --all`.
2. **`/predict` takes `horizon_days`, not `horizon_hours`,** and returns `item_name`, `daily[]` and a real `model_version`.
3. **`/health` returns 503 when no model is servable** (previously 200 with `status: degraded`).
4. **`DB_PASSWORD` and `REDIS_PASSWORD` are mandatory** for docker-compose.
5. **Postgres (5432) and Redis (6379) are no longer published to the host.** Uncomment the loopback mapping in `docker-compose.yml` if you need `psql` from outside.
6. **`get_session()` no longer commits for you.**
7. **Feature names changed** with the single-library indicator rewrite (`MACD_12_26_9` → `macd`, `BBU_20_2.0` → `bb_upper`, and `ema_7` now exists). Old rows in `features` are stale.
8. **`scripts/predict_and_save.py` was removed.** Use `python scripts/predict_item.py 40520`.

## Deferred deliberately

- Rotating the weak `DB_PASSWORD` in the local `.env` (would break the running database; the file is git-ignored and annotated).
- Multivariate LSTM: the engineered features are stored but the model still consumes price only. `ModelMetadata.feature_columns` is the extension point.
- ARIMA/XGBoost baselines and hyperparameter search would add dependencies; they are labelled roadmap rather than implemented.
- Integration tests against a live TimescaleDB (no database was reachable in this environment). The `integration` pytest marker is registered for them.
