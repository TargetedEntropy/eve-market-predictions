# Jita implementation validation, 2026-11-05

These are observed local results, not a production/GPU deployment certificate.

| Check | Observed result |
| --- | --- |
| Full legacy + new prediction suite | 260 passed |
| Minimal pinned Jita environment, no legacy fixtures | 75 passed |
| Paired EveMarket backend, disposable PostgreSQL | 408 passed, 10 skipped |
| Paired frontend unit tests | 65 passed |
| Isolated Chromium workstation suite | 14 passed |
| Production frontend build | Passed |
| Real cross-repository source/engine/serving/proxy/built-browser harness | 53 assertions passed |
| CPU Docker target | Built; imports without torch; non-root/read-only health/auth checked |
| GPU Docker target | Built; torch 2.7.1+cu128 imports, CUDA build 12.8 |
| CUDA preflight without an allocated GPU | Correctly failed; no CPU fallback |
| Both jobs, checksum-verified Nomad 1.10.5 CLI | Job validation successful; runtime driver config not checked offline |

The cross-repository test seeds actual disposable PostgreSQL rows, issues and
revokes a real database-backed read key, reads through the actual source SQL and
HTTP serialization, runs actual Torch fitting and chronological evaluation,
publishes/reads canonical snapshots, and calls the standalone API through the
authenticated EveMarket proxy. Chromium renders the built SPA using those actual
service responses. Network transport is bridged through ASGI; no production
data or private credentials are used. A reduced smoke model configuration proves
training mechanics, not production model skill.

It also executes the actual production-default batch orchestration twice and
checks archival deduplication/immutable runs, then makes the fixture books stale
and verifies a failed batch preserves the previous snapshot byte-for-byte.

Browser fixtures for separate UI tests are explicitly labeled synthetic. The
responsive suite checks document overflow at 375, 768 and 1280 pixels. Screenshots
were visually inspected; none is presented as live market evidence.

## Reproduce

For this repository, install `requirements-jita-test.txt` and CPU torch 2.7.1 in
an isolated environment, then run:

```sh
python -m pytest --noconftest tests/test_jita_*.py -q
```

The full legacy suite needs its existing legacy dependency stack in addition.
For cross-repository acceptance, check out both repositories as siblings and
follow the guarded fixture prerequisites in EveMarket's
`scripts/test-jita-fullstack.py`, then run that script with `--browser` using an
environment containing both declared backend and prediction dependencies and
Python Playwright. It creates and removes only its own randomly named fixture
database. It refuses a checkout containing a backend `.env`.

## Not established

- Real RTX 5080 kernel execution or production allocation runtime.
- A running Nomad cluster's NVIDIA device plugin/runtime or CSI locking/rename behavior.
- Live source credentials, real station history coverage or fresh production snapshots.
- Profitable predictive edge, calibrated coverage, execution volume or fill assumptions.
- A deployment of either repository to the existing public site.

All remain explicit operator acceptance gates. Existing legacy dependency
warnings remain; the new isolated platform does not require the legacy LSTM,
MLflow, database or dashboard runtime in its serving image.
