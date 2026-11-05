# Jita CPU serving and RTX 5080 Nomad deployment

Two actual jobs: CPU service and directly allocated GPU periodic batch. No service
is deployed by adding these files. Only operator-approved digest-pinned images and
provisioned storage/secrets may be promoted. Physical RTX 5080 verification remains
an operator acceptance gate, not a claim established by CPU tests or job validation.

## Source and artifact contract

`JITA_SOURCE_URL` is the HTTPS EveMarket origin, not an endpoint path. The worker
fetches authenticated `/api/predictions/source/universe` and
`/api/predictions/source/types/{type_id}` using `JITA_SOURCE_TOKEN`. Scope is always
station60003760/system30000142/region10000002. Daily history is observed station
midpoint, **not regional executed-trade history**. Standing depth is not executed
volume, and unsupported profit probabilities remain null.

`python -m src.jita.cli batch --output-dir /artifacts --require-cuda` accumulates
up to730 daily observations, trains, chronologically evaluates/selects against
persistence, produces empirical intervals when supported, forecasts1/7/30 days
and atomically publishes `snapshot.json` plus immutable `runs/<run_id>.json`.
Candidate model selection can legitimately retain persistence. GPU availability is
not evidence of sufficient historical training data or a superior neural forecast.

The periodic job runs hourly at :15 UTC with `prohibit_overlap=true`. The wrapper
enforces a45-minute subprocess wall-clock deadline; failures retain the last valid
snapshot and expose failure metrics. It does not silently fall back to CPU.

## CPU HTTP contract and EveMarket proxy

Standalone `uvicorn src.jita.api:app` imports only CPU-readable Pydantic schemas,
never legacy config, DB, torch or the training engine. It reads at most32MiB,
caches validated snapshots by inode/mtime/size (including failed versions), rejects
nonfinite numbers, and fails closed on corrupt replacement. Files must be atomically
replaced, not edited in place. Every other endpoint except liveness requires
`Authorization: Bearer <JITA_API_TOKEN>`, including metrics and readiness. Missing
or invalid24-256-byte token prevents startup. Use high-entropy ASCII secrets.
Equality uses constant-time comparison, credentials never enter query strings,
and access logging is disabled. Put TLS/private ingress in front of the API.

| GET | Response |
| --- | --- |
| `/v1/health` | Unauthenticated `{"data":{"status":"alive"}}`; only process liveness |
| `/v1/ready` | Authenticated503 absent, corrupt or stale;200 valid fresh artifact |
| `/v1/overview` | `data={tracked_types,forecast_count,horizons_days,quality}` |
| `/v1/opportunities?horizon_days=7&limit=100` | `data=[ItemForecast fields + one flattened Forecast]`, no `forecasts` array; includes last90 daily history rows for actual movers |
| `/v1/types/{type_id}` | `data=ItemForecast`, complete bounded history/horizon array |
| `/v1/models` | `data=snapshot.model`, real provenance and device/evaluation metadata |
| `/metrics` | Authenticated Prometheus exposition |

Snapshot responses are `{data,meta}`. Meta is exactly
`schema_version:1,run_id,generated_at,data_as_of,scope:"station",station_id:60003760,
system_id:30000142,region_id:10000002,price_basis:"jita_station_midpoint",stale,age_seconds`.
Timestamps are ISO8601 with timezone. Field definitions: `src/jita/schemas.py`.
Staleness uses source `data_as_of`, default3h, matching hourly refresh and the
engine's stricter per-item actionability checks. Stale data endpoints return200
with explicit meta.stale, while readiness returns503. UI may flag staleness earlier
as a conservative display warning, not a service-readiness assertion.

Opportunity filters: horizon1/7/30 only, limit1-500 (universe max200), optional
positive int32 `type_id`, finite `min_return_pct` from-100 to1,000,000. Sorted
net return descending, null last, type ID stable tie-breaker. Error codes401 auth,
404 unknown type,422 invalid params,429 rate limit,503 unavailable artifact.
Rate limiting: process-local token bucket default120/min/network client, bounded
1024-client LRU; job uses240/min for shared proxy. Proxy headers disabled. Scale
with ingress-wide limits, not uncoordinated process limits. Unauthenticated access
also needs upstream ingress limits to protect against hostile flooding.

EveMarket server env `PREDICTIONS_URL` points to the private CPU API origin and
`PREDICTIONS_TOKEN` matches API token. Never expose it to browsers. Proxy validates
envelopes/Jita scope. Refresh mixed `run_id` responses rather than combining two
artifact generations. Source token and serving token are different credentials.

## Images and runtime

```sh
docker build -f Dockerfile.jita --target api -t REGISTRY/jita-api:VERSION .
docker build -f Dockerfile.jita --target gpu-worker -t REGISTRY/jita-gpu:VERSION .
```

Linuxamd64, Python3.11, version-pinned minimal requirements. CPU image does not
install torch. Worker installs official **PyTorch2.7.1 cu128**, with CUDA12.8 and
native Blackwell `sm_120` support. The wrapper checks exactly one visible GPU,
RTX5080 name, capability12.0, CUDA12.8+, wheel sm_120 and a real finite CUDA
matrix multiplication before fetching source data. GPU job requests
`resources { device "nvidia/gpu" { count=1 } }` and constrains device model.
No privileged task, manual `/dev/nvidia*` bind or `NVIDIA_VISIBLE_DEVICES=all`.

Both images UID/GID10001, read-only roots in jobs, cap-dropALL/no-new-privileges.
Explicit `/tmp` tmpfs:64MiB API/1GiB worker, writable/nosuid/nodev. Worker allows
execution there because native libraries/CUDA tooling may require it. Persistent
artifact mount is RW only for worker, RO for API. Batch has no daemon healthcheck:
exit code and artifact/metrics timestamp are its health. Docker API healthcheck is
liveness only. Pin promoted registry images by digest; the Python base is version
pinned, not a supply-chain digest lock. Resolve/approve base digests and wheel
attestations/SBOMs before promotion.

## Operator provisioning prerequisites

No hosts are modified by these files or local checks.

1. Nomad1.10+ cluster with TLS, ACLs, workload identities and Variables, private
   registry access, Docker driver and CSI plugin. Never expose Nomad/Docker sockets.
2. Actual RTX5080 host: NVIDIA Linux driver supporting CUDA12.8/Blackwell, R570+
   or newer security-supported branch; compatible NVIDIA Container Toolkit and
   Docker `nvidia` runtime. Check host `nvidia-smi` and container CUDA first.
   Any daemon/client restart requires an approved maintenance window, especially
   on a shared desktop. Never modify a live graphical session to test this.
3. Checksum-verified `hashicorp/nomad-device-nvidia` binary in plugin_dir. Merge
   reviewed `deploy/nomad-client-gpu.example.hcl`, matching binary/plugin name.
   Verify `nomad node status -verbose` exposes healthy `nvidia/gpu` inventory and
   exact model name. Operator must verify driver/runtime plugin integration.
4. Provision/register shared **filesystem CSI** supporting
   `multi-node-multi-writer`, e.g. operator-managed NFS/CephFS. `artifact_volume_id`
   must be identical in both jobs. Initially10GiB with capacity/backup monitoring,
   UID/GID10001 ownership, no world permissions, and verified cross-client atomic
   rename/advisory locking. Do not mark a block-only volume as shared filesystem.
   Storage-provider credentials/provisioning are outside job files. Alternative
   single-client host volume requires changing both type fields to `host`, removing
   CSI access/attachment attributes and pinning both jobs to the exact same node.
   Matching host-volume names on separate nodes do not share content.
5. Provision namespace and least-privilege operator ACL, reviewing
   `deploy/operator-policy.hcl`. Workload identity implicit grants let each job
   read only its `nomad/jobs/<job>` variable path. If implicit grants are disabled,
   explicitly grant exact-path read only. Human submit/mount permissions must
   not confer cross-namespace secrets access; no management token in job env.
6. Provision encrypted Nomad Variables via secret-management automation, not
   committed var files, shell arguments or history:
   * `nomad/jobs/jita-api`: `api_token`.
   * `nomad/jobs/jita-gpu`: `source_url` HTTPS origin, `source_token`.
   Templates render allocation `secrets/`0400 env files and restart on rotation.
   Rotate proxy/API tokens together. Vault installations may replace template
   blocks with equivalent narrowly scoped Vault secret/identity policies.

## Deployment gates and rollback

Nonsecret per-job var files supply namespace, datacenter, artifact_volume_id and
either api_image or worker_image. Separate var files avoid undeclared variables.
Registry auth belongs in the Docker credential helper, not HCL. Set NOMAD_NAMESPACE.

```sh
nomad job validate -var-file=api.vars.hcl deploy/jita-api.nomad.hcl
nomad job validate -var-file=gpu.vars.hcl deploy/jita-gpu.nomad.hcl
nomad job plan -var-file=api.vars.hcl deploy/jita-api.nomad.hcl
nomad job plan -var-file=gpu.vars.hcl deploy/jita-gpu.nomad.hcl
# Only after operator approval:
nomad job run -var-file=api.vars.hcl deploy/jita-api.nomad.hcl
nomad job run -var-file=gpu.vars.hcl deploy/jita-gpu.nomad.hcl
nomad job periodic force jita-gpu
```

Initial readiness503 is expected until successful first publication. Registry
liveness does not assert artifact readiness, avoiding restart loops on batch
failures. Add a private authenticated readiness probe to ingress/promotion gates.

Acceptance evidence must include: digest/alloc IDs; actual5080 preflight and GPU
device metadata; batch success/validation provenance; cross-client shared output;
no CPU fallback on invalid GPU; unauthenticated401 for every protected endpoint;
fresh readiness200 and missing/corrupt/stale503; proxy Jita scope/run IDs; actual
UI history/chart values; null unsupported statistics; no overlapping batch; real
driver/runtime/plugin versions, elapsed time and peak GPU memory. HCL parsing or
offline validation does not validate runtime driver configuration or GPU hardware.

Rollback: stop/pause periodic parent and ensure no child batch is active. Restore
schema-valid immutable `runs/<run_id>.json` through a same-filesystem temporary
file, Pydantic validation and atomic rename to `snapshot.json`, never partial
in-place writes. Preserve bad artifacts elsewhere for investigation. Re-submit
previous digest-pinned job or `nomad job revert` API version. Resume periodic only
after review; repeat source age/readiness/scope/proxy checks. Restored old artifacts
may correctly remain stale. API auto-revert covers liveness only, not semantic
readiness regressions; those need operator intervention.

## Metrics and tests

Authenticated `/metrics`: request count route-template/status-class, latency by
route-template, ready/stale/source-age gauges. No type IDs, queries, tokens or run
IDs as labels. Scrape60s with secret-mounted bearer token and account for rate
limit. Alert on ready0, stale1, sustained5xx and age>10800s. Worker atomically writes
`batch.prom` and `last-success.prom` for operator-managed node-exporter textfile
collector RO mount; ignore `.tmp`, alert on absent metrics, failed last attempt,
or last success older than3h. Nomad failed allocations and JSON logs corroborate.

```sh
python -m pytest --noconftest tests/test_jita_api.py
```

The switch bypasses unrelated legacy conftest importing its old ML/DB stack. Tests
cover auth/startup, unready/stale/corrupt/oversize/nonfinite, strict scope, filters,
rate bounds, cache behavior and CPU-only imports. `nomad job validate` offline
reports driver configuration not checked. No RTX5080 run has been performed here.
