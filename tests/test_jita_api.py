"""Run with --noconftest: legacy tests/conftest.py imports the old ML stack."""
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import runpy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.jita.api import Settings, create_app

TOKEN = "test-only-token-" + "a" * 32
AUTH = {"Authorization": "Bearer " + TOKEN}


def document():
    now = datetime.now(timezone.utc)
    stamp = now.isoformat()
    item = {"type_id": 34, "item_name": "Tritanium", "current_price": 10,
            "book": {"best_bid": 9, "best_ask": 11, "observed_at": stamp},
            "history": [], "data_quality": {"status": "insufficient_history"},
            "signals": {"fee_entry_pct": 1, "fee_exit_pct": 1, "slippage_pct": 1},
            "forecasts": [{"horizon_days": 7, "target_date": (now + timedelta(days=7)).date().isoformat(),
                           "predicted_price": 12, "expected_return_pct": 20, "net_return_pct": 17,
                           "model_name": "persistence", "validation": {"status": "insufficient_history"}}]}
    return {"schema_version": 1, "run_id": "test-run", "generated_at": stamp,
            "data_as_of": stamp, "scope": "station", "station_id": 60003760,
            "system_id": 30000142, "region_id": 10000002,
            "price_basis": "jita_station_midpoint", "horizons_days": [1, 7, 30],
            "model": {"runtime": "cpu"}, "quality": {}, "items": [item]}


def publish(path, value):
    temporary = path / "snapshot.tmp"
    temporary.write_text(json.dumps(value))
    temporary.replace(path / "snapshot.json")


@pytest.fixture
def client(tmp_path):
    publish(tmp_path, document())
    with TestClient(create_app(Settings(artifact_dir=tmp_path, token=TOKEN))) as client:
        yield client


def test_auth_liveness_and_metrics(client):
    assert client.get("/v1/health").json() == {"data": {"status": "alive"}}
    for path in ("/metrics", "/v1/ready", "/v1/overview", "/v1/opportunities", "/v1/types/34", "/v1/models"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get("/metrics", headers=AUTH)
    assert response.status_code == 200
    assert "jita_snapshot_ready 1.0" in response.text
    assert TOKEN not in response.text


def test_empty_artifact_unready(tmp_path):
    with TestClient(create_app(Settings(artifact_dir=tmp_path, token=TOKEN))) as client:
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/ready", headers=AUTH).status_code == 503
        assert client.get("/v1/overview", headers=AUTH).status_code == 503


def test_missing_token_fails_startup(tmp_path):
    with pytest.raises(RuntimeError, match="must be configured"):
        with TestClient(create_app(Settings(artifact_dir=tmp_path))):
            pass


def test_stale_explicit_and_not_ready(tmp_path):
    data = document()
    data["data_as_of"] = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    publish(tmp_path, data)
    with TestClient(create_app(Settings(artifact_dir=tmp_path, token=TOKEN))) as client:
        response = client.get("/v1/overview", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["meta"]["stale"] is True
        assert response.json()["meta"]["age_seconds"] >= 259000
        assert client.get("/v1/ready", headers=AUTH).status_code == 503


@pytest.mark.parametrize("field,value", [("station_id", 60008494), ("scope", "region"),
    ("price_basis", "regional_trade_average"), ("schema_version", True), ("system_id", "30000142")])
def test_scope_fail_closed(tmp_path, field, value):
    data = document()
    data[field] = value
    publish(tmp_path, data)
    with TestClient(create_app(Settings(artifact_dir=tmp_path, token=TOKEN))) as client:
        assert client.get("/v1/overview", headers=AUTH).status_code == 503


def test_corrupt_replacement_does_not_serve_cached_good_snapshot(tmp_path):
    publish(tmp_path, document())
    with TestClient(create_app(Settings(artifact_dir=tmp_path, token=TOKEN))) as client:
        assert client.get("/v1/overview", headers=AUTH).status_code == 200
        (tmp_path / "snapshot.json").write_text('{"secret":"must-not-leak"')
        response = client.get("/v1/overview", headers=AUTH)
        assert response.status_code == 503
        assert "must-not-leak" not in response.text
        assert str(tmp_path) not in response.text
        publish(tmp_path, document())
        assert client.get("/v1/overview", headers=AUTH).status_code == 200


def test_filters_order_detail_and_bounds(tmp_path):
    data = document()
    second = copy.deepcopy(data["items"][0])
    second["type_id"] = 35
    second["forecasts"][0]["net_return_pct"] = None
    data["items"].append(second)
    publish(tmp_path, data)
    with TestClient(create_app(Settings(artifact_dir=tmp_path, token=TOKEN))) as client:
        result = client.get("/v1/opportunities", headers=AUTH).json()
        assert [row["type_id"] for row in result["data"]] == [34, 35]
        assert "forecasts" not in result["data"][0] and "history" in result["data"][0]
        assert len(client.get("/v1/opportunities?min_return_pct=0&limit=1", headers=AUTH).json()["data"]) == 1
        assert client.get("/v1/opportunities?type_id=999", headers=AUTH).json()["data"] == []
        assert client.get("/v1/types/34", headers=AUTH).json()["data"]["type_id"] == 34
        assert client.get("/v1/types/999", headers=AUTH).status_code == 404
        assert client.get("/v1/models", headers=AUTH).json()["data"] == {"runtime": "cpu"}
        for query in ("limit=0", "limit=501", "horizon_days=2", "horizon_days=31", "min_return_pct=nan"):
            assert client.get("/v1/opportunities?" + query, headers=AUTH).status_code == 422


def test_snapshot_cache_does_not_reparse_same_inode(client, monkeypatch):
    assert client.get("/v1/overview", headers=AUTH).status_code == 200
    def forbidden(*args, **kwargs):
        raise AssertionError("cached file must not be re-read")
    monkeypatch.setattr(Path, "open", forbidden)
    assert client.get("/v1/overview", headers=AUTH).status_code == 200


def test_oversize_and_nonfinite_fail_closed(tmp_path):
    data = document()
    data["quality"] = {"huge": "x" * 2048}
    publish(tmp_path, data)
    with TestClient(create_app(Settings(artifact_dir=tmp_path, token=TOKEN, max_snapshot_bytes=1024))) as client:
        assert client.get("/v1/overview", headers=AUTH).status_code == 503
    data = document()
    data["quality"] = {"invalid": float("nan")}
    publish(tmp_path, data)
    with TestClient(create_app(Settings(artifact_dir=tmp_path, token=TOKEN))) as client:
        assert client.get("/v1/overview", headers=AUTH).status_code == 503


def test_rate_limit_bounded(tmp_path):
    with TestClient(create_app(Settings(artifact_dir=tmp_path, token=TOKEN, requests_per_minute=1))) as client:
        assert client.get("/v1/models", headers=AUTH).status_code == 503
        assert client.get("/v1/models", headers=AUTH).status_code == 429
        assert client.get("/v1/health").status_code == 200


def test_cpu_import_has_no_legacy_or_torch():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import src.jita.api; assert 'torch' not in sys.modules; "
        "assert 'src.config' not in sys.modules; assert 'sqlalchemy' not in sys.modules"], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("failure", ["preflight", "batch_timeout", "success"])
def test_gpu_wrapper_deadlines_and_atomic_metrics(tmp_path, monkeypatch, failure):
    calls = []
    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1 and failure == "preflight":
            raise subprocess.CalledProcessError(1, args)
        if len(calls) == 2 and failure == "batch_timeout":
            raise subprocess.TimeoutExpired(args, 2700)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("JITA_ARTIFACT_DIR", str(tmp_path))
    wrapper = runpy.run_path(str(Path(__file__).resolve().parents[1] / "deploy/run_worker.py"))
    code = wrapper["main"]()
    assert calls[0][1] == {"timeout": 60, "check": True}
    if failure != "preflight":
        assert calls[1][1] == {"timeout": 2700, "check": False}
        assert "--require-cuda" in calls[1][0]
    assert (code == 0) is (failure == "success")
    assert (tmp_path / "last-success.prom").exists() is (failure == "success")
    assert "jita_batch_last_success " + str(int(failure == "success")) in (tmp_path / "batch.prom").read_text()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("available,cuda,count,name,cap,arch,reason", [
    (False, "12.8", 1, "NVIDIA GeForce RTX 5080", (12, 0), ["sm_120"], "CUDA required"),
    (True, "12.6", 1, "NVIDIA GeForce RTX 5080", (12, 0), ["sm_120"], "CUDA 12.8"),
    (True, "12.8", 2, "NVIDIA GeForce RTX 5080", (12, 0), ["sm_120"], "exactly one"),
    (True, "12.8", 1, "NVIDIA GeForce RTX 4070 Ti", (8, 9), ["sm_89"], "RTX 5080"),
    (True, "12.8", 1, "NVIDIA GeForce RTX 5080", (12, 0), ["sm_90"], "sm_120"),
])
def test_gpu_preflight_rejects_incompatible_runtime(monkeypatch, available, cuda, count, name, cap, arch, reason):
    fake = SimpleNamespace(version=SimpleNamespace(cuda=cuda), cuda=SimpleNamespace(
        is_available=lambda: available, device_count=lambda: count,
        get_device_properties=lambda _: SimpleNamespace(name=name, major=cap[0], minor=cap[1]),
        get_arch_list=lambda: arch))
    monkeypatch.setitem(sys.modules, "torch", fake)
    preflight = runpy.run_path(str(Path(__file__).resolve().parents[1] / "deploy/gpu_preflight.py"))
    with pytest.raises(RuntimeError, match=reason):
        preflight["check_gpu"]()
