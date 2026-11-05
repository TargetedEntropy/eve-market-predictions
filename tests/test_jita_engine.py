"""No live EVE data or credentials are used in these synthetic unit fixtures."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.jita import engine
from src.jita.engine import EngineConfig, ForecastEngine
from src.jita.schemas import TypeSource

NOW = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)


def source(days=35):
    return TypeSource.model_validate({"type_id":34,"item_name":"Tritanium",
        "station_id":60003760,"region_id":10000002,
        "book":{"best_bid":99,"best_ask":101,"buy_volume":1000,"sell_volume":2000,
                "observed_at":NOW.isoformat()},
        "history":[{"date":(NOW.date()-timedelta(days=days-i)).isoformat(),
                    "mid_price":100+i*0.1,"snapshot_count":100,
                    "buy_volume":None,"sell_volume":None} for i in range(days)]})


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(engine, "resolve_device", lambda require_cuda=False: "cpu")
    return ForecastEngine()


def test_short_history_horizon_support_and_honest_liquidity(cpu):
    result = cpu.forecast_item(source(), NOW)
    one, seven, thirty = result.forecasts
    assert [f.horizon_days for f in result.forecasts] == [1, 7, 30]
    assert one.validation.backtest_samples == 20
    assert one.lower_price is not None and one.validation.interval_samples == 20
    assert seven.lower_price is None
    assert thirty.validation.backtest_samples == 0
    assert thirty.lower_price is None and thirty.validation.model_mae is None
    assert all(f.model_name == "naive_persistence" for f in result.forecasts)
    assert all(f.target_date == NOW.date()+timedelta(days=f.horizon_days) for f in result.forecasts)
    assert result.signals.probability_of_profit is None
    assert result.signals.executed_daily_volume is None
    assert result.signals.actionable is False
    assert result.signals.book_imbalance == pytest.approx(-1/3)
    assert one.net_return_pct < 0


def test_fees_spread_slippage():
    cfg = EngineConfig(entry_fee_pct=1,exit_fee_pct=2,slippage_pct=0.5)
    expected = ((100*0.99*0.98*0.995)/(101*1.01)-1)*100
    assert engine.net_return(100,101,0.02,cfg) == pytest.approx(expected)
    assert engine.net_return(100,None,0.02,cfg) is None
    with pytest.raises(ValueError):
        EngineConfig(exit_fee_pct=-1)


def test_gaps_not_compressed_and_partial_today_excluded(cpu):
    data = source(35)
    data.history.pop(-5)
    data.history.append(data.history[-1].model_copy(update={"date":NOW.date(),"mid_price":1e8}))
    result = cpu.forecast_item(data,NOW)
    assert result.data_quality["contiguous_completed_days"] == 4
    assert result.data_quality["partial_today_excluded"]
    assert result.data_quality["gapped_history"]
    assert all(f.validation.backtest_samples == 0 for f in result.forecasts)


def test_stale_and_future_rejection(cpu):
    data = source()
    data.book.observed_at = NOW-timedelta(hours=7)
    result = cpu.forecast_item(data,NOW)
    assert not result.data_quality["fresh_book"]
    assert all(f.net_return_pct is None and f.lower_price is None for f in result.forecasts)
    data.book.observed_at = NOW+timedelta(seconds=1)
    with pytest.raises(ValueError,match="future"):
        cpu.forecast_item(data,NOW)
    with pytest.raises(ValueError):
        TypeSource.model_validate({**source().model_dump(),"station_id":60008494})


def test_walk_forward_never_sees_target(monkeypatch):
    prices = np.exp(np.arange(100)*0.002)
    cfg = EngineConfig(lookback=4,min_train_samples=6,max_folds=8,epochs=2)
    prefixes = []
    def fake_fit(prefix,horizon,config,device):
        prefixes.append(prefix.copy())
        return float(prefix[-1]),{}
    monkeypatch.setattr(engine,"train_predict",fake_fit)
    folds = engine.walk_forward(prices,7,cfg)
    for fold,prefix in zip(folds,prefixes):
        assert len(prefix) == fold["origin"]+1
        np.testing.assert_array_equal(prefix,prices[:fold["origin"]+1])
        assert fold["train_end"] < fold["target"]
    assert all(b["origin"]-a["origin"] == 7 for a,b in zip(folds,folds[1:]))
    # Mutating a future endpoint cannot change the training prefix for that fold.
    modified = prices.copy()
    modified[folds[0]["target"]:] *= 100
    prefixes.clear()
    engine.walk_forward(modified,7,cfg)
    np.testing.assert_array_equal(prefixes[0],prices[:folds[0]["origin"]+1])


def test_training_arrays_labels_stay_inside_prefix():
    prices = np.exp(np.arange(20)*0.1)
    x,y = engine.make_training_arrays(prices,7,4)
    assert x.shape == (9,4)
    np.testing.assert_allclose(y,0.7)


def test_empirical_band_support():
    folds = [{"actual":101,"baseline":100} for _ in range(19)]
    assert engine.empirical_band(200,folds,"baseline",20) == (None,None)
    folds.append({"actual":101,"baseline":100})
    assert engine.empirical_band(200,folds,"baseline",20) == pytest.approx((202,202))


def test_tiny_torch_cpu_training_is_real_and_reproducible():
    pytest.importorskip("torch")
    prices = np.exp(4+np.sin(np.arange(55)*0.25)*0.04+np.arange(55)*0.001)
    cfg = EngineConfig(lookback=4,min_train_samples=8,epochs=4,hidden_units=4)
    a,meta = engine.train_predict(prices,7,cfg,"cpu")
    b,again = engine.train_predict(prices,7,cfg,"cpu")
    assert a > 0 and np.isfinite(a)
    assert a == b and meta["weights_sha256"] == again["weights_sha256"]
    assert meta["training_samples"] == 44
    assert np.isfinite(meta["training_loss_standardized_mse"])
    c,_ = engine.train_predict(prices,7,replace(cfg,epochs=1),"cpu")
    assert c != a


def test_selection_failure_uses_baseline(cpu,monkeypatch):
    cpu.config = EngineConfig(lookback=2,min_train_samples=2,min_selection_samples=2,
                             min_audit_samples=2,epochs=1)
    def bad_folds(prices,horizon,config,device):
        return [{"actual":100,"baseline":100,"candidate":150} for _ in range(8)]
    monkeypatch.setattr(engine,"walk_forward",bad_folds)
    result = cpu.forecast_item(source(100),NOW)
    assert result.forecasts[0].model_name == "naive_persistence"
    assert result.forecasts[0].validation.candidate_mae == 50
    assert result.forecasts[0].validation.model_mae == 0


def test_cuda_required_fails_without_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="CUDA required"):
        engine.resolve_device(True)


def test_snapshot_cpu_contract(cpu):
    snap = cpu.snapshot([source()],NOW)
    assert snap.schema_version == 1
    assert snap.price_basis == "jita_station_midpoint"
    assert snap.station_id == 60003760
    assert snap.model["device"] == "cpu"
    assert len(snap.model["config_sha256"]) == 64
    assert snap.model_dump_json()


async def test_batch_accumulates_and_publishes_real_contract(cpu,monkeypatch,tmp_path):
    from types import SimpleNamespace
    from src.jita import cli
    from src.jita.storage import SnapshotStore
    # Controlled source fixture, not a live request. Retimestamp to actual batch clock.
    data = source()
    now = datetime.now(timezone.utc)
    shift = now.date()-NOW.date()
    data.book.observed_at = now
    for row in data.history:
        row.date += shift
    class MockSource:
        skipped_type_ids = []
        def __init__(self,*a,**kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*a): pass
        async def fetch_all(self,**kwargs): return [data]
    monkeypatch.setattr(cli,"SourceClient",MockSource)
    monkeypatch.setattr(cli,"ForecastEngine",lambda *a,**kw:cpu)
    monkeypatch.setenv("JITA_SOURCE_URL","https://example.test")
    monkeypatch.setenv("JITA_SOURCE_TOKEN","not-a-real-key")
    args = SimpleNamespace(output_dir=tmp_path,entry_fee_pct=0,exit_fee_pct=3.6,
        slippage_pct=0.5,require_cuda=False,concurrency=1,limit=1)
    result = await cli.run_batch(args)
    saved = SnapshotStore(tmp_path).load()
    assert result["items"] == 1 and saved.items[0].type_id == 34
    assert (tmp_path/"history"/"34.json").exists()


def test_actual_neural_gate_can_select_learned_candidate(cpu,monkeypatch):
    cpu.config = EngineConfig(lookback=4,min_train_samples=8,min_selection_samples=2,
                             min_audit_samples=2,epochs=3,hidden_units=4)
    def good_folds(prices,horizon,config,device):
        return [{"actual":101,"baseline":100,"candidate":101} for _ in range(8)]
    monkeypatch.setattr(engine,"walk_forward",good_folds)
    result = cpu.forecast_item(source(100),NOW)
    assert result.forecasts[0].model_name == "direct_return_mlp"
    assert result.forecasts[0].validation.candidate_mae == 0
    assert cpu.fit_metadata["34:1"]["training_samples"] > 0


def test_real_walkforward_cpu_smoke():
    cfg = EngineConfig(lookback=4,min_train_samples=6,max_folds=3,epochs=2,hidden_units=4)
    prices = np.exp(4+np.arange(40)*0.001+0.01*np.sin(np.arange(40)))
    folds = engine.walk_forward(prices,7,cfg,"cpu")
    assert len(folds) == 3
    assert all(f["candidate"] > 0 and f["target"]-f["origin"] == 7 for f in folds)
