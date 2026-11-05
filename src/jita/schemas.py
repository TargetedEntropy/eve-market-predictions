"""Shared CPU-readable artifact and station-source contracts."""
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

STATION_ID = 60003760
SYSTEM_ID = 30000142
REGION_ID = 10000002


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Book(Model):
    best_bid: FiniteFloat | None = Field(default=None, gt=0)
    best_ask: FiniteFloat | None = Field(default=None, gt=0)
    buy_volume: FiniteFloat | None = Field(default=None, ge=0)
    sell_volume: FiniteFloat | None = Field(default=None, ge=0)
    observed_at: datetime

    @model_validator(mode="after")
    def valid_book(self):
        if self.observed_at.tzinfo is None:
            raise ValueError("book timestamp must be timezone-aware")
        if self.best_bid is not None and self.best_ask is not None and self.best_bid > self.best_ask:
            raise ValueError("crossed station book")
        return self


class HistoryRow(Model):
    date: date
    mid_price: FiniteFloat = Field(gt=0)
    best_bid: FiniteFloat | None = Field(default=None, gt=0)
    best_ask: FiniteFloat | None = Field(default=None, gt=0)
    buy_volume: FiniteFloat | None = Field(default=None, ge=0)
    sell_volume: FiniteFloat | None = Field(default=None, ge=0)
    snapshot_count: int = Field(ge=1)


class UniverseItem(Book):
    type_id: int = Field(gt=0)
    item_name: str = Field(min_length=1, max_length=300)
    market_group_id: int | None = None
    market_group_name: str | None = None


class TypeSource(Model):
    type_id: int = Field(gt=0)
    item_name: str = Field(min_length=1, max_length=300)
    station_id: Literal[60003760]
    region_id: Literal[10000002]
    system_id: Literal[30000142] = SYSTEM_ID
    market_group_id: int | None = None
    market_group_name: str | None = None
    history: list[HistoryRow] = Field(max_length=730)
    book: Book


class Validation(Model):
    status: str
    backtest_samples: int = 0
    selection_samples: int = 0
    interval_samples: int = 0
    baseline_mae: FiniteFloat | None = None
    model_mae: FiniteFloat | None = None
    candidate_mae: FiniteFloat | None = None
    selection_baseline_mae: FiniteFloat | None = None
    selection_model_mae: FiniteFloat | None = None
    directional_accuracy: FiniteFloat | None = None
    interval_method: str | None = None
    interval_nominal_coverage: FiniteFloat | None = None
    training_observations: int = 0
    last_training_date: date | None = None


class Forecast(Model):
    horizon_days: Literal[1, 7, 30]
    target_date: date
    predicted_price: FiniteFloat = Field(gt=0)
    lower_price: FiniteFloat | None = Field(default=None, ge=0)
    upper_price: FiniteFloat | None = Field(default=None, ge=0)
    expected_return_pct: FiniteFloat
    net_return_pct: FiniteFloat | None = None
    model_name: str
    validation: Validation


class Signals(Model):
    spread_pct: FiniteFloat | None = None
    book_imbalance: FiniteFloat | None = None
    standing_bid_depth: FiniteFloat | None = None
    standing_ask_depth: FiniteFloat | None = None
    executed_daily_volume: None = None
    probability_of_profit: None = None
    liquidity_basis: str = "standing_order_stock_not_executed_volume"
    fee_entry_pct: FiniteFloat
    fee_exit_pct: FiniteFloat
    slippage_pct: FiniteFloat
    actionable: bool = False


class ItemForecast(Model):
    type_id: int
    item_name: str
    market_group_id: int | None = None
    market_group_name: str | None = None
    current_price: FiniteFloat = Field(gt=0)
    book: Book
    history: list[HistoryRow] = Field(max_length=90)
    data_quality: dict[str, Any]
    forecasts: list[Forecast]
    signals: Signals


class Snapshot(Model):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    generated_at: datetime
    data_as_of: datetime
    scope: Literal["station"] = "station"
    station_id: Literal[60003760] = STATION_ID
    system_id: Literal[30000142] = SYSTEM_ID
    region_id: Literal[10000002] = REGION_ID
    price_basis: Literal["jita_station_midpoint"] = "jita_station_midpoint"
    horizons_days: list[int] = Field(default_factory=lambda: [1, 7, 30])
    model: dict[str, Any]
    quality: dict[str, Any]
    items: list[ItemForecast] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_snapshot(self):
        if self.generated_at.tzinfo is None or self.data_as_of.tzinfo is None:
            raise ValueError("snapshot timestamps must be timezone-aware")
        if self.data_as_of > self.generated_at:
            raise ValueError("data_as_of cannot be in the future")
        if len({item.type_id for item in self.items}) != len(self.items):
            raise ValueError("duplicate item type")
        return self
