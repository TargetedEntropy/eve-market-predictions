"""SQLAlchemy models for EVE market data.

All ``DateTime`` columns are naive UTC (``TIMESTAMP WITHOUT TIME ZONE``).
Populate them via :func:`src.utils.timeutils.utcnow` -- never ``datetime.now()``.
"""

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Index, Integer, Numeric, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class MarketHistory(Base):
    """Daily aggregated market history"""

    __tablename__ = "market_history"

    time = Column(DateTime, primary_key=True, nullable=False)
    region_id = Column(Integer, primary_key=True, nullable=False)
    type_id = Column(Integer, primary_key=True, nullable=False)
    average = Column(Numeric(20, 2))
    highest = Column(Numeric(20, 2))
    lowest = Column(Numeric(20, 2))
    volume = Column(BigInteger)
    order_count = Column(Integer)

    __table_args__ = (
        Index("idx_mh_region_type", "region_id", "type_id", "time"),
    )

    def __repr__(self):
        return f"<MarketHistory(time={self.time}, region={self.region_id}, type={self.type_id})>"


class OrderSnapshot(Base):
    """Order book snapshots collected every 5 minutes"""

    __tablename__ = "order_snapshots"

    snapshot_time = Column(DateTime, primary_key=True, nullable=False)
    order_id = Column(BigInteger, primary_key=True, nullable=False)
    type_id = Column(Integer, nullable=False, index=True)
    region_id = Column(Integer, nullable=False)
    location_id = Column(BigInteger, nullable=False)
    price = Column(Numeric(20, 2), nullable=False)
    # Mineral orders (Tritanium and friends) routinely exceed 2^31 units.
    volume_remain = Column(BigInteger, nullable=False)
    volume_total = Column(BigInteger, nullable=False)
    is_buy_order = Column(Boolean, nullable=False)
    duration = Column(Integer)
    issued = Column(DateTime)
    min_volume = Column(BigInteger, default=1)
    range = Column(String(20))

    __table_args__ = (
        Index("idx_os_type_time", "type_id", "snapshot_time"),
        Index("idx_os_region_type", "region_id", "type_id"),
    )

    def __repr__(self):
        return f"<OrderSnapshot(time={self.snapshot_time}, order={self.order_id})>"


class Feature(Base):
    """Computed features for ML models"""

    __tablename__ = "features"

    time = Column(DateTime, primary_key=True, nullable=False)
    type_id = Column(Integer, primary_key=True, nullable=False)
    region_id = Column(Integer, primary_key=True, nullable=False)
    feature_name = Column(String(100), primary_key=True, nullable=False)
    feature_value = Column(Numeric, nullable=True)

    __table_args__ = (
        Index("idx_feat_type_time", "type_id", "time"),
    )

    def __repr__(self):
        return f"<Feature(time={self.time}, type={self.type_id}, name={self.feature_name})>"


class Prediction(Base):
    """Model predictions, written by every prediction path and scored later.

    ``actual_price`` is backfilled by ``scripts/evaluate_predictions.py`` once
    the horizon has elapsed and the realized price exists in ``market_history``.
    """

    __tablename__ = "predictions"

    prediction_time = Column(DateTime, primary_key=True, nullable=False)
    type_id = Column(Integer, primary_key=True, nullable=False)
    region_id = Column(Integer, primary_key=True, nullable=False)
    prediction_horizon = Column(Integer, primary_key=True, nullable=False)  # days ahead
    target_time = Column(DateTime, nullable=False)  # prediction_time + horizon
    current_price = Column(Numeric(20, 2), nullable=False)
    predicted_price = Column(Numeric(20, 2), nullable=False)
    model_version = Column(String(50))
    actual_price = Column(Numeric(20, 2), nullable=True)  # filled in by the evaluation job
    confidence = Column(Numeric(5, 4), nullable=True)

    __table_args__ = (
        Index("idx_pred_type_time", "type_id", "prediction_time"),
        Index("idx_pred_target", "target_time"),
    )

    def __repr__(self):
        return f"<Prediction(time={self.prediction_time}, type={self.type_id}, predicted={self.predicted_price})>"


class ModelPerformance(Base):
    """Realized accuracy of a model version, computed from scored predictions"""

    __tablename__ = "model_performance"

    time = Column(DateTime, primary_key=True, nullable=False)
    type_id = Column(Integer, primary_key=True, nullable=False)
    region_id = Column(Integer, primary_key=True, nullable=False)
    model_version = Column(String(50), primary_key=True, nullable=False)
    horizon_days = Column(Integer, primary_key=True, nullable=False)
    n_samples = Column(Integer, nullable=False)
    mae = Column(Float)
    rmse = Column(Float)
    mape = Column(Float)
    directional_accuracy = Column(Float)

    __table_args__ = (
        Index("idx_perf_type_time", "type_id", "time"),
    )

    def __repr__(self):
        return (
            f"<ModelPerformance(type={self.type_id}, version={self.model_version}, "
            f"mape={self.mape})>"
        )


class MarketPrice(Base):
    """Universe-wide reference prices from ESI ``/markets/prices/``"""

    __tablename__ = "market_prices"

    time = Column(DateTime, primary_key=True, nullable=False)
    type_id = Column(Integer, primary_key=True, nullable=False)
    adjusted_price = Column(Numeric(20, 2))
    average_price = Column(Numeric(20, 2))

    __table_args__ = (
        Index("idx_mp_type_time", "type_id", "time"),
    )

    def __repr__(self):
        return f"<MarketPrice(time={self.time}, type={self.type_id})>"


class Item(Base):
    """Reference data for tradeable items"""

    __tablename__ = "items"

    type_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    group_id = Column(Integer)
    group_name = Column(String(100))
    category = Column(String(100))  # resolved category *name*, not an id
    liquidity_tier = Column(String(20))  # high, medium, low
    avg_daily_volume = Column(BigInteger)
    avg_daily_orders = Column(Integer)
    is_tradeable = Column(Boolean, default=True)
    updated_at = Column(DateTime)

    def __repr__(self):
        return f"<Item(id={self.type_id}, name={self.name})>"
