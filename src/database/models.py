"""SQLAlchemy models for EVE market data"""

from datetime import datetime
from sqlalchemy import Column, Integer, BigInteger, Numeric, Boolean, String, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base

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
    volume_remain = Column(Integer, nullable=False)
    volume_total = Column(Integer, nullable=False)
    is_buy_order = Column(Boolean, nullable=False)
    duration = Column(Integer)
    issued = Column(DateTime)
    min_volume = Column(Integer, default=1)
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
    """Model predictions for evaluation"""

    __tablename__ = "predictions"

    prediction_time = Column(DateTime, primary_key=True, nullable=False)
    type_id = Column(Integer, primary_key=True, nullable=False)
    region_id = Column(Integer, primary_key=True, nullable=False)
    prediction_horizon = Column(Integer, primary_key=True, nullable=False)  # hours ahead
    current_price = Column(Numeric(20, 2), nullable=False)
    predicted_price = Column(Numeric(20, 2), nullable=False)
    model_version = Column(String(50))
    actual_price = Column(Numeric(20, 2), nullable=True)  # filled in later for evaluation
    confidence = Column(Numeric(5, 4), nullable=True)

    __table_args__ = (
        Index("idx_pred_type_time", "type_id", "prediction_time"),
    )

    def __repr__(self):
        return f"<Prediction(time={self.prediction_time}, type={self.type_id}, predicted={self.predicted_price})>"


class Item(Base):
    """Reference data for tradeable items"""

    __tablename__ = "items"

    type_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100))
    liquidity_tier = Column(String(20))  # high, medium, low
    avg_daily_volume = Column(BigInteger)
    avg_daily_orders = Column(Integer)
    is_tradeable = Column(Boolean, default=True)

    def __repr__(self):
        return f"<Item(id={self.type_id}, name={self.name})>"
