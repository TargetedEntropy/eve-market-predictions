"""Model drift detection using Evidently AI"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json

import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
)
from sqlalchemy import select

from src.database import get_session, MarketHistory, Prediction


@dataclass
class DriftReport:
    """Drift detection report results"""

    timestamp: datetime
    type_id: int
    region_id: int
    data_drift_detected: bool
    prediction_drift_detected: bool
    drift_score: float
    drifted_features: List[str]
    metrics: Dict[str, float]
    report_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'type_id': self.type_id,
            'region_id': self.region_id,
            'data_drift_detected': self.data_drift_detected,
            'prediction_drift_detected': self.prediction_drift_detected,
            'drift_score': self.drift_score,
            'drifted_features': self.drifted_features,
            'metrics': self.metrics,
            'report_path': self.report_path,
        }


class DriftDetector:
    """Detect model drift using Evidently"""

    def __init__(
        self,
        reference_window_days: int = 30,
        current_window_days: int = 7,
        drift_threshold: float = 0.5,
    ):
        """
        Initialize drift detector

        Args:
            reference_window_days: Days of data for reference (baseline)
            current_window_days: Days of data for current comparison
            drift_threshold: Threshold for drift detection (0-1)
        """
        self.reference_window_days = reference_window_days
        self.current_window_days = current_window_days
        self.drift_threshold = drift_threshold

    async def detect_drift(
        self,
        type_id: int,
        region_id: int,
        save_report: bool = True,
    ) -> DriftReport:
        """
        Detect drift for a specific item

        Args:
            type_id: Item type ID
            region_id: Region ID
            save_report: Whether to save HTML report

        Returns:
            DriftReport with detection results
        """
        # Get reference and current data
        reference_data = await self._get_reference_data(type_id, region_id)
        current_data = await self._get_current_data(type_id, region_id)

        if reference_data.empty or current_data.empty:
            return DriftReport(
                timestamp=datetime.now(),
                type_id=type_id,
                region_id=region_id,
                data_drift_detected=False,
                prediction_drift_detected=False,
                drift_score=0.0,
                drifted_features=[],
                metrics={'error': 'Insufficient data'},
            )

        # Prepare dataframes
        reference_df = self._prepare_features(reference_data)
        current_df = self._prepare_features(current_data)

        # Define column mapping
        column_mapping = ColumnMapping(
            target='average',
            numerical_features=[
                'volume', 'highest', 'lowest', 'order_count',
                'price_change', 'volume_change',
            ],
        )

        # Create and run drift report
        report = Report(metrics=[
            DataDriftPreset(drift_share=self.drift_threshold),
            TargetDriftPreset(),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ])

        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping,
        )

        # Extract results
        results = report.as_dict()

        # Parse drift metrics
        drift_metrics = self._parse_drift_metrics(results)

        # Save HTML report if requested
        report_path = None
        if save_report:
            report_path = await self._save_report(
                report, type_id, region_id
            )

        return DriftReport(
            timestamp=datetime.now(),
            type_id=type_id,
            region_id=region_id,
            data_drift_detected=drift_metrics['data_drift_detected'],
            prediction_drift_detected=drift_metrics['target_drift_detected'],
            drift_score=drift_metrics['drift_score'],
            drifted_features=drift_metrics['drifted_features'],
            metrics=drift_metrics,
            report_path=report_path,
        )

    async def _get_reference_data(
        self, type_id: int, region_id: int
    ) -> pd.DataFrame:
        """Get reference (baseline) data"""
        end_date = datetime.now() - timedelta(days=self.current_window_days)
        start_date = end_date - timedelta(days=self.reference_window_days)

        async with get_session() as session:
            result = await session.execute(
                select(MarketHistory)
                .where(MarketHistory.type_id == type_id)
                .where(MarketHistory.region_id == region_id)
                .where(MarketHistory.time >= start_date)
                .where(MarketHistory.time < end_date)
                .order_by(MarketHistory.time)
            )
            records = result.scalars().all()

        return pd.DataFrame([{
            'time': r.time,
            'average': float(r.average) if r.average else 0.0,
            'highest': float(r.highest) if r.highest else 0.0,
            'lowest': float(r.lowest) if r.lowest else 0.0,
            'volume': r.volume if r.volume else 0,
            'order_count': r.order_count if r.order_count else 0,
        } for r in records])

    async def _get_current_data(
        self, type_id: int, region_id: int
    ) -> pd.DataFrame:
        """Get current (recent) data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.current_window_days)

        async with get_session() as session:
            result = await session.execute(
                select(MarketHistory)
                .where(MarketHistory.type_id == type_id)
                .where(MarketHistory.region_id == region_id)
                .where(MarketHistory.time >= start_date)
                .where(MarketHistory.time < end_date)
                .order_by(MarketHistory.time)
            )
            records = result.scalars().all()

        return pd.DataFrame([{
            'time': r.time,
            'average': float(r.average) if r.average else 0.0,
            'highest': float(r.highest) if r.highest else 0.0,
            'lowest': float(r.lowest) if r.lowest else 0.0,
            'volume': r.volume if r.volume else 0,
            'order_count': r.order_count if r.order_count else 0,
        } for r in records])

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for drift detection"""
        if df.empty:
            return df

        # Sort by time
        df = df.sort_values('time').copy()

        # Calculate derived features
        df['price_change'] = df['average'].pct_change().fillna(0)
        df['volume_change'] = df['volume'].pct_change().fillna(0)
        df['price_volatility'] = df['average'].rolling(7, min_periods=1).std().fillna(0)

        # Drop time column for drift detection
        df = df.drop(columns=['time'])

        # Fill any remaining NaN values
        df = df.fillna(0)

        return df

    def _parse_drift_metrics(self, results: dict) -> dict:
        """Parse Evidently results into metrics"""
        metrics = {
            'data_drift_detected': False,
            'target_drift_detected': False,
            'drift_score': 0.0,
            'drifted_features': [],
            'n_features': 0,
            'n_drifted_features': 0,
        }

        try:
            # Extract dataset drift
            for metric in results.get('metrics', []):
                metric_type = metric.get('metric')

                if metric_type == 'DatasetDriftMetric':
                    result = metric.get('result', {})
                    metrics['data_drift_detected'] = result.get('dataset_drift', False)
                    metrics['drift_score'] = result.get('drift_share', 0.0)
                    metrics['n_features'] = result.get('number_of_columns', 0)
                    metrics['n_drifted_features'] = result.get('number_of_drifted_columns', 0)

                    # Get drifted feature names
                    drift_by_columns = result.get('drift_by_columns', {})
                    metrics['drifted_features'] = [
                        col for col, info in drift_by_columns.items()
                        if info.get('drift_detected', False)
                    ]

                elif 'ColumnDriftMetric' in metric_type:
                    # Target drift
                    result = metric.get('result', {})
                    if result.get('column_name') == 'average':
                        metrics['target_drift_detected'] = result.get('drift_detected', False)
                        metrics['target_drift_score'] = result.get('drift_score', 0.0)

        except Exception as e:
            metrics['error'] = str(e)

        return metrics

    async def _save_report(
        self, report: Report, type_id: int, region_id: int
    ) -> str:
        """Save HTML drift report"""
        reports_dir = Path('data/drift_reports')
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'drift_report_{type_id}_{region_id}_{timestamp}.html'
        filepath = reports_dir / filename

        report.save_html(str(filepath))

        return str(filepath)

    async def check_all_items(
        self,
        items: List[Tuple[int, int]],
        save_reports: bool = True,
    ) -> List[DriftReport]:
        """
        Check drift for multiple items

        Args:
            items: List of (type_id, region_id) tuples
            save_reports: Whether to save HTML reports

        Returns:
            List of drift reports
        """
        reports = []

        for type_id, region_id in items:
            try:
                report = await self.detect_drift(
                    type_id=type_id,
                    region_id=region_id,
                    save_report=save_reports,
                )
                reports.append(report)
            except Exception as e:
                print(f"Error checking drift for {type_id}/{region_id}: {e}")

        return reports

    async def save_drift_summary(
        self, reports: List[DriftReport], output_path: str
    ):
        """Save drift detection summary to JSON"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_items': len(reports),
            'items_with_drift': sum(1 for r in reports if r.data_drift_detected),
            'items_with_target_drift': sum(1 for r in reports if r.prediction_drift_detected),
            'reports': [r.to_dict() for r in reports],
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Drift summary saved to: {output}")
        print(f"Items with data drift: {summary['items_with_drift']}/{summary['total_items']}")
        print(f"Items with target drift: {summary['items_with_target_drift']}/{summary['total_items']}")
