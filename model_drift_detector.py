"""
Model Drift Detection System
=============================
Production-grade model monitoring and drift detection.

Features:
- Feature drift detection (PSI, KS-test)
- Prediction distribution shift
- Performance degradation alerts
- Auto-retrain triggers
- Data quality monitoring

Based on:
- MLOps best practices
- Production ML monitoring standards
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
from enum import Enum
from scipy import stats
import json
import os
import pickle


class DriftType(Enum):
    """Types of drift"""
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    CONCEPT_DRIFT = "concept_drift"
    DATA_QUALITY = "data_quality"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Alert for detected drift"""
    timestamp: datetime
    drift_type: DriftType
    level: AlertLevel
    feature: str
    metric_name: str
    metric_value: float
    threshold: float
    message: str
    recommended_action: str = ""


@dataclass
class DriftReport:
    """Comprehensive drift report"""
    timestamp: datetime
    alerts: List[DriftAlert]
    feature_scores: Dict[str, float]
    prediction_drift_score: float
    overall_health: str  # "healthy", "warning", "critical"
    recommendations: List[str]
    requires_retraining: bool = False


class PopulationStabilityIndex:
    """
    Population Stability Index (PSI) calculator.
    
    PSI measures how much a distribution has shifted.
    - PSI < 0.1: No significant shift (healthy)
    - 0.1 <= PSI < 0.2: Moderate shift (warning)
    - PSI >= 0.2: Significant shift (critical)
    """
    
    @staticmethod
    def calculate(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate PSI between reference and current distributions.
        
        Args:
            reference: Reference/baseline distribution
            current: Current distribution to compare
            n_bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Handle edge cases
        if len(reference) < n_bins or len(current) < n_bins:
            return 0.0
        
        # Create bins based on reference distribution
        _, bin_edges = np.histogram(reference, bins=n_bins)
        
        # Calculate proportions for each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions (add small epsilon to avoid division by zero)
        epsilon = 1e-6
        ref_props = (ref_counts + epsilon) / (len(reference) + epsilon * n_bins)
        cur_props = (cur_counts + epsilon) / (len(current) + epsilon * n_bins)
        
        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        return psi


class KolmogorovSmirnovTest:
    """
    Kolmogorov-Smirnov test for distribution comparison.
    
    Tests if two samples come from the same distribution.
    """
    
    @staticmethod
    def test(
        reference: np.ndarray,
        current: np.ndarray,
        significance_level: float = 0.05,
    ) -> Tuple[float, bool]:
        """
        Perform KS test.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            significance_level: Significance level for the test
            
        Returns:
            Tuple of (statistic, is_significantly_different)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        is_different = p_value < significance_level
        
        return statistic, is_different


class FeatureDriftDetector:
    """
    Detect drift in input features.
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        psi_threshold_warning: float = 0.1,
        psi_threshold_critical: float = 0.2,
        ks_significance: float = 0.05,
    ):
        """
        Initialize feature drift detector.
        
        Args:
            reference_data: Reference/training data for comparison
            psi_threshold_warning: PSI threshold for warning
            psi_threshold_critical: PSI threshold for critical alert
            ks_significance: Significance level for KS test
        """
        self.reference_data = reference_data
        self.psi_threshold_warning = psi_threshold_warning
        self.psi_threshold_critical = psi_threshold_critical
        self.ks_significance = ks_significance
        
        # Store feature statistics
        self.feature_stats = {}
        self._calculate_reference_stats()
        
        logger.info(f"FeatureDriftDetector initialized with {len(reference_data.columns)} features")
    
    def _calculate_reference_stats(self):
        """Calculate reference statistics for each feature"""
        for col in self.reference_data.columns:
            if self.reference_data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                self.feature_stats[col] = {
                    "mean": self.reference_data[col].mean(),
                    "std": self.reference_data[col].std(),
                    "min": self.reference_data[col].min(),
                    "max": self.reference_data[col].max(),
                    "median": self.reference_data[col].median(),
                }
    
    def check_drift(
        self,
        current_data: pd.DataFrame,
    ) -> Dict[str, Dict]:
        """
        Check for feature drift.
        
        Args:
            current_data: Current data to check
            
        Returns:
            Dictionary of drift results per feature
        """
        results = {}
        
        for col in self.reference_data.columns:
            if col not in current_data.columns:
                continue
            
            if self.reference_data[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                continue
            
            ref_values = self.reference_data[col].dropna().values
            cur_values = current_data[col].dropna().values
            
            if len(ref_values) < 10 or len(cur_values) < 10:
                continue
            
            # Calculate PSI
            psi = PopulationStabilityIndex.calculate(ref_values, cur_values)
            
            # Perform KS test
            ks_stat, ks_different = KolmogorovSmirnovTest.test(
                ref_values, cur_values, self.ks_significance
            )
            
            # Determine drift level
            if psi >= self.psi_threshold_critical or (ks_different and ks_stat > 0.3):
                drift_level = AlertLevel.CRITICAL
            elif psi >= self.psi_threshold_warning or ks_different:
                drift_level = AlertLevel.WARNING
            else:
                drift_level = AlertLevel.INFO
            
            results[col] = {
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_different": ks_different,
                "drift_level": drift_level,
                "ref_mean": self.feature_stats[col]["mean"],
                "cur_mean": current_data[col].mean(),
                "mean_shift": abs(current_data[col].mean() - self.feature_stats[col]["mean"]) / (self.feature_stats[col]["std"] + 1e-6),
            }
        
        return results


class PredictionDriftDetector:
    """
    Detect drift in model predictions.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        psi_threshold: float = 0.15,
    ):
        """
        Initialize prediction drift detector.
        
        Args:
            window_size: Number of predictions to compare
            psi_threshold: PSI threshold for drift
        """
        self.window_size = window_size
        self.psi_threshold = psi_threshold
        
        self.prediction_history: List[float] = []
        self.confidence_history: List[float] = []
        self.reference_predictions: Optional[np.ndarray] = None
    
    def set_reference(self, predictions: np.ndarray):
        """Set reference prediction distribution"""
        self.reference_predictions = predictions
        logger.info(f"Set reference with {len(predictions)} predictions")
    
    def add_prediction(self, prediction: float, confidence: float = 1.0):
        """Add new prediction to history"""
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        # Keep only recent history
        if len(self.prediction_history) > self.window_size * 2:
            self.prediction_history = self.prediction_history[-self.window_size * 2:]
            self.confidence_history = self.confidence_history[-self.window_size * 2:]
    
    def check_drift(self) -> Dict:
        """
        Check for prediction drift.
        
        Returns:
            Dictionary with drift metrics
        """
        if len(self.prediction_history) < self.window_size:
            return {"status": "insufficient_data", "n_predictions": len(self.prediction_history)}
        
        current_predictions = np.array(self.prediction_history[-self.window_size:])
        
        # Use reference if available, otherwise use older predictions
        if self.reference_predictions is not None and len(self.reference_predictions) >= self.window_size:
            reference = self.reference_predictions
        elif len(self.prediction_history) >= self.window_size * 2:
            reference = np.array(self.prediction_history[-self.window_size * 2:-self.window_size])
        else:
            return {"status": "insufficient_reference"}
        
        # Calculate PSI
        psi = PopulationStabilityIndex.calculate(reference, current_predictions)
        
        # Check for class imbalance shift (for classification)
        ref_positive_rate = np.mean(reference > 0.5) if len(reference) > 0 else 0.5
        cur_positive_rate = np.mean(current_predictions > 0.5)
        
        # Confidence drift
        current_confidence = np.array(self.confidence_history[-self.window_size:])
        confidence_mean = np.mean(current_confidence)
        confidence_std = np.std(current_confidence)
        
        return {
            "status": "checked",
            "psi": psi,
            "is_drifting": psi > self.psi_threshold,
            "ref_positive_rate": ref_positive_rate,
            "cur_positive_rate": cur_positive_rate,
            "positive_rate_shift": abs(cur_positive_rate - ref_positive_rate),
            "confidence_mean": confidence_mean,
            "confidence_std": confidence_std,
            "drift_level": AlertLevel.CRITICAL if psi > 0.2 else (AlertLevel.WARNING if psi > 0.1 else AlertLevel.INFO),
        }


class PerformanceMonitor:
    """
    Monitor model performance over time.
    """
    
    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        degradation_threshold: float = 0.10,  # 10% degradation triggers alert
    ):
        """
        Initialize performance monitor.
        
        Args:
            baseline_metrics: Baseline performance metrics
            degradation_threshold: Threshold for performance degradation
        """
        self.baseline_metrics = baseline_metrics
        self.degradation_threshold = degradation_threshold
        
        self.metric_history: Dict[str, List[float]] = {k: [] for k in baseline_metrics}
        self.timestamp_history: List[datetime] = []
    
    def add_metrics(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None):
        """Add performance metrics"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.timestamp_history.append(timestamp)
        
        for key, value in metrics.items():
            if key in self.metric_history:
                self.metric_history[key].append(value)
    
    def check_degradation(self, window: int = 10) -> Dict[str, Dict]:
        """
        Check for performance degradation.
        
        Args:
            window: Number of recent metrics to average
            
        Returns:
            Dictionary of degradation results per metric
        """
        results = {}
        
        for metric, values in self.metric_history.items():
            if len(values) < window:
                continue
            
            baseline = self.baseline_metrics.get(metric, 0)
            if baseline == 0:
                continue
            
            recent_avg = np.mean(values[-window:])
            
            # Calculate degradation (higher is worse for most metrics like win_rate)
            if metric in ["accuracy", "win_rate", "profit_factor", "sharpe_ratio"]:
                degradation = (baseline - recent_avg) / baseline
            else:  # Lower is worse (e.g., loss, drawdown)
                degradation = (recent_avg - baseline) / abs(baseline)
            
            results[metric] = {
                "baseline": baseline,
                "current": recent_avg,
                "degradation_pct": degradation,
                "is_degraded": degradation > self.degradation_threshold,
                "trend": self._calculate_trend(values[-window:]),
            }
        
        return results
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 3:
            return "stable"
        
        # Simple linear regression
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)
        
        if abs(r_value) < 0.5:
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "declining"


class DataQualityMonitor:
    """
    Monitor data quality issues.
    """
    
    def __init__(
        self,
        max_missing_rate: float = 0.05,
        max_outlier_rate: float = 0.02,
        max_constant_rate: float = 0.10,
    ):
        """
        Initialize data quality monitor.
        
        Args:
            max_missing_rate: Maximum acceptable missing data rate
            max_outlier_rate: Maximum acceptable outlier rate
            max_constant_rate: Maximum rate of constant values
        """
        self.max_missing_rate = max_missing_rate
        self.max_outlier_rate = max_outlier_rate
        self.max_constant_rate = max_constant_rate
    
    def check_quality(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Check data quality.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Quality report per column
        """
        results = {}
        n_rows = len(data)
        
        for col in data.columns:
            if data[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                continue
            
            # Missing rate
            missing_rate = data[col].isna().mean()
            
            # Outlier rate (using IQR method)
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((data[col] < q1 - 3 * iqr) | (data[col] > q3 + 3 * iqr)).mean()
            
            # Constant rate
            value_counts = data[col].value_counts(normalize=True)
            constant_rate = value_counts.iloc[0] if len(value_counts) > 0 else 0
            
            # Determine alert level
            issues = []
            if missing_rate > self.max_missing_rate:
                issues.append("high_missing")
            if outliers > self.max_outlier_rate:
                issues.append("high_outliers")
            if constant_rate > self.max_constant_rate:
                issues.append("high_constant")
            
            results[col] = {
                "missing_rate": missing_rate,
                "outlier_rate": outliers,
                "constant_rate": constant_rate,
                "issues": issues,
                "is_healthy": len(issues) == 0,
                "alert_level": AlertLevel.CRITICAL if len(issues) > 1 else (AlertLevel.WARNING if len(issues) == 1 else AlertLevel.INFO),
            }
        
        return results


class ModelDriftMonitor:
    """
    Comprehensive model drift monitoring system.
    
    Integrates all drift detection components.
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        baseline_metrics: Dict[str, float],
        model_name: str = "trading_model",
        alert_callback: Optional[Callable[[DriftAlert], None]] = None,
    ):
        """
        Initialize model drift monitor.
        
        Args:
            reference_data: Reference/training data
            baseline_metrics: Baseline performance metrics
            model_name: Name of the model being monitored
            alert_callback: Callback for alerts
        """
        self.model_name = model_name
        self.alert_callback = alert_callback
        
        # Initialize detectors
        self.feature_detector = FeatureDriftDetector(reference_data)
        self.prediction_detector = PredictionDriftDetector()
        self.performance_monitor = PerformanceMonitor(baseline_metrics)
        self.data_quality_monitor = DataQualityMonitor()
        
        # Alert history
        self.alert_history: List[DriftAlert] = []
        self.report_history: List[DriftReport] = []
        
        logger.info(f"ModelDriftMonitor initialized for {model_name}")
    
    def add_prediction(self, prediction: float, confidence: float = 1.0):
        """Add a prediction for monitoring"""
        self.prediction_detector.add_prediction(prediction, confidence)
    
    def add_performance_metrics(self, metrics: Dict[str, float]):
        """Add performance metrics"""
        self.performance_monitor.add_metrics(metrics)
    
    def run_check(
        self,
        current_data: Optional[pd.DataFrame] = None,
    ) -> DriftReport:
        """
        Run comprehensive drift check.
        
        Args:
            current_data: Current data for feature drift analysis
            
        Returns:
            DriftReport with all findings
        """
        alerts = []
        recommendations = []
        feature_scores = {}
        requires_retraining = False
        
        # Check feature drift
        if current_data is not None:
            feature_results = self.feature_detector.check_drift(current_data)
            
            for feature, result in feature_results.items():
                feature_scores[feature] = result["psi"]
                
                if result["drift_level"] in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
                    alert = DriftAlert(
                        timestamp=datetime.now(),
                        drift_type=DriftType.FEATURE_DRIFT,
                        level=result["drift_level"],
                        feature=feature,
                        metric_name="psi",
                        metric_value=result["psi"],
                        threshold=self.feature_detector.psi_threshold_warning,
                        message=f"Feature {feature} has drifted (PSI={result['psi']:.3f})",
                        recommended_action="Review feature engineering and data source",
                    )
                    alerts.append(alert)
                    
                    if result["drift_level"] == AlertLevel.CRITICAL:
                        requires_retraining = True
            
            # Data quality check
            quality_results = self.data_quality_monitor.check_quality(current_data)
            unhealthy_features = [f for f, r in quality_results.items() if not r["is_healthy"]]
            
            if unhealthy_features:
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    drift_type=DriftType.DATA_QUALITY,
                    level=AlertLevel.WARNING,
                    feature=", ".join(unhealthy_features[:3]),
                    metric_name="quality_issues",
                    metric_value=len(unhealthy_features),
                    threshold=0,
                    message=f"{len(unhealthy_features)} features have data quality issues",
                    recommended_action="Check data pipeline and preprocessing",
                )
                alerts.append(alert)
        
        # Check prediction drift
        prediction_result = self.prediction_detector.check_drift()
        prediction_drift_score = prediction_result.get("psi", 0.0)
        
        if prediction_result.get("is_drifting", False):
            alert = DriftAlert(
                timestamp=datetime.now(),
                drift_type=DriftType.PREDICTION_DRIFT,
                level=prediction_result.get("drift_level", AlertLevel.WARNING),
                feature="predictions",
                metric_name="psi",
                metric_value=prediction_drift_score,
                threshold=self.prediction_detector.psi_threshold,
                message=f"Prediction distribution has shifted (PSI={prediction_drift_score:.3f})",
                recommended_action="Investigate model behavior and retrain if needed",
            )
            alerts.append(alert)
            
            if prediction_drift_score > 0.2:
                requires_retraining = True
        
        # Check performance degradation
        performance_results = self.performance_monitor.check_degradation()
        
        for metric, result in performance_results.items():
            if result.get("is_degraded", False):
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    drift_type=DriftType.CONCEPT_DRIFT,
                    level=AlertLevel.WARNING if result["degradation_pct"] < 0.2 else AlertLevel.CRITICAL,
                    feature=metric,
                    metric_name="degradation_pct",
                    metric_value=result["degradation_pct"],
                    threshold=self.performance_monitor.degradation_threshold,
                    message=f"Performance metric {metric} degraded by {result['degradation_pct']:.1%}",
                    recommended_action="Consider model retraining or parameter tuning",
                )
                alerts.append(alert)
                
                if result["degradation_pct"] > 0.2:
                    requires_retraining = True
        
        # Generate recommendations
        if requires_retraining:
            recommendations.append("Model retraining is recommended based on detected drift")
        
        if any(a.drift_type == DriftType.FEATURE_DRIFT for a in alerts):
            recommendations.append("Review data sources and feature engineering pipeline")
        
        if any(a.drift_type == DriftType.DATA_QUALITY for a in alerts):
            recommendations.append("Investigate data quality issues in preprocessing")
        
        # Determine overall health
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        warning_alerts = [a for a in alerts if a.level == AlertLevel.WARNING]
        
        if critical_alerts:
            overall_health = "critical"
        elif warning_alerts:
            overall_health = "warning"
        else:
            overall_health = "healthy"
        
        # Create report
        report = DriftReport(
            timestamp=datetime.now(),
            alerts=alerts,
            feature_scores=feature_scores,
            prediction_drift_score=prediction_drift_score,
            overall_health=overall_health,
            recommendations=recommendations,
            requires_retraining=requires_retraining,
        )
        
        # Store report
        self.report_history.append(report)
        self.alert_history.extend(alerts)
        
        # Trigger callbacks for alerts
        if self.alert_callback:
            for alert in alerts:
                self.alert_callback(alert)
        
        logger.info(f"Drift check complete: {overall_health} ({len(alerts)} alerts)")
        
        return report
    
    def get_summary(self) -> Dict:
        """Get monitoring summary"""
        recent_alerts = self.alert_history[-10:] if self.alert_history else []
        
        return {
            "model_name": self.model_name,
            "total_alerts": len(self.alert_history),
            "critical_alerts": sum(1 for a in self.alert_history if a.level == AlertLevel.CRITICAL),
            "warning_alerts": sum(1 for a in self.alert_history if a.level == AlertLevel.WARNING),
            "recent_alerts": [
                {
                    "time": a.timestamp.isoformat(),
                    "type": a.drift_type.value,
                    "level": a.level.value,
                    "message": a.message,
                }
                for a in recent_alerts
            ],
            "requires_retraining": any(r.requires_retraining for r in self.report_history[-5:]) if self.report_history else False,
        }
    
    def save_state(self, filepath: str):
        """Save monitor state"""
        state = {
            "model_name": self.model_name,
            "alert_history": [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "drift_type": a.drift_type.value,
                    "level": a.level.value,
                    "feature": a.feature,
                    "metric_name": a.metric_name,
                    "metric_value": a.metric_value,
                    "message": a.message,
                }
                for a in self.alert_history
            ],
        }
        
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved monitor state to {filepath}")


def create_drift_monitor(
    training_data: pd.DataFrame,
    baseline_accuracy: float = 0.65,
    baseline_win_rate: float = 0.60,
    baseline_sharpe: float = 2.0,
) -> ModelDriftMonitor:
    """
    Factory function to create drift monitor.
    
    Args:
        training_data: Training data for reference
        baseline_accuracy: Baseline model accuracy
        baseline_win_rate: Baseline trading win rate
        baseline_sharpe: Baseline Sharpe ratio
        
    Returns:
        Configured ModelDriftMonitor
    """
    baseline_metrics = {
        "accuracy": baseline_accuracy,
        "win_rate": baseline_win_rate,
        "sharpe_ratio": baseline_sharpe,
        "profit_factor": 2.0,
    }
    
    return ModelDriftMonitor(
        reference_data=training_data,
        baseline_metrics=baseline_metrics,
        model_name="trading_model",
    )


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    logger.info("=" * 60)
    logger.info("Testing Model Drift Detection System")
    logger.info("=" * 60)
    
    # Create sample reference data
    np.random.seed(42)
    n_samples = 500
    
    reference_data = pd.DataFrame({
        "feature1": np.random.normal(0, 1, n_samples),
        "feature2": np.random.normal(5, 2, n_samples),
        "feature3": np.random.exponential(2, n_samples),
        "feature4": np.random.uniform(-1, 1, n_samples),
    })
    
    baseline_metrics = {
        "accuracy": 0.68,
        "win_rate": 0.62,
        "sharpe_ratio": 2.5,
    }
    
    # Create monitor
    monitor = ModelDriftMonitor(
        reference_data=reference_data,
        baseline_metrics=baseline_metrics,
        model_name="test_model",
    )
    
    # Simulate some predictions
    for _ in range(150):
        monitor.add_prediction(np.random.random(), np.random.uniform(0.5, 1.0))
    
    # Create drifted current data
    current_data = pd.DataFrame({
        "feature1": np.random.normal(0.5, 1.2, 100),  # Mean shifted
        "feature2": np.random.normal(5, 2, 100),      # Same
        "feature3": np.random.exponential(3, 100),    # Scale changed
        "feature4": np.random.uniform(-1, 1, 100),    # Same
    })
    
    # Add some performance metrics (simulating degradation)
    for i in range(20):
        degradation = i * 0.02  # Gradual degradation
        monitor.add_performance_metrics({
            "accuracy": 0.68 - degradation,
            "win_rate": 0.62 - degradation * 0.5,
            "sharpe_ratio": 2.5 - degradation * 2,
        })
    
    # Run drift check
    print("\n" + "=" * 60)
    print("DRIFT CHECK RESULTS")
    print("=" * 60)
    
    report = monitor.run_check(current_data)
    
    print(f"\nOverall Health: {report.overall_health.upper()}")
    print(f"Requires Retraining: {report.requires_retraining}")
    print(f"Prediction Drift Score (PSI): {report.prediction_drift_score:.4f}")
    
    print(f"\nFeature Drift Scores:")
    for feature, score in report.feature_scores.items():
        status = "⚠️" if score > 0.1 else "✅"
        print(f"  {status} {feature}: PSI = {score:.4f}")
    
    print(f"\nAlerts ({len(report.alerts)}):")
    for alert in report.alerts:
        icon = "🔴" if alert.level == AlertLevel.CRITICAL else "🟡"
        print(f"  {icon} [{alert.level.value}] {alert.message}")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  → {rec}")
    
    # Get summary
    summary = monitor.get_summary()
    print(f"\nMonitor Summary:")
    print(f"  Total Alerts: {summary['total_alerts']}")
    print(f"  Critical: {summary['critical_alerts']}")
    print(f"  Warnings: {summary['warning_alerts']}")
    
    print("\n✅ Model Drift Detection Test Complete")
