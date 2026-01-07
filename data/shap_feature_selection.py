"""
SHAP Feature Selection
======================
ใช้ SHAP (SHapley Additive exPlanations) เพื่อเลือก features ที่สำคัญ

Features:
1. Calculate SHAP values for each feature
2. Rank features by importance
3. Select top N features
4. Remove redundant features
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Install with: pip install shap")


class SHAPFeatureSelector:
    """
    Feature selection using SHAP values
    
    ความสามารถ:
    1. คำนวณ SHAP importance
    2. เลือก top features
    3. ลบ redundant features
    """
    
    def __init__(
        self,
        n_top_features: int = 40,
        correlation_threshold: float = 0.95,
    ):
        self.n_top_features = n_top_features
        self.correlation_threshold = correlation_threshold
        
        self.feature_importance: Dict[str, float] = {}
        self.selected_features: List[str] = []
        
        logger.info(f"SHAPFeatureSelector initialized (top {n_top_features})")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model = None,
    ) -> List[str]:
        """
        คำนวณ SHAP values และเลือก features
        
        Args:
            X: Feature DataFrame
            y: Target array
            model: Trained model (optional, will train XGBoost if None)
            
        Returns:
            List of selected feature names
        """
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, using correlation-based selection")
            return self._fallback_selection(X, y)
        
        # Train model if not provided
        if model is None:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
            model.fit(X, y)
        
        # Calculate SHAP values
        logger.info("Calculating SHAP values...")
        
        try:
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            
            # Sample for faster computation
            sample_size = min(5000, len(X))
            X_sample = X.sample(n=sample_size, random_state=42)
            
            shap_values = explainer.shap_values(X_sample)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Calculate mean absolute SHAP value per feature
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance dict
            self.feature_importance = {
                col: float(imp) 
                for col, imp in zip(X.columns, mean_shap)
            }
            
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}, using fallback")
            return self._fallback_selection(X, y)
        
        # Sort by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        logger.info(f"Top 10 features by SHAP:")
        for feat, imp in sorted_features[:10]:
            logger.info(f"  {feat}: {imp:.4f}")
        
        # Select top features
        top_features = [f[0] for f in sorted_features[:self.n_top_features * 2]]
        
        # Remove highly correlated features
        self.selected_features = self._remove_correlated(
            X[top_features],
            top_features
        )
        
        # Limit to n_top_features
        self.selected_features = self.selected_features[:self.n_top_features]
        
        logger.info(f"Selected {len(self.selected_features)} features")
        
        return self.selected_features
    
    def _fallback_selection(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> List[str]:
        """Fallback selection using correlation"""
        
        # Calculate correlation with target
        correlations = {}
        for col in X.columns:
            try:
                corr = np.corrcoef(X[col].fillna(0).values, y)[0, 1]
                correlations[col] = abs(corr) if not np.isnan(corr) else 0
            except:
                correlations[col] = 0
        
        self.feature_importance = correlations
        
        # Sort by correlation
        sorted_features = sorted(
            correlations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_features = [f[0] for f in sorted_features[:self.n_top_features * 2]]
        
        # Remove correlated
        self.selected_features = self._remove_correlated(
            X[top_features],
            top_features
        )[:self.n_top_features]
        
        return self.selected_features
    
    def _remove_correlated(
        self,
        X: pd.DataFrame,
        features: List[str],
    ) -> List[str]:
        """Remove highly correlated features"""
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Track features to remove
        to_remove = set()
        
        for i, feat1 in enumerate(features):
            if feat1 in to_remove:
                continue
            
            for feat2 in features[i+1:]:
                if feat2 in to_remove:
                    continue
                
                try:
                    if corr_matrix.loc[feat1, feat2] > self.correlation_threshold:
                        # Remove the one with lower importance
                        if self.feature_importance.get(feat1, 0) >= self.feature_importance.get(feat2, 0):
                            to_remove.add(feat2)
                        else:
                            to_remove.add(feat1)
                            break
                except:
                    pass
        
        remaining = [f for f in features if f not in to_remove]
        
        logger.info(f"Removed {len(to_remove)} correlated features")
        
        return remaining
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select only the important features"""
        
        available = [f for f in self.selected_features if f in X.columns]
        return X[available]
    
    def get_importance_df(self) -> pd.DataFrame:
        """Get feature importance as DataFrame"""
        
        return pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ])
    
    def save(self, path: str):
        """Save selected features"""
        
        import json
        with open(path, 'w') as f:
            json.dump({
                'selected_features': self.selected_features,
                'feature_importance': self.feature_importance,
            }, f, indent=2)
        
        logger.info(f"Saved to {path}")
    
    def load(self, path: str):
        """Load selected features"""
        
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.selected_features = data['selected_features']
        self.feature_importance = data['feature_importance']
        
        logger.info(f"Loaded {len(self.selected_features)} features from {path}")


def select_features_shap(
    X: pd.DataFrame,
    y: np.ndarray,
    n_features: int = 40,
) -> Tuple[List[str], Dict[str, float]]:
    """Quick function to select features using SHAP"""
    
    selector = SHAPFeatureSelector(n_top_features=n_features)
    selected = selector.fit(X, y)
    
    return selected, selector.feature_importance


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   SHAP FEATURE SELECTION TEST")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    # Generate features
    X = pd.DataFrame({
        f"feature_{i}": np.random.randn(n)
        for i in range(50)
    })
    
    # Add some correlated features
    X['corr_1'] = X['feature_0'] * 0.9 + np.random.randn(n) * 0.1
    X['corr_2'] = X['feature_1'] * 0.95 + np.random.randn(n) * 0.05
    
    # Create target (depends on some features)
    y = (X['feature_0'] + X['feature_1'] * 2 + X['feature_2'] * 0.5 > 0).astype(int)
    
    # Run selection
    selector = SHAPFeatureSelector(n_top_features=20)
    selected = selector.fit(X, y)
    
    print(f"\nSelected {len(selected)} features:")
    for i, feat in enumerate(selected[:10]):
        print(f"  {i+1}. {feat}: {selector.feature_importance.get(feat, 0):.4f}")
