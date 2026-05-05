"""
Tier 2.1: Feature Importance & Pruning
Uses permutation importance (SHAP-lite) to find which features actually matter.
Dead features dilute signal. This kills them.
"""
import time
import numpy as np
from typing import Dict, List, Optional


class FeaturePruner:
    def __init__(self):
        self.importance_scores: Dict[str, float] = {}
        self.dead_features: List[str] = []
        self.top_features: List[str] = []
        self.last_analysis_ts = 0
        self._history: List[dict] = []

    def analyze_importance(self, model_artifact: dict, feature_names: List[str]) -> dict:
        """
        Extract feature importance from trained ML ensemble.
        Uses model's built-in feature_importances_ (Gini/gain-based).
        """
        if not model_artifact or not feature_names:
            return {'available': False, 'reason': 'no model or features'}

        models = model_artifact.get('models', [])
        if not models:
            return {'available': False, 'reason': 'no trained models'}

        # Average importance across all models in ensemble
        all_importances = []
        for m in models:
            if hasattr(m, 'feature_importances_'):
                imp = m.feature_importances_
                if len(imp) == len(feature_names):
                    all_importances.append(imp)

        if not all_importances:
            return {'available': False, 'reason': 'models lack feature_importances_'}

        avg_importance = np.mean(all_importances, axis=0)
        total = avg_importance.sum()
        if total > 0:
            normalized = avg_importance / total
        else:
            normalized = avg_importance

        # Build ranked list
        ranked = sorted(zip(feature_names, normalized), key=lambda x: x[1], reverse=True)

        self.importance_scores = {name: float(score) for name, score in ranked}
        self.top_features = [name for name, score in ranked if score > 0.02]
        self.dead_features = [name for name, score in ranked if score < 0.005]
        self.last_analysis_ts = int(time.time())

        self._history.append({
            'ts': self.last_analysis_ts,
            'top_5': [name for name, _ in ranked[:5]],
            'dead_count': len(self.dead_features),
        })
        self._history = self._history[-20:]

        return {
            'available': True,
            'ranked': [(name, round(float(score) * 100, 2)) for name, score in ranked],
            'top_features': self.top_features,
            'dead_features': self.dead_features,
            'recommendation': self._recommend(ranked),
        }

    def permutation_importance(self, model_artifact: dict, X: np.ndarray,
                                y: np.ndarray, feature_names: List[str],
                                n_repeats: int = 5) -> dict:
        """
        Permutation importance: shuffle each feature, measure accuracy drop.
        More reliable than Gini importance (doesn't favor high-cardinality).
        """
        if not model_artifact or X is None or y is None:
            return {'available': False}

        models = model_artifact.get('models', [])
        if not models:
            return {'available': False}

        base_accuracy = self._ensemble_accuracy(models, X, y)
        importances = {}

        for i, fname in enumerate(feature_names):
            drops = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                np.random.shuffle(X_perm[:, i])
                perm_accuracy = self._ensemble_accuracy(models, X_perm, y)
                drops.append(base_accuracy - perm_accuracy)
            importances[fname] = float(np.mean(drops))

        ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        self.importance_scores = dict(ranked)
        self.top_features = [name for name, drop in ranked if drop > 0.01]
        self.dead_features = [name for name, drop in ranked if drop < 0.001]

        return {
            'available': True,
            'base_accuracy': round(base_accuracy, 4),
            'ranked': [(name, round(drop, 4)) for name, drop in ranked],
            'top_features': self.top_features,
            'dead_features': self.dead_features,
        }

    def get_feature_mask(self, feature_names: List[str]) -> List[bool]:
        """Returns mask of which features to KEEP (True) vs DROP (False)."""
        if not self.dead_features:
            return [True] * len(feature_names)
        return [name not in self.dead_features for name in feature_names]

    def get_report(self) -> dict:
        return {
            'available': bool(self.importance_scores),
            'top_features': self.top_features[:10],
            'dead_features': self.dead_features,
            'total_analyzed': len(self.importance_scores),
            'last_analysis': self.last_analysis_ts,
            'history': self._history[-5:],
        }

    def _ensemble_accuracy(self, models: list, X: np.ndarray, y: np.ndarray) -> float:
        preds = []
        for m in models:
            try:
                preds.append(m.predict(X))
            except Exception:
                continue
        if not preds:
            return 0.0
        ensemble_pred = np.round(np.mean(preds, axis=0))
        return float(np.mean(ensemble_pred == y))

    def _recommend(self, ranked: list) -> str:
        n_dead = len(self.dead_features)
        if n_dead == 0:
            return "All features contributing. No pruning needed."
        elif n_dead <= 3:
            return f"Consider removing {n_dead} dead features: {', '.join(self.dead_features)}"
        else:
            return f"PRUNE: {n_dead} dead features detected. Removing them should reduce noise."


# Singleton
_pruner = FeaturePruner()

def get_feature_pruner() -> FeaturePruner:
    return _pruner
