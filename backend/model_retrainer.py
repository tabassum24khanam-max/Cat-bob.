"""
Tier 2.2: Automatic Model Retraining
Markets shift. A model trained in January is stale by March.
Retrains weekly on rolling 90-day window. Old models decay, new ones adapt.
"""
import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class RetrainEvent:
    ts: int
    asset: str
    n_samples: int
    old_accuracy: float
    new_accuracy: float
    improved: bool


class ModelRetrainer:
    def __init__(self):
        self.retrain_interval = 7 * 24 * 3600  # 7 days
        self.min_samples = 30
        self.rolling_window_days = 90
        self.last_retrain: Dict[str, int] = {}
        self.history: list = []
        self._running = False

    def needs_retrain(self, asset: str) -> bool:
        """Check if this asset's model is stale."""
        last = self.last_retrain.get(asset, 0)
        return (time.time() - last) > self.retrain_interval

    async def retrain_asset(self, asset: str, get_predictions_fn, train_fn, ml_cache: dict) -> dict:
        """Retrain a single asset's ML model on recent data."""
        from database import get_predictions
        from ml_engine import train_ensemble

        predictions = await get_predictions(asset, 500)

        # Filter to rolling window
        cutoff = time.time() - (self.rolling_window_days * 24 * 3600)
        recent = [p for p in predictions if p.get('ts', 0) > cutoff]
        rated = [p for p in recent if p.get('feedback') in ('correct', 'wrong')]

        if len(rated) < self.min_samples:
            return {
                'retrained': False,
                'reason': f'Not enough rated predictions ({len(rated)}/{self.min_samples})',
            }

        # Get old accuracy
        old_model = ml_cache.get(asset)
        old_accuracy = old_model.get('train_accuracy', 0) if old_model else 0

        # Train new model
        new_model = await train_fn(rated)
        if not new_model:
            return {'retrained': False, 'reason': 'Training failed'}

        new_accuracy = new_model.get('train_accuracy', 0)

        # Only replace if new model is better or old is very stale (>14 days)
        days_since = (time.time() - self.last_retrain.get(asset, 0)) / 86400
        improved = new_accuracy >= old_accuracy or days_since > 14

        if improved:
            ml_cache[asset] = new_model
            self.last_retrain[asset] = int(time.time())

        event = RetrainEvent(
            ts=int(time.time()),
            asset=asset,
            n_samples=len(rated),
            old_accuracy=old_accuracy,
            new_accuracy=new_accuracy,
            improved=improved,
        )
        self.history.append(event)
        self.history = self.history[-50:]

        return {
            'retrained': improved,
            'asset': asset,
            'n_samples': len(rated),
            'old_accuracy': round(old_accuracy * 100, 1),
            'new_accuracy': round(new_accuracy * 100, 1),
            'improved': improved,
        }

    async def retrain_all_stale(self, assets: list, ml_cache: dict) -> list:
        """Check all assets, retrain any that are stale."""
        from database import get_predictions
        from ml_engine import train_ensemble

        results = []
        for asset in assets:
            if self.needs_retrain(asset):
                result = await self.retrain_asset(
                    asset, get_predictions, train_ensemble, ml_cache
                )
                results.append(result)
                await asyncio.sleep(1)  # don't hammer
        return results

    def get_status(self) -> dict:
        return {
            'retrain_interval_days': self.retrain_interval / 86400,
            'rolling_window_days': self.rolling_window_days,
            'min_samples': self.min_samples,
            'assets_tracked': len(self.last_retrain),
            'last_retrains': {k: v for k, v in self.last_retrain.items()},
            'recent_events': [
                {
                    'ts': e.ts, 'asset': e.asset,
                    'samples': e.n_samples, 'improved': e.improved,
                    'accuracy': f"{e.old_accuracy*100:.0f}%→{e.new_accuracy*100:.0f}%",
                }
                for e in self.history[-10:]
            ],
        }


# Singleton
_retrainer = ModelRetrainer()

def get_retrainer() -> ModelRetrainer:
    return _retrainer
