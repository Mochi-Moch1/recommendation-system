import numpy as np
from sklearn.metrics import mean_squared_error
from util.models import Metrics
from typing import Dict, List


class MetricCalculator:
    def calc(
            self,
            true_rating: List[float],
            pred_rating: List[float],
            true_user2items: Dict[int, List[int]],
            pred_user2items: Dict[int, List[int]],
            k: int,
    ) -> Metrics:
        rmse = self._calc_rmse(true_rating, pred_rating)
        precision_at_k = self._calc_precision_at_k(true_user2items, pred_user2items, k)
        recall_at_k = self._calc_recall_at_k(true_user2items, pred_user2items, k)
        return Metrics(rmse, precision_at_k, recall_at_k)
    
    def 