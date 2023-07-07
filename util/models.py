import dataclasses
import pandas as pd
from typing import Dict, List

@dataclasses.dataclass(frozen=True)
# Dataset used to train and evaluate a recommendation system
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    # ranking
    test_user2items: Dict[int, List[int]]
    item_content: pd.DataFrame

@dataclasses.dataclass(frozen=True)
# Prediction result of recommendation system
class RecommendResult:
    rating: pd.DataFrame
    user2items: Dict[int, List[int]]

@dataclasses.dataclass(frozen=True)
# Evaluation of recommendation system
class Metrics:
    rmse: float
    precision_at_k: float
    recall_at_k: float

    def __repr__(self):
        return f"rmse={self.rmse: .3f}, Precision@K={self.precision_at_k:.3f}, Recall@K={self.recall_at_k:.3f}"