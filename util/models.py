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