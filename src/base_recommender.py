from abc import ABC, abstractmethod
from util.models import Dataset, RecommendResult
from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator

class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        pass

    def run_sample(self) -> None:
        movielens = DataLoader(num_users=1000, num_test_items=5, data_path="/workspace/local/ml-10M100K/").load()
        # Recommendation result
        recomend_result = self.recommend(movielens)
        # Evaluation of Recommendation
        metrics = MetricCalculator().calc(
            movielens.test.rating.tolist(),
            recommend_result.rating.tolist(),
            movielens.test_user2items,
            recommend_result.user2items,
            k=10,
        )
        print(metrics)