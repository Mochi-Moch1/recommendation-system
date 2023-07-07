#%%
!cd /workspace/recommendation-system
import sys; sys.path.insert(0, '..')

from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator

#%% Load movie lens dataset
data_loader = DataLoader(num_users=1000, num_test_items=5, data_path='/workspace/local/ml-10M100K/')
movielens = data_loader.load()


# %% Random recommend
from src.random_recommend import RandomRecommender
recommender = RandomRecommender()
recommend_result = recommender.recommend(movielens)
# %% Evaluate
metric_calculator = MetricCalculator()
metrics = metric_calculator.calc(
    movielens.test.rating.tolist(),
    recommend_result.rating.tolist(),
    movielens.test_user2items, 
    recommend_result.user2items,
    k=10
)
print(metrics)

# %%
