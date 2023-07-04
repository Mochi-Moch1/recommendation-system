#%%
import pandas as pd
import numpy as np
# %%
msg = "Hello World"
print(msg)

path = '/workspace/local/ml-10M100K/'

# Load movie data
movie_col = ['movie_id', 'title', 'genre']
movies = pd.read_csv(path+'movies.dat', names=movie_col, sep='::', encoding='latin-1', engine='python')

# Save genre as list
movies['genre'] = movies.genre.apply(lambda x:x.split('|'))
movies.head()

# %% Tag Information
# Load
t_cols = ['user_id', 'movie_id', 'tag', 'timestamp']
user_tagged_movies = pd.read_csv(path+'tags.dat', names=t_cols, sep='::', engine='python')

# Lower case
user_tagged_movies['tag'] = user_tagged_movies['tag'].str.lower()
user_tagged_movies.head() 

# Show data
print(f'タグ種類={len(user_tagged_movies.tag.unique())}')
print(f'タグレコード数={len(user_tagged_movies)}')
print(f'タグがついている映画={len(user_tagged_movies.movie_id.unique())}')

# Make tag list
movie_tags = user_tagged_movies.groupby('movie_id').agg({'tag': list})

# Concat tag info
movies = movies.merge(movie_tags, on='movie_id', how='left')
movies.head()

# %% Evaluate data
# Load
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(path+'ratings.dat', names=r_cols, sep='::', engine='python')

# Limit the number of users to 1000
valid_user_id = sorted(ratings.user_id.unique())[:1000]
ratings = ratings[ratings['user_id'].isin(valid_user_id)]

#%% Static info
movielens = ratings.merge(movies, on='movie_id')
print(movielens.head())

print(movielens.groupby('user_id').agg({'movie_id': [min, max, np.mean, len]}))

print(movielens.groupby('movie_id').agg({'user_id': len}).agg({'user_id': [min, max, np.mean, len]}))
# %% Rating
print(f'評価値数={len(movielens)}')
print(movielens.groupby('rating').agg({'movie_id': len})) 


# %% Evaluation
# Split the dataset
movielens['timestamp_rank'] = movielens.groupby('user_id')['timestamp'].rank(ascending=False, method='first')
movielens_train = movielens[movielens['timestamp_rank'] > 5]
movielens_test = movielens[movielens['timestamp_rank'] <= 5]
 
# %% RMSE
from typing import List, Dict
from sklearn.metrics import mean_squared_error
def calc_rmse(self, true_rating: List[float], pred_rating: List[float]) -> float:
    return np.sqrt(mean_squared_error(true_rating, pred_rating))

# Recall@K
def calc_recall_at_k(
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int
) -> float:
    scores = []
    for user_id in true_user2items.keys():
        r_at_k = _recall_at_k(true_user2items[user_id], pred_user2items[user_id], k)
        scores.append(r_at_k)
    return np.mean(scores)

def _recall_at_k(self, true_items: List[int], 
                 pred_items: List[int], k:int) -> float:
    if len(true_items) == 0 or k == 0:
        return 0.0
    r_at_k = (len(set(true_items) & set(pred_items[:k])) / len(true_items))
    return r_at_k

#Precision@K
def calc_precision_at_k(
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int
) -> float:
    scores = []
    for user_id in true_user2items.keys():
        p_at_k = _precision_at_k(true_user2items[user_id], pred_user2items[user_id], k)
        scores.append(p_at_k)
    return np.mean(scores)

def _precision_at_k(self, true_items: List[int], 
                 pred_items: List[int], k:int) -> float:
    if k == 0:
        return 0.0
    p_at_k = (len(set(true_items) & set(pred_items[:k])) / k)
    return p_at_k
    

