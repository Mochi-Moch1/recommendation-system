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
movielens.head()

# %%
print(movielens.groupby('user_id').agg({'movie_id': [min, max, np.mean, len]}))
# %%
print(movielens.groupby('movie_id').agg({'user_id': len}).agg({'user_id': [min, max, np.mean, len]}))
# %% Rating
print(f'評価値数={len(movielens)}')
print(movielens.groupby('rating').agg({'movie_id': len})) 

# %%
