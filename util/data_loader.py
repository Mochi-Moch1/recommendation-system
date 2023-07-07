import pandas as pd
import os
from util.models import Dataset


class DataLoader:
    def __init__(self, num_users: int = 1000, num_test_items: int = 5,
                 data_path: str = "/workspace/local/ml-10M100K/"):
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path

    def load(self) -> Dataset:
        ratings, movie_content = self._load()
        movielens_train, movielens_test = self._split_data(ratings)

        # Only movies with rating of 4 or higher are considered answer.
        # key is user id. value is an item id highly rated by user.
        movielens_test_user2items = (
            movielens_test[movielens_test.rating >= 4].groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        ) 
        return Dataset(movielens_train, movielens_test, movielens_test_user2items, movie_content)
    
    def _split_data(self, movielens:pd.DataFrame) -> (pd.DataFrame, pd.DataFrame): 
        # Sort by latest first
        movielens["rating_order"] = movielens.groupby("user_id")["timestamp"].rank(ascending=False, method="first")

        # The last n movies are for testing, the others are for training.
        movielens_train = movielens[movielens["rating_order"] > self.num_test_items]
        movielens_test = movielens[movielens["rating_order"] <= self.num_test_items]
        
        return movielens_train, movielens_test

    def _load(self) -> (pd.DataFrame, pd.DataFrame):
        # Load the movie information
        m_cols = ["movie_id", "title", "genre"]
        movies = pd.read_csv(os.path.join(self.data_path, "movies.dat"), names=m_cols, sep="::", engine="python")
        # to list
        movies["genre"] = movies.genre.apply(lambda x: list(x.split("|")))

        # Load movie tag
        t_cols = ["user_id", "movie_id", "tag", "timestamp"]
        user_tagged_movies = pd.read_csv(os.path.join(self.data_path, "tags.dat"), names=t_cols, sep="::", engine="python")
        # lowerscale tag
        user_tagged_movies["tag"] = user_tagged_movies["tag"].str.lower()
        movie_tags = user_tagged_movies.groupby("movie_id").agg({"tag": list})
        # Integrate tag
        movies = movies.merge(movie_tags, on="movie_id", how="left")
        
        # Load rating data
        r_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(os.path.join(self.data_path, "ratings.dat"), names=r_cols, sep="::", engine="python")
        # Reduce user number
        valid_user_ids = sorted(ratings.user_id.unique())[:self.num_users]
        ratings = ratings[ratings.user_id <= max(valid_user_ids)]
        # Integrage data
        movielens_ratings = ratings.merge(movies, on="movie_id")
        return movielens_ratings, movies
    