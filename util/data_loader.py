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
            movielens_test[movielens_test >= 4].groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        ) 
        return Dataset(movielens_train, movielens_test, movielens_test_user2items, movie_content)
    
    def _split_data(self, movielens:pd.DataFrame) -> (pd.DataFrame, pd.DataFrame): 
        # Sort by latest first
        movielens["rating_order"] = movielens.groupby("user_id")["timestamp"].rank(ascending=False, method="first")

        # The last n movies are for testing, the others are for training.
        movielens_train = movielens[movielens["rating_order"] > self.num_test_items]
        movielens_test = movielens[movielens["rating_order"] <= self.num_test_items]
        
        return movielens_train, movielens_test

    def _load(self)