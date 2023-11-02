import pandas as pd
import numpy as np

from .BaseDataset import BaseDataset


class ML200mTo100kByUser(BaseDataset):
    def __init__(self):
        super().__init__()
        self.DIR_NAME = self.DIR_NAME + 'ml-20m/'
        self.RATING_FILE_NAME = 'ratings.csv'

        # load data
        df_rating = pd.read_csv(
            self.DIR_NAME + self.RATING_FILE_NAME,
            sep=',', header=0, index_col=None,
            names=["user_id", "item_id", "rating", "timestamp"]
        )

        # 1~137000の範囲でランダムに1000ユーザを抽出
        selected_users = np.random.choice(range(1, 137001), size=1000, replace=False)
        # selected_usersに含まれるuser_idの行だけを抽出
        selected_ratings = df_rating[df_rating['user_id'].isin(selected_users)]

        pos_pairs = super().preprocess(selected_ratings)
        super().set_values(pos_pairs)
