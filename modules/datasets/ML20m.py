import pandas as pd

from .BaseDataset import BaseDataset


class ML20m(BaseDataset):
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

        # convert rating to implicit feedback
        df_rating["rating"] = (df_rating["rating"] >= 4.0).astype(int)
        pos_pairs = df_rating[df_rating["rating"] == 1].copy()

        # ignore users or items that have no implicit feedback
        pos_pairs['user_id'] = pd.factorize(pos_pairs['user_id'])[0]
        pos_pairs['item_id'] = pd.factorize(pos_pairs['item_id'])[0]

        self.pos_pairs = pos_pairs
        self.n_user = pos_pairs.user_id.nunique()
        self.n_item = pos_pairs.item_id.nunique()
