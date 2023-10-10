import pandas as pd

from .BaseDataset import BaseDataset


class Yelp(BaseDataset):
    def __init__(self):
        super().__init__()
        self.DIR_NAME = self.DIR_NAME + 'yelp_dataset/'
        self.RATING_FILE_NAME = 'yelp_academic_dataset_review.json'

        # load data
        df_rating = pd.read_json(self.DIR_NAME + self.RATING_FILE_NAME, lines=True)
        df_rating = df_rating[["user_id", "business_id", "stars", "date"]]
        df_rating = df_rating.rename(columns={
            "business_id": "item_id",
            "stars": "rating",
            "date": "timestamp"
        })

        # encoding user_id and item_id to integer (0-index)
        df_rating["user_id"] = df_rating["user_id"].astype('category').cat.codes
        df_rating["item_id"] = df_rating["item_id"].astype('category').cat.codes

        # convert rating to implicit feedback
        df_rating["rating"] = (df_rating["rating"] >= 4.0).astype(int)
        pos_pairs = df_rating[df_rating["rating"] == 1].copy()

        # ignore users or items that have no implicit feedback
        pos_pairs['user_id'] = pd.factorize(pos_pairs['user_id'])[0]
        pos_pairs['item_id'] = pd.factorize(pos_pairs['item_id'])[0]

        self.pos_pairs = pos_pairs
        self.n_user = pos_pairs.user_id.nunique()
        self.n_item = pos_pairs.item_id.nunique()