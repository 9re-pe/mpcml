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

        pos_pairs = super().preprocess(df_rating)
        super().set_values(pos_pairs)
