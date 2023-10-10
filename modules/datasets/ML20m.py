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

        pos_pairs = super().preprocess(df_rating)
        super().set_values(pos_pairs)
