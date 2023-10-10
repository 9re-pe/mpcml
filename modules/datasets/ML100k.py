import pandas as pd

from .BaseDataset import BaseDataset


class ML100k(BaseDataset):
    def __init__(self):
        super().__init__()
        self.DIR_NAME = self.DIR_NAME + 'ml-100k/'
        self.RATING_FILE_NAME = 'u.data'
        self.GENRE_FILE_NAME = 'u.genre'
        self.ITEM_FILE_NAME = 'u.item'

        # load data
        df_rating = pd.read_csv(
            self.DIR_NAME + self.RATING_FILE_NAME,
            sep='\t', header=None, index_col=None,
            names=["user_id", "item_id", "rating", "timestamp"]
        )

        pos_pairs = super().preprocess(df_rating)
        super().set_values(pos_pairs)
