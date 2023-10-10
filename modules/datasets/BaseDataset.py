from random import randint
from itertools import product
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class BaseDataset:
    def __init__(self):
        self.pos_pairs = None
        self.n_user = None
        self.n_item = None
        self.DIR_NAME = '../../../data/'

    def get_train_and_test_set(
            self,
            use_nagative_sampling: bool = False,
            use_popularity: bool = True,
            neg_pair_weight: int = 10,
            popular_threshold: int = 20
    ):
        """Convert explicit feedback data to implicit feedback data

        Args:
            strict_negative: ネガティブとしてサンプリングされるペアの中に本当はポジティブなものがあってもよいか
            neg_item_weight: ポジティブなペアのneg_item_weight倍の数のネガティブアイテムをサンプリングする
            tail_threshold (int): The threshold for popular items and tail items (%).
                                  If you don't have to use UnpopularEvaluator, give 0.
        Returns:
            train_set (Numpy array): user_id and positive item_id pair for training [n_train_samples, 2]
            test_set (Numpy array): user_id, item_id, rating(0, 1) for test (removed train pair) [n_test_samples, 3]
        """
        # train test split (75:25)
        train, test = train_test_split(self.pos_pairs)

        if use_nagative_sampling:
            pos_pairs_set = set(zip(self.pos_pairs['user_id'], self.pos_pairs['item_id']))
            n_negative_samples = len(test) * neg_pair_weight

            negative_pairs = []
            sample_cnt = 0
            while sample_cnt < n_negative_samples:
                sample_pair = (randint(0, self.n_user - 1), randint(0, self.n_item - 1))
                if sample_pair in pos_pairs_set:
                    continue
                df_negative_pairs = pd.DataFrame(negative_pairs, columns=['user_id', 'item_id'])
                df_negative_pairs['raiting'] = 0
                test = pd.concat([test, df_negative_pairs], ignore_index=True)

        else:
            # all user item pairs
            df_all = pd.DataFrame(
                [[u, i] for u, i in product(range(self.n_user), range(self.n_item))],
                columns=["user_id", "item_id"]
            )

            # join train feedback data
            df_all = pd.merge(
                df_all,
                train[["user_id", "item_id", "rating"]],
                on=["user_id", "item_id"],
                how="left"
            )

            # remove train pairs
            test = pd.merge(
                df_all[df_all.rating.isna()][["user_id", "item_id"]],
                test[["user_id", "item_id", "rating"]],
                on=["user_id", "item_id"],
                how="left"
            ).fillna(0)

        # Create a 'tail' column for test set. 0 if the item is in the top X%, 1 otherwise.
        if use_popularity:
            popularity_sorted = self.item_popularity_data().sort_values(by="feedback_num", ascending=False)

            top_20_percent_idx = int(popular_threshold * len(popularity_sorted) / 100)
            popular_items = set(popularity_sorted.iloc[:top_20_percent_idx]['item_id'])

            test['tail'] = np.where(test['item_id'].isin(popular_items), 0, 1)

        # numpy array
        train_set = train[["user_id", "item_id"]].values
        if use_popularity:
            test_set = test[["user_id", "item_id", "rating", "tail"]].values
        else:
            test_set = test[["user_id", "item_id", "rating"]].values

        return train_set, test_set

    def item_popularity_data(self):
        """
        各アイテムごとに、Implicit feedbackの観測数をカウントする
        """

        df_items = pd.DataFrame(
            [i for i in range(self.n_item)],
            columns=["item_id"]
        )

        # Filter rows from read_data where rating is 1
        filtered_data = self.pos_pairs[self.pos_pairs.rating == 1]

        # Count by item_id
        feedback_count = filtered_data.groupby('item_id').size().reset_index(name='feedback_num')

        # Merge feedback_count into df_items
        df_items = pd.merge(df_items, feedback_count, on='item_id', how='left')

        # Replace NaN with 0
        df_items['feedback_num'].fillna(0, inplace=True)
        df_items['feedback_num'] = df_items['feedback_num'].astype(int)

        return df_items

    def show_item_popularity_dist(self, tail_threshold: int = 20):
        """
        Plot the distribution of implicit feedback counts.
        """
        # Sort df_items by feedback_num in descending order
        sorted_df = self.item_popularity_data().sort_values(by='feedback_num', ascending=False)

        # Specify colors for items in the top X% and the rest
        top_20_percent = int(tail_threshold * len(sorted_df) / 100)
        colors = ['blue' if i < top_20_percent else 'red' for i in range(len(sorted_df))]

        # Plot the graph
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(sorted_df)), sorted_df['feedback_num'], color=colors)
        plt.xlabel('Items')
        plt.ylabel('Number of Feedbacks')
        plt.title('Feedback Counts per Item')
        plt.grid(axis='y')
        plt.xlim(0, len(sorted_df))
        plt.xticks([])
        plt.legend(handles=[bars[0], bars[-1]], labels=[f'Top {tail_threshold}%', f'Bottom {100-tail_threshold}%'])
        plt.show()

    # 以下使わないかも-----------------------------------------------------------------------------------------------------

    def item_feature_data_for_eval(self):
        """Make item feature matrix for evaluation of diversity

        Returns:
            item_feature_set (Numpy array):
                item_id, genre0,..., genre18 (Each genre is represented by 0 or 1.) [n_item, 20]
        """

        genre_col = (5, 24)

        # make header name
        names = ['item_id']
        with open(self.DIR_NAME + self.GENRE_FILE_NAME, 'r') as f:
            lines = f.readlines()
        for line in lines:
            genre = line.strip('\n').split('|')[0].strip()
            if genre:
                names.append(genre)

        # load data
        usecols = [0] + list(range(genre_col[0], genre_col[1]))
        df = pd.read_csv(
            self.DIR_NAME + self.ITEM_FILE_NAME,
            sep='|', header=None, index_col=None, usecols=usecols, names=names,
            encoding='latin1'
        )

        # numpy array
        item_feature_set = df.values

        return item_feature_set

    def interaction_data_for_eval(self):
        """Convert explicit feedback data to user item interaction data for evaluation of novelty

        interaction: Whether the user has ever rated the item.

        Returns:
            train_set (Numpy array): user_id and positive item_id pair for training [n_train_samples, 2]
        """

        filename = 'u.data'

        # load data
        read_data = pd.read_csv(
            self.dirname + filename,
            sep='\t', header=None, index_col=None,
            names=["user_id", "item_id", "rating", "timestamp"]
        )

        # set user/item ids
        read_data.user_id -= 1
        read_data.item_id -= 1

        # convert rating to interaction
        read_data.rating = (read_data.rating >= 1.0).astype(int)

        # # train test split (75:25)
        # train, test = train_test_split(read_data)

        # all user item pairs
        df_all = pd.DataFrame(
            [[u, i] for u, i in product(range(self.n_user), range(self.n_item))],
            columns=["user_id", "item_id"]
        )

        # join train feedback data
        df_all = pd.merge(
            df_all,
            read_data[["user_id", "item_id", "rating"]],
            on=["user_id", "item_id"],
            how="left"
        )

        # numpy array
        interaction_set = df_all[df_all.rating == 1][["user_id", "item_id"]].values

        return interaction_set
