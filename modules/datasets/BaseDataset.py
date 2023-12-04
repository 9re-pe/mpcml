import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


random.seed(42)


class BaseDataset:
    def __init__(self):
        self.pos_pairs = None
        self.n_user = None
        self.n_item = None
        # path from mpcml/experiment/notebook/
        self.DIR_NAME = '../../../data/'

    @staticmethod
    def preprocess(df_rating):
        """Preprocessing for explicit feedback data

        Args:
            df_rating (Pandas DataFrame): explicit feedback data
        """

        # convert rating to implicit feedback
        df_rating["rating"] = (df_rating["rating"] >= 4.0).astype(int)
        pos_pairs = df_rating[df_rating["rating"] == 1].copy()

        # Users who have less than 4 ratings are only included in the train/test set.
        user_feedback_cnt = pos_pairs['user_id'].value_counts()
        filter_idx = user_feedback_cnt[user_feedback_cnt >= 4].index
        pos_pairs = pos_pairs[pos_pairs['user_id'].isin(filter_idx)]

        # ignore users or items that have no implicit feedback
        pos_pairs['user_id'] = pd.factorize(pos_pairs['user_id'])[0]
        pos_pairs['item_id'] = pd.factorize(pos_pairs['item_id'])[0]

        return pos_pairs

    def set_values(self, pos_pairs):
        """set instance variables

        Args:
            pos_pairs (Pandas DataFrame): preprocessed implicit feedback data
        """

        self.pos_pairs = pos_pairs
        self.n_user = pos_pairs.user_id.nunique()
        self.n_item = pos_pairs.item_id.nunique()

    def get_train_and_test_set(
            self,
            neg_pair_weight: int = 10,
            popular_threshold: int = 20
    ):
        """Convert explicit feedback data to implicit feedback data

        Args:
            neg_pair_weight: Samples X times more negative pairs than positive pairs per user for test data
            popular_threshold: Labels items below the top X% as 'tail' items.
        Returns:
            train_set (Numpy array): user_id and positive item_id pair for training [n_train_samples, 2]
            test_set (Numpy array): user_id, item_id, rating, tail for test [n_test_samples, 4]
        """

        df_group_by_user = self.pos_pairs.groupby('user_id')
        li_df_train = []
        li_df_test = []
        pos_pairs_set = set(zip(self.pos_pairs['user_id'], self.pos_pairs['item_id']))

        for user_id, df_group in df_group_by_user:
            # train test split (75:25)
            df_train, df_test = train_test_split(df_group, random_state=42)
            li_df_train.append(df_train)
            li_df_test.append(df_test)

            # negative sampling for test set
            n_negative_samples = len(df_test) * neg_pair_weight
            negative_pairs = []
            sample_cnt = 0
            while sample_cnt < n_negative_samples:
                sampled_pair = (user_id, random.randint(0, self.n_item - 1))
                if sampled_pair in pos_pairs_set:
                    continue
                negative_pairs.append(sampled_pair)
                sample_cnt += 1
            df_negative_pairs = pd.DataFrame(negative_pairs, columns=['user_id', 'item_id'])
            df_negative_pairs['rating'] = 0
            li_df_test.append(df_negative_pairs)

        train = pd.concat(li_df_train)
        test = pd.concat(li_df_test)

        # Compute the popularity
        item_popularity = self.item_popularity_data()
        item_popularity["popularity"] = item_popularity["feedback_num"] / self.n_item
        test = pd.merge(test, item_popularity[["item_id", "popularity"]], on="item_id", how="left")

        # add a 'tail' column to test set. 0 if the item is in the top X%, 1 otherwise.
        popularity_sorted = self.item_popularity_data().sort_values(by="feedback_num", ascending=False)
        top_item_idx = int(popular_threshold * len(popularity_sorted) / 100)
        popular_items = set(popularity_sorted.iloc[:top_item_idx]['item_id'])
        test['tail'] = np.where(test['item_id'].isin(popular_items), 0, 1)

        # numpy array
        train_set = train[["user_id", "item_id"]].values
        test_set = test[["user_id", "item_id", "rating", "popularity", "tail"]].values

        return train_set, test_set

    def item_popularity_data(self):
        """Count the number of implicit feedbacks for each item."""

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
        """Plot the distribution of implicit feedback counts."""

        # Sort df_items by feedback_num in descending order
        sorted_df = self.item_popularity_data().sort_values(by='feedback_num', ascending=False)

        # Number of items in the top X%
        n_head_items = int(tail_threshold * len(sorted_df) / 100)

        # Get default colors from matplotlib palette
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        head_color = default_colors[0]
        tail_color = default_colors[1]

        # Specify colors for items in the top X% and the rest
        colors = [head_color if i < n_head_items else tail_color for i in range(len(sorted_df))]

        # Plot the graph
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(sorted_df)), sorted_df['feedback_num'], color=colors)
        plt.xlabel('Items')
        plt.ylabel('Number of Feedbacks')
        plt.grid(axis='y')
        plt.xlim(0, len(sorted_df))
        plt.xticks([])

        # Creating custom handles for the legend
        head_handle = plt.Rectangle((0, 0), 1, 1, color=head_color)
        tail_handle = plt.Rectangle((0, 0), 1, 1, color=tail_color)
        plt.legend(handles=[head_handle, tail_handle],
                   labels=[f'Head ({tail_threshold}%)', f'Tail ({100 - tail_threshold}%)'])

        plt.show()
