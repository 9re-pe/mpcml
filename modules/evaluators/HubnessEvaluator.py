from collections import defaultdict
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy.stats import skew
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..models import BaseEmbeddingModel
from .BaseEvaluator import BaseEvaluator


class HubnessEvaluator(BaseEvaluator):
    """Class of evaluators computing k-occurrence and hubness"""

    def __init__(
            self, test_set: torch.Tensor, ks: list = [5]
    ):
        """Set test data and metrics.

        Args:
            test_set (torch.Tensor): test data which column is [user_id, item_id, rating].
            ks (int, optional): A list of @k. Defaults to [5].

        arguments of each function must be
            y_hat_user (np.ndarray) : prediction of relevance
            k : a number of top item considered.
        """
        super().__init__(test_set)

        self.ks = ks

    def get_topk_items(self, model: BaseEmbeddingModel, uid: int, k: int):
        user_indices = self.test_set[:, 0] == uid
        test_set_pair = self.test_set[user_indices, :2]
        # Get the predicted distances of the target user and items
        y_hat_user = model.predict(test_set_pair).to("cpu").detach().numpy()
        # Sort and get the indices of the top k items
        topk_indices = (-y_hat_user).argsort()[:k]
        # Get the items corresponding to the indices and convert to a set
        items = self.test_set[topk_indices, 1].tolist()

        return items

    def get_k_occurrence(self, model: BaseEmbeddingModel, users, k, no_progressbar=False):
        k_occurrence = defaultdict(int)
        for uid in tqdm(users, disable=no_progressbar):
            items = self.get_topk_items(model, uid, k)
            for item_id in items:
                k_occurrence[item_id] += 1

        return k_occurrence

    def score(
            self, model: BaseEmbeddingModel, reduction="mean", no_progressbar=False
    ) -> pd.DataFrame:
        """Method of computing hubness for recommendations across all users.

        Args:
            model (BaseEmbeddingModel): models which have user and item embeddings.
            reduction (str, optional): reduction method. Defaults to "mean".
            no_progressbar (bool, optional): displaying progress bar or not during evaluating. Defaults to False.

        Returns:
            pd.DataFrame: a row of DataFrame which has average scores
        """

        score_dict = {}

        users = torch.unique(self.test_set[:, 0])

        for k in self.ks:
            eval_name = f"hubness@{k}"

            k_occurrence = self.get_k_occurrence(model, users, k)
            hubness = skew(list(k_occurrence.values()))

            score_dict[eval_name] = [hubness]

        # convert to a dataframe
        score = pd.DataFrame(score_dict)

        return score

    def show_k_occurrence(self, model: BaseEmbeddingModel, k: int):
        users = torch.unique(self.test_set[:, 0])
        k_occurrence = self.get_k_occurrence(model, users, k, no_progressbar=True)

        # make distribution
        distribution = Counter(k_occurrence.values())
        x = list(distribution.keys())
        y = list(distribution.values())

        # plot a graph
        plt.bar(x, y, width=1)
        plt.xlabel('k-occurrence')
        plt.ylabel('number of items')
        plt.title('K-occurrence Distribution')
        plt.show()
