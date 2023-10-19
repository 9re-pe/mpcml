import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import recall_score
from typing import List

from ..searches import BaseSearch
from .BaseEvaluator import BaseEvaluator


class UserwiseEvaluator(BaseEvaluator):
    """Class of evaluators computing accuracy metrics for each user and calculating average."""

    def __init__(
            self,
            test_set: torch.Tensor,
            score_function_dict: dict,
            ks: list = [5]
    ):
        """Set test data and metrics.

        Args:
            test_set (torch.Tensor): test data which column is [user_id, item_id, rating].
            score_function_dict (dict): dictionary whose keys are metrics name and values are user-wise function.
            ks (int, optional): A list of @k. Defaults to [5].

        for example, score_function_dict is

        score_function_dict = {
            "Recall"      : evaluators.recall,
            "Unpopularity": evaluators.unpopularity,
            "F1-score"    : evaluators.f1_score
        }

        arguments of each function must be
            y_test_user (List[np.ndarray]): grand truth for the user and popularity for the item
            y_hat_user (np.ndarray) : prediction of relevance
            k : a number of top item considered.
        """

        super().__init__(test_set)

        self.score_function_dict = score_function_dict
        self.ks = ks

        self.metrics_names = [
            f"{name}@{k}" for k in ks for name in score_function_dict.keys()
        ]

    def compute_score(
            self, y_test_user: List[np.ndarray], y_hat_user: np.ndarray
    ) -> pd.DataFrame:

        df_eval_sub = pd.DataFrame(
            {
                f"{name}@{k}": [metric(y_test_user, y_hat_user, k)]
                for k in self.ks
                for name, metric in self.score_function_dict.items()
            }
        )

        return df_eval_sub

    def eval_user(self, search: BaseSearch, uid: int) -> pd.DataFrame:
        """Method of evaluating for given user.
        This method make a row of DataFrame which has scores for each metrics and k for the user.

        Args:
            search (BaseSearch):
            uid (int): user id

        Returns:
            (pd.DataFrame): a row of DataFrame which has scores for each metrics and k for the user.
        """

        user_indices = self.test_set[:, 0] == uid
        test_set_pair = self.test_set[user_indices, :2].long()

        # distance for each user and item pair size (n_pairs)
        y_hat_user = search.predict(test_set_pair).to("cpu").detach().numpy()
        truth = self.test_set[user_indices, 2].to("cpu").detach().numpy()
        popularity = self.test_set[user_indices, 3].to("cpu").detach().numpy()
        tail = self.test_set[user_indices, 4].to("cpu").detach().numpy()
        y_test_user = [truth, popularity, tail]

        return self.compute_score(y_test_user, y_hat_user)

    def score(
            self, search: BaseSearch, reduction="mean", no_progressbar=False
    ) -> pd.DataFrame:
        """Method of calculating average score for all users.

        Args:
            search (BaseSearch):
            reduction (str, optional): reduction method. Defaults to "mean".
            no_progressbar (bool, optional): displaying progress bar or not during evaluating. Defaults to False.

        Returns:
            pd.DataFrame: a row of DataFrame which has average scores
        """

        users = torch.unique(self.test_set[:, 0])
        df_eval = pd.DataFrame({name: [] for name in self.metrics_names})

        for uid in tqdm(users, disable=no_progressbar):
            df_eval_sub = self.eval_user(search, uid)
            df_eval = pd.concat([df_eval, df_eval_sub])

        if reduction == "mean":
            score = pd.DataFrame(df_eval.mean(axis=0)).T

        else:
            score = df_eval.copy()

        return score


def recall(y_test_user: List[np.ndarray], y_hat_user: np.ndarray, k: int):
    """Function for user-wise evaluators calculating Recall@k

    Args:
        y_test_user (np.ndarray): grand truth for the user
        y_hat_user (np.ndarray): prediction of relevance
        k (int): a number of top item considered.

    Returns:
        (float): recall score
    """
    truth = y_test_user[0]
    if truth.sum() == 0:
        return 0

    pred_rank = (-y_hat_user).argsort().argsort() + 1
    pred_topk_flag = (pred_rank <= k).astype(int)

    return recall_score(truth, pred_topk_flag)


def unpopularity(y_test_user: List[np.ndarray], y_hat_user: np.ndarray, k: int):
    """Function for user-wise evaluators calculating Unpopularity@k

    Args:
        y_test_user (np.ndarray): popularity score for the item
        y_hat_user (np.ndarray): prediction of relevance
        k (int): a number of top item considered.

    Returns:
        (float): recall score
    """

    popularity = y_test_user[1]
    pred_rank = (-y_hat_user).argsort().argsort() + 1
    pred_topk_flag = (pred_rank <= k).astype(int)
    unpopularity_score = np.sum(pred_topk_flag * (1 - popularity)) / k

    return unpopularity_score


def unpopularity2(y_test_user: List[np.ndarray], y_hat_user: np.ndarray, k: int):
    """Function for user-wise evaluators calculating Unpopularity@k

    Args:
        y_test_user (np.ndarray): popularity score for the item
        y_hat_user (np.ndarray): prediction of relevance
        k (int): a number of top item considered.

    Returns:
        (float): recall score
    """

    popularity = y_test_user[1]
    pred_rank = (-y_hat_user).argsort().argsort() + 1
    pred_topk_flag = (pred_rank <= k).astype(int)
    unpopularity_score = np.sum(pred_topk_flag * (-np.log2(popularity))) / k

    return unpopularity_score


def unpopularity3(y_test_user: List[np.ndarray], y_hat_user: np.ndarray, k: int):
    tail = y_test_user[2]
    if tail.sum() == 0:
        return 0

    pred_rank = (-y_hat_user).argsort().argsort() + 1
    pred_topk_flag = (pred_rank <= k).astype(int)

    return recall_score(tail, pred_topk_flag)


def f1_score(y_test_user: List[np.ndarray], y_hat_user, k):
    recall_val = recall(y_test_user, y_hat_user, k)
    unpopularity_val = unpopularity(y_test_user, y_hat_user, k)
    if recall_val + unpopularity_val == 0.0:
        return 0.0
    else:
        return (2.0 * recall_val * unpopularity_val) / (recall_val + unpopularity_val)


def f1_score2(y_test_user: List[np.ndarray], y_hat_user, k):
    recall_val = recall(y_test_user, y_hat_user, k)
    unpopularity_val = unpopularity2(y_test_user, y_hat_user, k)
    if recall_val + unpopularity_val == 0.0:
        return 0.0
    else:
        return (2.0 * recall_val * unpopularity_val) / (recall_val + unpopularity_val)


def f1_score3(y_test_user: List[np.ndarray], y_hat_user, k):
    recall_val = recall(y_test_user, y_hat_user, k)
    unpopularity_val = unpopularity3(y_test_user, y_hat_user, k)
    if recall_val + unpopularity_val == 0.0:
        return 0.0
    else:
        return (2.0 * recall_val * unpopularity_val) / (recall_val + unpopularity_val)
