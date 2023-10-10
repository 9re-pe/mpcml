import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score
from tqdm import tqdm

from ..models import BaseEmbeddingModel
from .BaseEvaluator import BaseEvaluator


class UnpopularityEvaluator(BaseEvaluator):
    """Class of evaluators computing unpopularity metrics for each user and calculating average."""

    def __init__(
        self, test_set: torch.Tensor, ks: list = [5]
    ):
        """Set test data and metrics.

        Args:
            test_set (torch.Tensor): test data which column is [user_id, item_id, rating].
            ks (int, optional): A list of @k. Defaults to [5].

        """
        super().__init__(test_set)

        self.TAIL_COLUMN_NUM = 3
        self.ks = ks

        self.metrics_names = [
            f"Unpopularity@{k}" for k in ks
        ]

    def compute_score(
        self, y_test_user: np.ndarray, y_hat_user: np.ndarray
    ) -> pd.DataFrame:
        """Method of computing score for metric.
        This method make a row of DataFrame which has scores for each metrics and k for the user.

        Args:
            y_test_user (np.ndarray): grand truth for the user
            y_hat_user (np.ndarray): prediction of relevance

        Returns:
            (pd.DataFrame): a row of DataFrame which has scores for each metrics and k for the user.
        """

        if y_test_user.sum() == 0:
            return pd.DataFrame({name: [0] for name in self.metrics_names})

        else:
            df_eval_sub = pd.DataFrame(
                {
                    f"Unpopularity@{k}": [unpopularity_recall(y_test_user, y_hat_user, k)] for k in self.ks
                }
            )

        return df_eval_sub

    def eval_user(self, model: BaseEmbeddingModel, uid: int) -> pd.DataFrame:
        """Method of evaluating for given user.

        Args:
            model (BaseEmbeddingModel): models which have user and item embeddings.
            uid (int): user id

        Returns:
            (pd.DataFrame): a row of DataFrame which has scores for each metrics and k for the user.
        """
        user_indices = self.test_set[:, 0] == uid
        test_set_pair = self.test_set[user_indices, :2]

        # distance for each user and item pair size (n_pairs)
        y_hat_user = model.predict(test_set_pair).to("cpu").detach().numpy()
        y_test_user = self.test_set[user_indices, self.TAIL_COLUMN_NUM].to("cpu").detach().numpy()

        return self.compute_score(y_test_user, y_hat_user)

    def score(
        self, model: BaseEmbeddingModel, reduction="mean", no_progressbar=False
    ) -> pd.DataFrame:
        """Method of calculating average score for all users.

        Args:
            model (BaseEmbeddingModel): models which have user and item embeddings.
            reduction (str, optional): reduction method. Defaults to "mean".
            no_progressbar (bool, optional): displaying progress bar or not during evaluating. Defaults to False.

        Returns:
            pd.DataFrame: a row of DataFrame which has average scores
        """

        users = torch.unique(self.test_set[:, 0])
        df_eval = pd.DataFrame({name: [] for name in self.metrics_names})

        for uid in tqdm(users, disable=no_progressbar):
            df_eval_sub = self.eval_user(model, uid)
            df_eval = pd.concat([df_eval, df_eval_sub])

        if reduction == "mean":
            score = pd.DataFrame(df_eval.mean(axis=0)).T

        else:
            score = df_eval.copy()

        return score


def unpopularity_recall(y_test_user: np.ndarray, y_hat_user: np.ndarray, k: int):
    """Function for user-wise evaluators calculating Unpopularity@k

    Args:
        y_test_user (np.ndarray): grand truth for the user
        y_hat_user (np.ndarray): prediction of relevance
        k (int): a number of top item considered.

    Returns:
        (float): recall score
    """
    pred_rank = (-y_hat_user).argsort().argsort() + 1
    pred_topk_flag = (pred_rank <= k).astype(int)
    return recall_score(y_test_user, pred_topk_flag)
