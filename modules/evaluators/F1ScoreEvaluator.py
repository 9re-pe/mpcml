import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score
from tqdm import tqdm

from ..models import BaseEmbeddingModel
from .BaseEvaluator import BaseEvaluator


class F1ScoreEvaluator(BaseEvaluator):
    """Class of evaluators computing unpopularity metrics for each user and calculating average."""

    def __init__(
        self, test_set: torch.Tensor, ks: list = [5]
    ):
        """Set test data and metrics.

        Args:
            test_set (torch.Tensor): test data which column is [user_id, item_id, rating, tail].
            ks (int, optional): A list of @k. Defaults to [5].
        """
        super().__init__(test_set)

        self.RATING_COLUMN_NUM = 2
        self.TAIL_COLUMN_NUM = 3
        self.ks = ks

        self.metrics_names = [
            f"F1-Score@{k}" for k in ks
        ]

    def compute_score(
        self, y_test_user_recall: np.ndarray, y_test_user_unpopular: np.ndarray, y_hat_user: np.ndarray
    ) -> pd.DataFrame:
        """Method of computing score for metric.
        This method make a row of DataFrame which has scores for each metrics and k for the user.

        Args:
            y_test_user_recall (np.ndarray): grand truth for computing recall
            y_test_user_unpopular (np.ndarray): grand truth for computing unpopularity
            y_hat_user (np.ndarray): prediction of relevance

        Returns:
            (pd.DataFrame): a row of DataFrame which has scores for each metrics and k for the user.
        """

        if y_test_user_recall.sum() == 0 or y_test_user_unpopular.sum() == 0:
            return pd.DataFrame({name: [0] for name in self.metrics_names})

        else:
            df_eval_sub = pd.DataFrame(
                {
                    f"F1-Score@{k}": [f1_score(y_test_user_recall, y_test_user_unpopular, y_hat_user, k)] for k in self.ks
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
        y_test_user_recall = self.test_set[user_indices, self.RATING_COLUMN_NUM].to("cpu").detach().numpy()
        y_test_user_unpopular = self.test_set[user_indices, self.TAIL_COLUMN_NUM].to("cpu").detach().numpy()

        return self.compute_score(y_test_user_recall, y_test_user_unpopular, y_hat_user)

    def score(
        self, model: BaseEmbeddingModel, reduction="mean", verbose=True
    ) -> pd.DataFrame:
        """Method of calculating average score for all users.

        Args:
            model (BaseEmbeddingModel): models which have user and item embeddings.
            reduction (str, optional): reduction method. Defaults to "mean".
            verbose (bool, optional): displaying progress bar or not during evaluating. Defaults to True.

        Returns:
            pd.DataFrame: a row of DataFrame which has average scores
        """

        users = torch.unique(self.test_set[:, 0])
        df_eval = pd.DataFrame({name: [] for name in self.metrics_names})

        if verbose:
            for uid in tqdm(users):
                df_eval_sub = self.eval_user(model, uid)
                df_eval = pd.concat([df_eval, df_eval_sub])
        else:
            for uid in users:
                df_eval_sub = self.eval_user(model, uid)
                df_eval = pd.concat([df_eval, df_eval_sub])

        if reduction == "mean":
            score = pd.DataFrame(df_eval.mean(axis=0)).T

        else:
            score = df_eval.copy()

        return score


def f1_score(y_test_user_recall, y_test_user_unpopular, y_hat_user, k):
    pred_rank = (-y_hat_user).argsort().argsort() + 1
    pred_topk_flag = (pred_rank <= k).astype(int)

    recall = recall_score(y_test_user_recall, pred_topk_flag)
    unpopularity = recall_score(y_test_user_unpopular, pred_topk_flag)

    if recall + unpopularity == 0.0:
        return 0.0
    else:
        return (2.0 * recall * unpopularity) / (recall + unpopularity)
