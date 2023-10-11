import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score
from typing import List

from .UserwiseEvaluator import UserwiseEvaluator


class F1ScoreEvaluator(UserwiseEvaluator):
    """Class of evaluators computing unpopularity metrics for each user and calculating average."""

    def __init__(
        self, test_set: torch.Tensor, ks: list = [5]
    ):
        """Set test data and metrics.

        Args:
            test_set (torch.Tensor): test data which column is [user_id, item_id, rating, tail].
            ks (int, optional): A list of @k. Defaults to [5].
        """
        super().__init__(test_set, ks)
        self.metric_name = 'F1-score'
        self.eval_cols = [2, 3]

    def compute_score(
            self, li_y_test_user: List[np.ndarray], y_hat_user: np.ndarray
    ) -> pd.DataFrame:
        """Method of computing score for metric.
        This method make a row of DataFrame which has scores for each metrics and k for the user.

        Args:
            li_y_test_user (List[np.ndarray]): grand truth for the user
            y_hat_user (np.ndarray): prediction of relevance

        Returns:
            (pd.DataFrame): a row of DataFrame which has scores for each metrics and k for the user.
        """

        y_test_user_recall, y_test_user_unpopular = li_y_test_user
        if y_test_user_recall.sum() == 0 or y_test_user_unpopular.sum() == 0:
            return pd.DataFrame({f"{self.metric_name}@{k}": [0] for k in self.ks})
        else:
            df_eval_sub = pd.DataFrame(
                {
                    f"{self.metric_name}@{k}": [self.f1_score(y_test_user_recall, y_test_user_unpopular, y_hat_user, k)]
                    for k in self.ks
                }
            )

        return df_eval_sub

    @staticmethod
    def f1_score(y_test_user_recall, y_test_user_unpopular, y_hat_user, k):
        pred_rank = (-y_hat_user).argsort().argsort() + 1
        pred_topk_flag = (pred_rank <= k).astype(int)

        recall = recall_score(y_test_user_recall, pred_topk_flag)
        unpopularity = recall_score(y_test_user_unpopular, pred_topk_flag)

        if recall + unpopularity == 0.0:
            return 0.0
        else:
            return (2.0 * recall * unpopularity) / (recall + unpopularity)
