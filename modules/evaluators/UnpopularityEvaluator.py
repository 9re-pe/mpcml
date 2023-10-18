import numpy as np
import pandas as pd
import torch
from typing import List

from .UserwiseEvaluator import UserwiseEvaluator


class UnpopularityEvaluator(UserwiseEvaluator):
    """Class of evaluators computing Unpopularity for each user and calculating average."""

    def __init__(
        self, test_set: torch.Tensor, ks: list = [5]
    ):
        """Set test data and metrics.

        Args:
            test_set (torch.Tensor): test data which column is [user_id, item_id, rating].
            ks (int, optional): A list of @k. Defaults to [5].

        """
        super().__init__(test_set, ks)
        self.metric_name = 'Unpopularity'
        self.eval_cols = [3]

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

        y_test_user = li_y_test_user[0]
        if y_test_user.sum() == 0:
            return pd.DataFrame({f"{self.metric_name}@{k}": [0] for k in self.ks})
        else:
            df_eval_sub = pd.DataFrame(
                {
                    f"{self.metric_name}@{k}": [self.unpopularity(y_test_user, y_hat_user, k)] for k in self.ks
                }
            )

        return df_eval_sub

    @staticmethod
    def unpopularity(y_test_user: np.ndarray, y_hat_user: np.ndarray, k: int):
        """Function for user-wise evaluators calculating Unpopularity@k

        Args:
            y_test_user (np.ndarray): popularity score for the item
            y_hat_user (np.ndarray): prediction of relevance
            k (int): a number of top item considered.

        Returns:
            (float): recall score
        """
        pred_rank = (-y_hat_user).argsort().argsort() + 1
        pred_topk_flag = (pred_rank <= k).astype(int)

        unpopularity_score = np.sum(pred_topk_flag * (1 - y_test_user))

        return unpopularity_score
