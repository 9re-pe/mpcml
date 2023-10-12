import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import List

from ..models import BaseEmbeddingModel
from .BaseEvaluator import BaseEvaluator


class UserwiseEvaluator(BaseEvaluator):
    """Class of evaluators computing accuracy metrics for each user and calculating average."""

    def __init__(
            self,
            test_set: torch.Tensor,
            ks: list = [5]
    ):
        """Set test data and metrics.

        Args:
            test_set (torch.Tensor): test data which column is [user_id, item_id, rating].
            metric_name (str): name of metric
            ks (int, optional): A list of @k. Defaults to [5].

        arguments of each function must be
            y_test_user (np.ndarray): grand truth for the user
            y_hat_user (np.ndarray) : prediction of relevance
            k : a number of top item considered.
        """
        super().__init__(test_set)
        self.ks = ks
        self.metric_name = None
        self.eval_cols = []

    def compute_score(
            self, li_y_test_user: List[np.ndarray], y_hat_user: np.ndarray
    ) -> pd.DataFrame:

        raise NotImplementedError()

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
        li_y_test_user = []
        for col in self.eval_cols:
            li_y_test_user.append(self.test_set[user_indices, col].to("cpu").detach().numpy())
        y_hat_user = model.predict(test_set_pair).to("cpu").detach().numpy()

        return self.compute_score(li_y_test_user, y_hat_user)

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
        df_eval = pd.DataFrame({f"{self.metric_name}@{k}": [] for k in self.ks})

        for uid in tqdm(users, disable=no_progressbar):
            df_eval_sub = self.eval_user(model, uid)
            df_eval = pd.concat([df_eval, df_eval_sub])

        if reduction == "mean":
            score = pd.DataFrame(df_eval.mean(axis=0)).T

        else:
            score = df_eval.copy()

        return score
