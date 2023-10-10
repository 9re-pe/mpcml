import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score
from tqdm import tqdm

from ..models import BaseEmbeddingModel
from .BaseEvaluator import BaseEvaluator


class RecallEvaluator(BaseEvaluator):
    """Class of evaluators computing accuracy metrics for each user and calculating average."""

    def __init__(
        self, test_set: torch.Tensor, ks: list = [5]
    ):
        """Set test data and metrics.

        Args:
            test_set (torch.Tensor): test data which column is [user_id, item_id, rating].
            ks (int, optional): A list of @k. Defaults to [5].

        arguments of each function must be
            y_test_user (np.ndarray): grand truth for the user
            y_hat_user (np.ndarray) : prediction of relevance
            k : a number of top item considered.
        """
        super().__init__(test_set)

        self.RATING_COLUMN_NUM = 2

        self.ks = ks

        self.metrics_names = [
            f"Recall@{k}" for k in ks
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
                    f"Recall@{k}": [recall(y_test_user, y_hat_user, k)] for k in self.ks
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
        # test_setから対象ユーザ(uid)の行をすべて取り出す(user_idとitem_idの列のみ)
        user_indices = self.test_set[:, 0] == uid
        test_set_pair = self.test_set[user_indices, :2]

        # y_hat_user, y_test_userにはそれぞれ予測値と真値が[0, 1, 0, ..., 0]というような形で格納される必要がある
        # y_hat_user, y_test_userのサイズは一緒で、どのアイテムに対する評価か、という順番は対応している必要がある
        # y_hat_user, y_test_userのサイズはユーザごとに変わってくる(trainに含まれる分が違うため)

        # distance for each user and item pair size (n_pairs)
        # test_set_pairに対する結果の予測を行う(proximityを返す)
        y_hat_user = model.predict(test_set_pair).to("cpu").detach().numpy()
        # test_setから対象ユーザのraitingの真値を取得する
        y_test_user = self.test_set[user_indices, self.RATING_COLUMN_NUM].to("cpu").detach().numpy()

        return self.compute_score(y_test_user, y_hat_user)

    def score(
        self, model: BaseEmbeddingModel, reduction="mean", no_progressbar=False
    ) -> pd.DataFrame:
        """Method of calculating average score for all users.

        Args:
            model (BaseEmbeddingModel): models which have user and item embeddings.
            reduction (str, optional): reduction method. Defaults to "mean".
            verbose (bool, optional): displaying progress bar or not during evaluating. Defaults to True.

        Returns:
            pd.DataFrame: a row of DataFrame which has average scores
        """

        # test_setに含まれるuser_idを一意に取り出している => 全ユーザ
        # user_numから作ればいいだけでは？
        # 全映画を評価していて、それがすべてtrainの中に含まれるというパターンがない限り(ないに等しい)
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


def recall(y_test_user: np.ndarray, y_hat_user: np.ndarray, k: int):
    """Function for user-wise evaluators calculating Recall@k

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
