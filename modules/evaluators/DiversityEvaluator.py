import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..models import BaseEmbeddingModel
from .BaseEvaluator import BaseEvaluator


class DiversityEvaluator(BaseEvaluator):
    """Class of evaluators computing diversity for each user and calculating average."""

    def __init__(
            self, test_set: torch.Tensor, item_feature_set: torch.Tensor, ks: list = [5], emb_sim: bool = False
    ):
        """Set test data and metrics.

        Args:
            test_set (torch.Tensor): test data which column is [user_id, item_id, rating].
            item_feature_set (torch.Tensor): item feature data column is [item_id, feature0(0,1), feature1(0, 1), ...]
            ks (int, optional): A list of @k. Defaults to [5].

        arguments of each function must be
            y_hat_user (np.ndarray) : prediction of relevance
            k : a number of top item considered.
        """
        super().__init__(test_set)

        self.item_feature_set = item_feature_set
        self.ks = ks
        self.emb_sim = emb_sim

    def eval_user(self, model: BaseEmbeddingModel, uid: int) -> pd.DataFrame:
        """Method of evaluating for given user.

        Args:
            model (BaseEmbeddingModel): models which have user and item embeddings.
            uid (int): user id

        Returns:
            (pd.DataFrame): a row of DataFrame which has scores for each metrics and k for the user.
        """

        score_dict = {}

        # Extract pairs of the target user and items from the test data
        user_indices = self.test_set[:, 0] == uid
        test_set_pair = self.test_set[user_indices, :2]
        # Get the predicted distances of the target user and items
        y_hat_user = model.predict(test_set_pair).to("cpu").detach().numpy()

        for k in self.ks:
            eval_name = f"diversity@{k}"

            # Get the IDs of the top k items
            pred_topk_indices = (-y_hat_user).argsort()[:k]
            topk_item_ids = self.test_set[pred_topk_indices, 1]

            if self.emb_sim or self.item_feature_set is None:
                i_emb = model.item_embedding(topk_item_ids)
                dist = torch.cdist(i_emb, i_emb).reshape(-1)
                sum_sim = torch.sum(dist)
            else:
                topk_items = self.item_feature_set[np.isin(self.item_feature_set[:, 0], topk_item_ids)]
                topk_items = topk_items.float()
                normalized_topk_items = F.normalize(topk_items[:, 1:], p=2, dim=1)
                sim = torch.mm(normalized_topk_items, normalized_topk_items.T)
                sum_sim = torch.sum(sim) - torch.trace(sim)

            diversity_score = sum_sim / (k * (k - 1))
            score_dict[eval_name] = [diversity_score]

        df_eval_sub = pd.DataFrame(score_dict)

        return df_eval_sub

    def score(
            self, model: BaseEmbeddingModel, reduction="mean", verbose=True
    ) -> pd.DataFrame:
        """Method of calculating average diversity for all users.

        Args:
            model (BaseEmbeddingModel): models which have user and item embeddings.
            reduction (str, optional): reduction method. Defaults to "mean".
            verbose (bool, optional): displaying progress bar or not during evaluating. Defaults to True.

        Returns:
            pd.DataFrame: a row of DataFrame which has average scores
        """

        users = torch.unique(self.test_set[:, 0])
        df_eval = pd.DataFrame({f"diversity@{k}": [] for k in self.ks})

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
