import pandas as pd
import torch
from tqdm import tqdm

from ..models import BaseEmbeddingModel
from .BaseEvaluator import BaseEvaluator


class CoverageEvaluator(BaseEvaluator):
    """Class of evaluators computing catalogue coverage for recommendations across all users"""

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

    def item_set_per_user(self, model: BaseEmbeddingModel, uid: int, k: int) -> set:
        """Method of calculating the recommended item sets for given user.

        Args:
            model (BaseEmbeddingModel): models which have user and item embeddings.
            uid (int): user i
            k (int): a number of top item considered.

        Returns:
            (set): the recommended item sets for given user
        """
        # Extract pairs of the target user and items from the test data
        user_indices = self.test_set[:, 0] == uid
        test_set_pair = self.test_set[user_indices, :2]
        # Get the predicted distances of the target user and items
        y_hat_user = model.predict(test_set_pair).to("cpu").detach().numpy()
        # Sort and get the indices of the top k items
        topk_indices = (-y_hat_user).argsort()[:k]
        # Get the items corresponding to the indices and convert to a set
        item_set = set(self.test_set[topk_indices, 1].tolist())

        return item_set

    def score(
        self, model: BaseEmbeddingModel, no_progressbar=False
    ) -> pd.DataFrame:
        """Method of computing catalogue coverage for recommendations across all users.

        Args:
            model (BaseEmbeddingModel): models which have user and item embeddings.
            no_progressbar (bool, optional): displaying progress bar or not during evaluating. Defaults to False.

        Returns:
            pd.DataFrame: a row of DataFrame which has catalogue coverage
        """

        score_dict = {}

        users = torch.unique(self.test_set[:, 0])
        all_items_num = len(torch.unique(self.test_set[:, 1]))

        for k in self.ks:
            eval_name = f"coverage@{k}"
            recommended_items = set()

            # Calculate the recommended item sets for each user and compute their union
            for uid in tqdm(users, disable=no_progressbar):
                recommended_items |= self.item_set_per_user(model, uid, k)

            recommended_items_num = len(recommended_items)
            catalog_coverage = recommended_items_num / all_items_num

            score_dict[eval_name] = [catalog_coverage]

        # convert to a dataframe
        score = pd.DataFrame(score_dict)

        return score

