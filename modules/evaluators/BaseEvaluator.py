import pandas as pd
import torch

from ..models.BaseEmbeddingModel import BaseEmbeddingModel


class BaseEvaluator:
    """ Class of abstract evaluators for trainers
    """

    def __init__(self, test_set: torch.Tensor):
        """ set test data.

        Args:
            test_set (torch.Tensor): tensor of shape (n_pairs, 3) which column is [user_id, item_id, rating]
        """
        self.test_set = test_set

    def score(self, model: BaseEmbeddingModel, no_progressbar=False) -> pd.DataFrame:

        raise NotImplementedError()
