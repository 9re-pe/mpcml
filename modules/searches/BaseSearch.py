import torch

from ..models import BaseEmbeddingModel

class BaseSearch:
    def __init__(self, model: BaseEmbeddingModel):
        self.model = model

    def predict(self, pairs: torch.Tensor) -> torch.Tensor:
        """Method of predicting relevance for each pair of user and item.

        Args:
            pairs (torch.Tensor): Tensor whose columns are [user_id, item_id]

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: Tensor of relevance size (pairs.shape[0])
        """
        raise NotImplementedError
