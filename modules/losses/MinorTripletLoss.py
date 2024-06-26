from typing import Optional
import torch
from torch import nn
import pandas as pd

from .BaseLoss import BaseLoss


class MinorTripletLoss(BaseLoss):
    """Class of Triplet Loss taking sum of negative sample."""

    def __init__(
            self,
            feedback_num: pd.DataFrame,
            margin: float = 1,
            a: float = 1,
            b: float = 0.5,
            regularizers: list = [],
            device: Optional[torch.device] = None
    ):
        super().__init__(regularizers)
        self.margin = margin
        self.a = a
        self.b = b
        self.ReLU = nn.ReLU()
        self.feedback_num_dict = feedback_num.set_index('item_id')['feedback_num'].to_dict()

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def main(
        self, embeddings_dict: dict, batch: torch.Tensor, column_names: dict
    ) -> torch.Tensor:
        """Method of forwarding main losses

        Args:
            embeddings_dict (dict): A dictionary of embddings which has following key and values
                user_embedding : embeddings of user, size (n_batch, 1, d)
                pos_item_embedding : embeddings of positive item, size (n_batch, 1, d)
                neg_item_embedding : embeddings of negative item, size (n_batch, n_neg_samples, d)

            batch (torch.Tensor) : A tensor of batch, size (n_batch, *).
            column_names (dict) : A dictionary that maps names to indices of rows of batch.

        Return:
            torch.Tensor : losses, L = Σ [m + pos_dist^2 - min(neg_dist)^2]
        """

        pos_dist = torch.cdist(
            embeddings_dict["user_embedding"], embeddings_dict["pos_item_embedding"]
        )

        neg_dist = torch.cdist(
            embeddings_dict["user_embedding"], embeddings_dict["neg_item_embedding"]
        )

        feedback_num = torch.tensor(
            [self.feedback_num_dict[item_id] for item_id in embeddings_dict["pos_items"].cpu().numpy().flatten().tolist()]
        ).view(-1, 1, 1).to(self.device).to(self.device)
        alpha = self.a * feedback_num ** (-self.b)

        tripletloss = alpha * self.ReLU(self.margin + pos_dist ** 2 - neg_dist ** 2)
        loss = torch.mean(tripletloss)

        return loss
