import random

import torch

from .BaseSearch import BaseSearch
from ..models import BaseEmbeddingModel


class Mymp(BaseSearch):
    def __init__(
            self,
            model: BaseEmbeddingModel,
            search_range: int = 100
    ):
        super().__init__(model)
        self.search_range = search_range

    def predict(self, pairs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pairs : tensor of indices for user and item pairs size (n_pairs, 2).
        Returns:
            MP : MP for each user and item pair size (n_pairs)
        """
        # set users and items
        user_id  = pairs[:, :1]
        item_ids = pairs[:, 1:2]

        u_emb = self.model.user_embedding(user_id)
        i_emb = self.model.item_embedding(item_ids)
        distances = torch.cdist(u_emb, i_emb).reshape(-1)

        search_range = min(self.search_range, distances.numel())
        _, top_item_indices = torch.topk(distances, search_range, largest=False)

        p_user2items = torch.zeros_like(distances)
        for i, index in enumerate(top_item_indices):
            rank = i + 1
            p = 1 - rank / search_range
            p_user2items[index] = p

        p_item2users = torch.zeros_like(distances)
        for index in top_item_indices:
            users_emb = self.model.user_embedding(torch.arange(self.model.n_user).unsqueeze(-1))
            item_emb = self.model.item_embedding(item_ids[index])
            distances_item = torch.cdist(users_emb, item_emb).reshape(-1)
            rank = (distances_item <= distances[index]).sum()
            p = 1 - rank / self.model.n_user
            p_item2users[index] = p

        mp = p_user2items * p_item2users

        return mp
