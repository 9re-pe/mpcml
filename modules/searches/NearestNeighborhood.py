import torch

from .BaseSearch import BaseSearch
from ..models import BaseEmbeddingModel


class NearestNeighborhood(BaseSearch):
    def __init__(self, model: BaseEmbeddingModel):
        super().__init__(model)

    def predict(self, pairs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pairs : tensor of indices for user and item pairs size (n_pairs, 2).
        Returns:
            dist : distance for each user and item pair size (n_pairs)
        """
        # set users and items
        users = pairs[:, :1]
        items = pairs[:, 1:2]

        # get embeddings
        u_emb = self.model.user_embedding(users)
        i_emb = self.model.item_embedding(items)

        # compute distance
        dist = torch.cdist(u_emb, i_emb).reshape(-1)

        # Because all the embeddings fit within a circle whose radius length is r,
        # the distances between embeddings cannot be longer than the diameter of that circle.
        max_dist = 2 * self.model.max_norm if self.model.max_norm is not None else 100

        # Return values such that the closer the distance between the embeddings, the larger the values.
        return max_dist - dist
