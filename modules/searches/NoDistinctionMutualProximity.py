import random

import torch

from .MutualProximity import MutualProximity
from ..models import BaseEmbeddingModel
from ..distributions.BaseDistribution import BaseDistribution


class NoDistinctionMutualProximity(MutualProximity):
    def __init__(
            self,
            model: BaseEmbeddingModel,
            distribution: BaseDistribution,
            n_sample: int = 30,
            bias: float = 0.5
    ):
        super().__init__(model)
        self.distribution = distribution
        self.bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.n_user_sample = int(n_sample * (self.model.n_user / (self.model.n_user + self.model.n_item)))
        self.n_item_sample = n_sample - self.n_user_sample

    def compute_users_distribution_params(self, users: torch.Tensor, n_pairs: int):
        """Compute the probability distribution's parameters for users.

        Args:
            users : tensor of indices for users size (n_pairs, 1)
            n_pairs : the number of user and item pairs
        Returns:
            params1, params2 : probability distribution's parameters size (n_pairs)
        """

        # sample users and items for probability distribution's parameter estimation
        sample_users = torch.tensor(self.sample_ids(self.model.n_user, self.n_user_sample)).unsqueeze(1)
        sample_items = torch.tensor(self.sample_ids(self.model.n_item, self.n_item_sample)).unsqueeze(1)
        sample_users = sample_users.to(self.device)
        sample_items = sample_items.to(self.device)

        u_emb_sampled = self.model.user_embedding(sample_users)
        i_emb_sampled = self.model.item_embedding(sample_items)
        sample_emb = torch.cat((u_emb_sampled, i_emb_sampled), dim=0)

        target_u_emb = self.model.user_embedding(users)[0].unsqueeze(0).repeat(sample_emb.size()[0], 1, 1)

        distances = torch.cdist(target_u_emb, sample_emb).reshape(-1)

        params = self.distribution.approximate_params(distances)

        # Reshape the size to compute MP
        params1 = torch.full((n_pairs,), float(params[0]))
        params2 = torch.full((n_pairs,), float(params[1]))
        params = [params1, params2]

        return params

    def compute_items_distribution_params(self, items: torch.Tensor, n_pairs: int):
        """Compute the probability distribution's parameters for items.

        Args:
            items : tensor of indices for items size (n_pairs, 1)
            n_pairs : the number of user and item pairs
        Returns:
            params1, params2 : probability distribution's parameters size (n_pairs)
        """

        # Sample users and items for probability distribution's parameter estimation
        sample_users = torch.zeros((n_pairs, self.n_user_sample), dtype=torch.int64)
        sample_items = torch.zeros((n_pairs, self.n_item_sample), dtype=torch.int64)
        for i in range(n_pairs):
            sample_users[i] = torch.tensor(self.sample_ids(self.model.n_user, self.n_user_sample))
            sample_items[i] = torch.tensor(self.sample_ids(self.model.n_item, self.n_item_sample))
        sample_users = sample_users.to(self.device)
        sample_items = sample_items.to(self.device)

        u_emb_sampled = self.model.user_embedding(sample_users.view(-1)).view(n_pairs, -1, self.model.n_dim)
        i_emb_sampled = self.model.item_embedding(sample_items.view(-1)).view(n_pairs, -1, self.model.n_dim)
        sample_emb = torch.cat((u_emb_sampled, i_emb_sampled), dim=1)

        target_i_emb = self.model.item_embedding(items)

        distances = self.compute_distance(target_i_emb, sample_emb)
        params = self.distribution.approximate_params(distances)

        return params
