import random

import torch

from .BaseSearch import BaseSearch
from ..models import BaseEmbeddingModel
from ..distributions.BaseDistribution import BaseDistribution


class MutualProximity(BaseSearch):
    def __init__(
            self,
            model: BaseEmbeddingModel,
            distribution: BaseDistribution,
            n_item_sample: int = 30,
            n_user_sample: int = 30,
    ):
        super().__init__(model)
        self.distribution = distribution
        self.n_item_sample = n_item_sample
        self.n_user_sample = n_user_sample
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def compute_users_distribution_params(self, users: torch.Tensor, n_pairs: int):
        """Compute the probability distribution's parameters for users.

        Args:
            users : tensor of indices for users size (n_pairs, 1)
            n_pairs : the number of user and item pairs
        Returns:
            params1, params2 : probability distribution's parameters size (n_pairs)
        """
        # sample items for probability distribution's parameter estimation
        sample_items = torch.tensor(self.sample_ids(self.model.n_item, self.n_item_sample)).unsqueeze(1)
        sample_items = sample_items.to(self.device)
        i_emb = self.model.item_embedding(sample_items)  # [n_sample, 1, n_dim]
        u_emb = self.model.user_embedding(users)[0].unsqueeze(0).repeat(i_emb.size()[0], 1, 1)  # [n_sample, 1, n_dim]
        distances = torch.cdist(u_emb, i_emb).reshape(-1)  # [n_sample]

        params = self.distribution.approximate_params(distances)

        # reshape the size to compute MP
        if self.distribution.name == 'Empirical':
            params = params.repeat(n_pairs, 1)
        else:
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
        # sample users for probability distribution's parameter estimation
        sample_users = torch.zeros((n_pairs, self.n_user_sample), dtype=torch.int64)
        for i in range(n_pairs):
            row = self.sample_ids(self.model.n_user, self.n_user_sample)
            sample_users[i] = torch.tensor(row)

        i_emb = self.model.item_embedding(items)  # [n_pairs, 1, n_dim]
        sample_users = sample_users.to(self.device)
        u_emb = self.model.user_embedding(sample_users)  # [n_pairs, n_sample, n_dim]
        distances = self.compute_distance(i_emb, u_emb)  # [n_pairs, n_sample]

        params = self.distribution.approximate_params(distances)

        return params

    def compute_mp(self, distances, users_params, items_params):
        """Compute the MP(d_xy) from object X to the other objects Y.

        Args:
            distances : distances for object X to the other objects Y
            users_params : X's probability distribution's parameters
            items_params : Y's probability distribution's parameters
        Returns:
            MP : MP(d_xy) from object X to the other objects Y
        """

        users_params = [param.to(self.device) for param in users_params]
        items_params = [param.to(self.device) for param in items_params]
        x_distribution = self.distribution.get_distribution(users_params)
        y_distribution = self.distribution.get_distribution(items_params)
        mp = (1 - x_distribution.cdf(distances)) * (1 - y_distribution.cdf(distances))

        return mp

    def predict(self, pairs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pairs : tensor of indices for user and item pairs size (n_pairs, 2).
        Returns:
            MP : MP for each user and item pair size (n_pairs)
        """
        # set users and items
        users = pairs[:, :1]
        items = pairs[:, 1:2]

        # get embeddings
        u_emb = self.model.user_embedding(users)
        i_emb = self.model.item_embedding(items)

        # compute distance
        distances = torch.cdist(u_emb, i_emb).reshape(-1)  # [n_pairs]

        n_pairs = pairs.size()[0]
        users_params = self.compute_users_distribution_params(users, n_pairs)  # [n_pairs]
        items_params = self.compute_items_distribution_params(items, n_pairs)  # [n_pairs]
        mp = self.compute_mp(distances, users_params, items_params)

        return mp

    @staticmethod
    def sample_ids(n_objects: int, n_sample: int):
        """
        Args:
            n_objects : the number of all user of item
            n_sample : the number of sample for estimate probability distribution's parameter
        Returns:
            samples : the indices of sampled users or items
        """
        return random.sample(range(n_objects), n_sample)

    @staticmethod
    def compute_distance(tensor1, tensor2):
        """Given 2 Tensor, compute the Euclidean distance between Tensor 1 and each column vector of Tensor 2.

        for example:
            tensor1 = [[1, 2, 3, 4, 5]
                      ,[6, 7, 8, 9, 3]]
            tensor2 = [[[1, 2, 3, 4, 5], [2, 3, 4, 5, 5]],
                       [[3, 4, 5, 6, 7], [4, 5, 6, 7, 8]]]
            distances = [[distance([1, 2, 3, 4, 5],[1, 2, 3, 4, 5]), distance([1, 2, 3, 4, 5],[2, 3, 4, 5, 6])],
                         [distance([6, 7, 8, 9, 3],[3, 4, 5, 6, 7]), distance([6, 7, 8, 9, 3],[4, 5, 6, 7, 8])]]

        Args:
            tensor1 : size (n_pairs, 1, n_dim)
            tensor2 : size (n_pairs, n_sample, n_dim)
        Returns:
            distances : size (n_pairs, n_sample)
        """
        n, _, _ = tensor1.size()
        n, s, _ = tensor2.size()

        # Duplicate tensor1 along the row dimension
        tensor1_repeated = tensor1.repeat(1, s, 1)

        squared_diff = (tensor1_repeated - tensor2) ** 2
        sum_squared_diff = torch.sum(squared_diff, dim=2)
        distances = torch.sqrt(sum_squared_diff.float())

        return distances
