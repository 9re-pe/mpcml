import torch
from torch.distributions import Distribution

from .BaseDistribution import BaseDistribution


class Empirical(BaseDistribution):
    def __init__(self):
        super().__init__('Empirical')

    @staticmethod
    def approximate_params(all_data):

        return all_data

    @staticmethod
    def get_distribution(all_data):

        return DistEmpirical(all_data)


class DistEmpirical(Distribution):
    arg_constraints = {}

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.sorted_data, _ = torch.sort(self.data, dim=1)

    def cdf(self, value):
        # value should be of size (n_pair)
        value = value.unsqueeze(1)  # reshape to (n_pair, 1)
        return (self.sorted_data.le(value).sum(dim=1).float() / self.data.size(1)).squeeze()
