import torch
import torch.distributions as dist

from .BaseDistribution import BaseDistribution


class Gaussian(BaseDistribution):
    def __init__(self):
        super().__init__('Gaussian')

    @staticmethod
    def approximate_params(sample_data):
        """
        Args:
            sample_data : distances between objects
        Returns:
            Gaussian distribution's parameter size (1) or (n_pairs)
        """
        avg = torch.mean(sample_data, dim=-1, keepdim=True)
        var = torch.var(sample_data, dim=-1, unbiased=False, keepdim=True)

        return [avg.squeeze(), var.squeeze()]

    @staticmethod
    def get_distribution(params):
        return dist.Normal(params[0], params[1])
