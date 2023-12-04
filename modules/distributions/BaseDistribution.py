class BaseDistribution:
    """ Class of abstract Distribution for MP
    """

    def __init__(self, name):
        self.name = name

    @staticmethod
    def approximate_params(sample_data):
        """

        Args:
            sample_data: distances between objects
        Returns:
            Gaussian distribution's parameter size (1) or (n_pairs)
        """
        raise NotImplementedError

    @staticmethod
    def get_distribution(params):
        """ Get PyTorch probability distribution instance

        Args:
            params: probability distribution's parameters
        Returns:
            PyTorch probability distribution instance
        """
        raise NotImplementedError
