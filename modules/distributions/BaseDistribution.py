class BaseDistribution:
    """ Class of abstract Distribution for MP
    """

    def __init__(self, name):
        self.name = name

    @staticmethod
    def approximate_params(sample_data):
        raise NotImplementedError

    @staticmethod
    def get_distribution(params):
        raise NotImplementedError
