import numpy


def get_normally_distributed_matrix(a: int, b: int) -> numpy.ndarray:
    """
     Generating (a x b) matrix of
     normally distributed random values (from -0.5 to 0.5)
    """
    normal_distribution_center = 0.0
    return numpy.random.normal(normal_distribution_center, pow(a, -0.5), (a, b))
