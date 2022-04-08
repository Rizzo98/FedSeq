import numpy as np


def generate_sigma_noise(dataset_size: int, epsilon: float, delta: float, C: float) -> float:
    c = np.sqrt(2 * np.log(1.25 / delta))
    delta_s = 2 * C / dataset_size
    return c * delta_s / epsilon
