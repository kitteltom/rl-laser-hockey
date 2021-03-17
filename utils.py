import numpy as np


def normalize(x, mu, sigma):
    return (x - mu) / (sigma + 1e-6)


def undo_normalize(x, mu, sigma):
    return x * sigma + mu


def moving_average(x, w, mode='valid'):
    return np.convolve(x, np.ones(w), mode) / w
