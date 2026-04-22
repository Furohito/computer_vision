import numpy as np

def get_hakim():
    data = np.load("data/hakim_final.npz")

    X = data["X"]
    y = data["y"]

    one_hot = np.zeros((y.size, 10))
    one_hot[np.arange(y.size), y.astype(int)] = 1

    return X, one_hot