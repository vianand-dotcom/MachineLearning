import numpy as np


def genoneddata(row, column):
    """
    This function is to generate a random one d array for our linear
    regression implementaion.
    """
    if (column == 1):
        x_train = np.random.rand(row, column) * 10
        y_train = np.random.rand(row, column) * 1000
        x_train = np.array([1.0, 2.0])  # features
        y_train = np.array([300.0, 500.0])
    else:
        print("colun value needs to be one")
    return x_train, y_train
