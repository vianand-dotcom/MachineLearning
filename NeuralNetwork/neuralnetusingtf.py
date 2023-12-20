import tensorflow as tf
from keras.layers import Dense
from keras import Sequential
from keras.losses import BinaryCrossentropy
import numpy as np


def load_data():
    """ Creates a  data set.
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5          # 12-15 min is best
    X[:, 0] = X[:, 0] * (285-150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1, 1))


def build_model(n_layer, activation_function, units):
    dense = []
    for i in range(n_layer):
        dense.append(Dense(units=units[i], activation=activation_function))
    model = Sequential(
        dense
    )
    print(model)
    return model


def model_compile(model):
    model.compile(loss=BinaryCrossentropy)
    print(model)
    return model


def train_model(X, Y, epochs, model):
    model.fit(X, Y, epochs=epochs)
    print(model)
    return model


n_layer = 10
units = np.random.randint(15, 30, size=10)
X, Y = load_data()
activation_function = "sigmoid"
model = build_model(n_layer, activation_function, units)
model = model_compile(model)
model = train_model(X, Y, 100000, model)
print(model)
