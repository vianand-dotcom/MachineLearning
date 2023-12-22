import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
# generate datasets for training model
centers = [
    [-5, 2], [-2, -2], [1, 2], [5, -2]
]
classes = 4
X_train, y_train = make_blobs(
    n_samples=10000, centers=centers, cluster_std=1.0, random_state=30)

print(f"unique classes {np.unique(y_train)}")
print(f"class representation {y_train[:10]}")
print(f"shape of X_train: {X_train.shape},shape of y_train: {y_train.shape}")

# setup model/neuralnetwork
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        Dense(2, activation='relu',   name="L1"),
        Dense(4, activation='linear', name="L2")
    ]
)
# compile model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)
# here Adam is optimization algorithm the learning rate of alpha is better than gradient descent
# train model
model.fit(
    X_train, y_train,
    epochs=200
)
l1 = model.get_layer("L1")
w1, b1 = l1.get_weights()
print(w1, b1)

l2 = model.get_layer("L2")
w2, b2 = l2.get_weights()
xl2 = np.maximum(0, np.dot(X_train, w1)+b1)
