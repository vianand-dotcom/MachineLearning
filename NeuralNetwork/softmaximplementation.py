import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def softmax(z):
    ez = np.exp(z)
    sm = ez/np/sum(ez)
    return (sm)


centers = [
    [-5, 2],
    [-2, -2],
    [1, 2],
    [5, -2]
]
X_train, y_train = make_blobs(
    n_samples=2000, centers=centers, cluster_std=1.0, random_state=30)
# here make blobs generate synthetic datasets for machine learning
# n_samples is the no of tarining set created you can add n_features to describe no if features
# centers is used to distribute your n_samples data into no of cluster by centers.count and each gaussian distrbution point states the cluster center cordinate
# cluster_std represents the standard deviation of the clusters. It controls the spread or dispersion of the generated data points within each cluster
# random_state is used to control the randomness of the data generated during the process
# so each time you run the function it will generate random dataset but with count always 30 in above one
print(X_train, y_train, X_train.shape, y_train.shape)

# setup model
# using Dense we are setting up model for each layer i.e first step after this we will create loss/cost function and optimizer
model = Sequential(
    [
        Dense(units=25, activation='relu', name="L1"),
        Dense(units=15, activation='relu', name="L2"),
        Dense(units=4, activation='softmax', name="L3")
    ]
)
# initialize cost,loss function and optimizers e.g gradient descent
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001)
)
# here Adam is optimization algorithm the learning rate of alpha is better than gradient descent
# train model with generated dataset
model.fit(
    X_train, y_train, epochs=100
)
p_nonpreffered = model.predict(X_train)
print(p_nonpreffered[:2])
print(
    f"largest value:{np.max(p_nonpreffered)},smallest value:{np.min(p_nonpreffered)}")
print(p_nonpreffered.shape)

# minimizing error
prefferd_model = Sequential(
    [
        Dense(units=25, activation='relu'),
        Dense(units=15, activation='relu'),
        Dense(units=4, activation='linear')
    ]
)
# compile by defining loss and optimization algorithm
prefferd_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
)
# here Adam is optimization algorithm the learning rate of alpha is better than gradient descent
# train model
prefferd_model.fit(X_train, y_train, epochs=100)

p_prefferd = prefferd_model.predict(X_train)
print(f"two example output vector:\n{p_prefferd[:2]}")
print(
    f"largest value: {np.max(p_prefferd)}, smallest value:{np.min(p_prefferd)}")

# here output prediction {p_preffered} are not probability distribution but it's a raw output
# to do a probability distribution we will use softmax function
sm_preferred = tf.nn.softmax(p_prefferd).numpy()
print(f"two example output vectors:\n {sm_preferred[:2]}")
print("largest value", np.max(sm_preferred),
      "smallest value", np.min(sm_preferred))

# To select the most likely category, the softmax is not required. One can find the index of the largest output using np.argmax().
for i in range(5):
    print(f"{p_prefferd[i]}, category: {np.argmax(p_prefferd[i])}")
