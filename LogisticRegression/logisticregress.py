import numpy as np
import matplotlib.pyplot as plt
import copy
import math
# generate training data
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [
                   3, 0.5], [2, 2], [1, 2.5]])  # (m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])
# plot the data
fig, ax = plt.subplots(1, 1, figsize=(4, 4))

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
# plt.show()


def sigmoid(z):
    return 1/(1+np.exp(-z))


def compute_cost_logistic(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = (np.dot(w, x[i])+b)
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m
    return cost


def compute_gradient_logistic(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        z_i = (np.dot(w, x[i])+b)
        f_wb_i = sigmoid(z_i)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i*x[i, j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m
    dj_db = dj_db / m
    return dj_dw, dj_db


w_tmp = np.array([1, 1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha*dj_db
        # Save cost J at each iteration
        if i < 100000:      # prevent resource exhaustion
            J_history.append(compute_cost_logistic(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w, b, J_history


w_tmp = np.zeros_like(X_train[0])
b_tmp = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")
