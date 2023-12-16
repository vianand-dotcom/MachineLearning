import numpy as np
import math
import copy
import matplotlib.pyplot as plt
# setup dataset
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40],
                   [1534, 3, 2, 30], [852, 2, 1, 36]])
y_train = np.array([460, 232, 315, 178])
print(x_train, y_train, x_train.shape, y_train.shape, sep="\n")
#
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])


def multiple_var_computecost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    try:
        for i in range(m):
            f_wb_i = np.dot(w, x[i])+b
            cost = cost + (f_wb_i - y[i]) ** 2
        total_cost = cost/2*m
        return total_cost
    except Exception as e:
        print(e)


def multiple_var_computegrad(x, y, w, b):
    # calculate gradient
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        err = np.dot(w, x[i])+b - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i, j]
        dj_db = dj_db+err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw


def multiple_var_gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    index = X.shape[0]
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_history.append(cost_function(X, y, w, b))
        if i % math.ceil(num_iters/100) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history  # return final w,b and J history for graphing


initial_w = np.zeros_like(w_init)
initial_b = 0
iterations = 1000000
alpha = 5.0e-7
w_final, b_final, J_hist = multiple_var_gradient_descent(
    x_train, y_train, initial_w, initial_b, multiple_var_computecost, multiple_var_computegrad, alpha, iterations)
print(f"b,w found by gradient descent :{b_final:0.2f},{w_final}")
m, _ = x_train.shape
for i in range(m):
    print(
        f"prediction:{np.dot(x_train[i],w_final)+b_final:0.2f}, target value: {y_train[i]}")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100+np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("cost vs iteration")
ax1.set_xlabel = ("iteratio step")
ax1.set_ylabel = ("Cost")
ax2.set_title("cost vs iteration(tail)")
ax2.set_xlabel = ("iteratio step")
ax2.set_ylabel = ("Cost")
plt.show()
