import generatedata
import matplotlib.pyplot as plt
from computegrad import compute_gradient
from costfunction import compute_cost
import math
import numpy as np
from gradientdescent import gradient_descent

x_train, y_train = generatedata.genoneddata(10, 1)
print(x_train)
print(y_train)

w_init = 0
b_init = 0
iterations = 100000
tmp_alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_gradient, compute_cost)

print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
print(w_final, b_final)
fig, (ax1, ax2) = plt.subplots(
    1, 2, constrained_layout=True, figsize=(12, 12))
# this is used to draw multiple layout figsize represents the size of each image 12,12 represents (1200x1200)
# 1 represents rows 3 represents column means it will just print 3 plot in one layout
# 2,2 means it will print total 4 plot 2 in each row
# note if it is 2,2 it will be unpacked in 2 touple each row will be one touple
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
# 2, 2, constrained_layout=True, figsize=(12, 12))
# fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(
# 2, 3, constrained_layout=True, figsize=(12, 12))
print(fig, (ax1, ax2))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()

print(
    f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
