import matplotlib.pyplot as plt
import numpy as np

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
    2, 3, constrained_layout=True, figsize=(12, 12))
# (size in 1000 square feet)
X_train = np.array([[1.0], [2.0]], dtype=np.float32)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)
ax1.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
# this is to depict the point on graph
ax1.legend(fontsize='xx-large')
ax1.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax1.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()
