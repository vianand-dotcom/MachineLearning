def compute_cost(x, y, w, b):
    """
    This compute cost function is the cost function
    Cost function takes feature aka x_train target aka y_train w and b
    This cost function is actually the metrics which will be used further in gradient descent 
    """
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1/(2*m)*cost
    return total_cost
