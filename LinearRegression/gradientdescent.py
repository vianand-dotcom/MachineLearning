import math


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, gradient_function, cost_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    w = w_in
    b = b_in
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)
        # update w and b
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 1000000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        if i % math.ceil(num_iters/10) == 0:
            print(f'Iteration {i:4}: Cost {J_history[-1]:0.2e} ',
                  f'dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ',
                  f'w: {w: 0.3e}, b:{b: 0.5e}')

    return w, b, J_history, p_history  # return w and J,w history for graphing
