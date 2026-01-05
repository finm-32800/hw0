import scipy.optimize
import numpy as np

def find_global_minimum_variance_portfolio(Sigma):
    ## TODO: YOUR CODE HERE
    # w0 = np.array([.2, .2, .2, .2, .2])
    # f = lambda w: w.T @ Sigma @ w
    # con = lambda w: np.sum(w) - 1
    # wstar = scipy.optimize.fmin_slsqp(f, w0, f_eqcons=con, acc=1e-09, iprint=2)
    wstar = np.array([.2, .2, .2, .2, .2])
    return wstar