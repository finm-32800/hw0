import scipy.optimize
import numpy as np

def find_global_minimum_variance_portfolio(Sigma):
    f = lambda w: w.transpose() @ Sigma @ w
    con = lambda w: np.sum(w) - 1
    w0 = np.array([.2, .2, .2, .2, .2])
    wstar = scipy.optimize.fmin_slsqp(f, w0, f_eqcons=con, acc=1e-09, iprint=2)
    return wstar