import sys
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator


def obj_pnorm(w, X, Y, C, p, weighting=True):
    """
        Objective with L2 regularisation and p-norm push loss

        Input:
            - w: current weight vector, flattened L x D
            - X: feature matrix, N x D
            - Y: label matrix,   N x L
            - C: regularisation constant, is consistent with scikit-learn C = 1 / (N * \lambda)
            - p: constant for p-norm push loss
    """
    N, D = X.shape
    K = Y.shape[1]
    assert(w.shape[0] == K * D)
    assert(p >= 1)
    assert(C > 0)

    W = w.reshape(K, D)  # reshape weight matrix
    OneN = np.ones(N)  # N by 1
    OneK = np.ones(K)  # K by 1

    if weighting is True:
        KPosAll = np.dot(Y, OneK)    # number of positive labels for each example, N by 1
        KNegAll = K - KPosAll        # number of negative labels for each example, N by 1
    else:
        KPosAll = np.ones(N)
        KNegAll = np.ones(N)
    A_diag = np.divide(1, KPosAll)  # N by 1
    P_diag = np.divide(1, KNegAll)  # N by 1

    T1 = np.dot(X, W.T)  # N by K

    T1p = np.multiply(Y, T1)
    T2 = np.multiply(Y, np.exp(-T1p))  # N by K
    T3 = T2 * A_diag[:, None]  # N by K
    T4 = np.dot(T3, OneK)  # N by 1

    # T1n = np.multiply(1-Y, T1)
    T1n = T1 - T1p
    T5 = np.multiply(1-Y, np.exp(p * T1n))  # N by K
    T6 = T5 * P_diag[:, None]  # N by K
    T7 = np.dot(T6, OneK)  # N by 1

    T8 = np.power(T4, p-1)  # N by 1
    T9 = np.power(T4, p)    # N by 1
    J = np.dot(w, w) * 0.5 / C + np.dot(OneN, T9 * T7) / N

    T10 = T8 * T7  # N by 1
    G1 = p * np.dot((T3 * T10[:, None]).T, -X)  # K by D
    G2 = p * np.dot((T6 * T9[:, None]).T, X)  # K by D

    G = W / C + (G1 + G2) / N

    return (J, G.ravel())


class PNormPushMLC(BaseEstimator):
    """All methods are necessary for a scikit-learn estimator"""

    def __init__(self, C=1, p=1, weighting=True):
        """Initialisation"""

        assert C > 0
        assert p >= 1
        self.C = C
        self.p = p
        self.weighting = weighting
        self.obj_func = obj_pnorm
        self.trained = False

    def fit(self, X_train, Y_train):
        """Model fitting by optimising the objective"""
        opt_method = 'L-BFGS-B'  # 'BFGS' #'Newton-CG'
        options = {'disp': 1, 'maxiter': 10**5, 'maxfun': 10**5}  # , 'iprint': 99}
        sys.stdout.write('\nC: %g, p: %g, weighting: %s\n' % (self.C, self.p, self.weighting))
        sys.stdout.flush()

        N, D = X_train.shape
        K = Y_train.shape[1]
        w0 = 0.001 * np.random.randn(K * D)
        opt = minimize(self.obj_func, w0, args=(X_train, Y_train, self.C, self.p, self.weighting),
                       method=opt_method, jac=True, options=options)
        if opt.success is True:
            self.W = np.reshape(opt.x, (K, D))
            self.trained = True
        else:
            sys.stderr.write('Optimisation failed')
            print(opt.items())
            self.trained = False

    def decision_function(self, X_test):
        """Make predictions (score is real number)"""

        assert self.trained is True, "Can't make prediction before training"
        return np.dot(X_test, self.W.T)  # log of prediction score

    def predict(self, X_test):
        return self.decision_function(X_test)
    #    """Make predictions (score is boolean)"""
    #    preds = sigmoid(self.decision_function(X_test))
    #    #return (preds >= 0)
    #    assert self.TH is not None
    #    return preds >= self.TH

    # inherit from BaseEstimator instead of re-implement
    #
    # def get_params(self, deep = True):
    # def set_params(self, **params):
