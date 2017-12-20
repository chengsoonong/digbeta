import sys
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from scipy.sparse import issparse


def obj_toppush(w, X, Y, C, r=1, weighting=True):
    """
        Objective with L2 regularisation and top push loss

        Input:
            - w: current weight vector, flattened L x D
            - X: feature matrix, N x D
            - Y: label matrix,   N x K
            - C: regularisation constant, C = 1 / lambda
            - r: parameter for log-sum-exp approximation
            - weighting: if True, divide K+ in top-push loss
    """
    N, D = X.shape
    K = Y.shape[1]
    assert(w.shape[0] == K * D)
    assert(r > 0)
    assert(C > 0)

    if issparse(Y):
        Y = Y.toarray()

    W = w.reshape(K, D)  # theta

    J = 0.0  # cost
    G = np.zeros_like(W)  # gradient matrix

    # instead of using diagonal matrix to scale each row of a matrix with a different factor,
    # we use Mat * Vec[:, None] which is more memory efficient

    if weighting is True:
        KPosAll = np.sum(Y, axis=1)  # number of positive labels for each example, N by 1
    else:
        KPosAll = np.ones(N)

    A_diag = 1.0 / KPosAll
    AY = Y * A_diag[:, None]

    # if issparse(Y):
    #    A = lil_matrix((N, N), dtype=np.float)
    #    A.setdiag(A_diag)
    #    AY = np.dot(A.tocsr(), Y.tocsr())
    # else:
    #    AY = Y * A_diag[:, None]

    T1 = np.dot(X, W.T)  # N by K
    # m0 = np.max(T1)  # underflow in np.exp(r*T1 - m1)
    m0 = 0.5 * (np.max(T1) + np.min(T1))
    m1 = r * m0
    # print('----------------')
    # print(m0, np.min(T1))

    T2 = np.multiply(1 - Y, np.exp(r * T1 - m1))  # N by K
    B_tilde_diag = np.dot(T2, np.ones(K))
    # print(np.max(B_tilde_diag), np.min(B_tilde_diag))  # big numbers here, can cause overflow in T3

    # T3 = np.exp(-T1 + m0) * np.power(B_tilde_diag, 1.0 / r)[:, None]
    # T4 = np.multiply(AY, np.log1p(T3))
    T3 = (-T1 + m0) + (1.0 / r) * np.log(B_tilde_diag)[:, None]
    # print(np.min(T3), np.max(T3))
    m2 = 0.5 * (np.min(T3) + np.max(T3))
    # T4 = np.logaddexp(0, T3)
    T4 = np.logaddexp(-m2, T3-m2) + m2
    T5 = np.multiply(AY, T4)

    J = np.dot(w, w) * 0.5 / C + np.dot(np.ones(N), np.dot(T5, np.ones(K))) / N

    # T5 = 1.0 / (1.0 + np.divide(1.0, T3))
    # T5 = np.divide(T3, 1 + T3)
    T6 = np.exp(T3 - T4)
    O_diag = np.dot(np.multiply(Y, T6), np.ones(K))
    T7 = A_diag * (1.0 / B_tilde_diag) * O_diag

    G1 = np.dot(np.multiply(AY, T6).T, -X)
    G2 = np.dot((T2 * T7[:, None]).T, X)

    G = W / C + (G1 + G2) / N

    # print(J)

    return (J, G.ravel())


class TopPushMLC(BaseEstimator):
    """All methods are necessary for a scikit-learn estimator"""

    def __init__(self, C=1, r=1, weighting=True):
        """Initialisation"""

        assert C > 0
        assert r > 0
        assert type(weighting) == bool
        self.C = C
        self.r = r
        self.weighting = weighting
        self.trained = False

    def fit(self, X_train, Y_train):
        """Model fitting by optimising the objective"""
        opt_method = 'L-BFGS-B'  # 'BFGS' #'Newton-CG'
        options = {'disp': 0, 'maxiter': 10**5, 'maxfun': 10**5}  # , 'iprint': 99}
        print('\nC: %g, r: %g' % (self.C, self.r))

        N, D = X_train.shape
        K = Y_train.shape[1]
        # w0 = np.random.rand(K * D) - 0.5  # initial guess in range [-1, 1]
        w0 = 0.001 * np.random.randn(K * D)
        opt = minimize(obj_toppush, w0, args=(X_train, Y_train, self.C, self.r, self.weighting),
                       method=opt_method, jac=True, options=options)
        if opt.success is True:
            self.W = np.reshape(opt.x, (K, D))
            self.trained = True
        else:
            sys.stderr.write('Optimisation failed')
            print(opt.items())
            self.trained = False

    def fit_SGD(self, X_train, Y_train, learning_rate=0.001, batch_size=200, n_epochs=100):
        np.random.seed(918273645)
        N, D = X_train.shape
        K = Y_train.shape[1]
        w = 0.001 * np.random.randn(K * D)
        n_batches = int((N-1) / batch_size) + 1
        for epoch in range(n_epochs):
            Je = 0.0
            indices = np.arange(N)
            np.random.shuffle(indices)
            for nb in range(n_batches):
                ix_start = nb * batch_size
                ix_end = min((nb+1) * batch_size, N)
                ix = indices[ix_start:ix_end]
                X = X_train[ix]
                Y = Y_train[ix]
                J, g = obj_toppush(w, X, Y, C=self.C, r=self.r, weighting=self.weighting)
                w = w - learning_rate * g
                Je += J
                # Je += J * len(ix)
                sys.stdout.write('\r %d / %d' % (nb+1, n_batches))
                sys.stdout.flush()
            print()
            Je /= n_batches
            print('epoch: %d / %d, obj: %.6f' % (epoch+1, n_epochs, Je))
        self.W = np.reshape(w, (K, D))
        self.trained = True

    def decision_function(self, X_test):
        """Make predictions (score is real number)"""
        assert self.trained is True, "Can't make prediction before training"
        return np.dot(X_test, self.W.T)

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
