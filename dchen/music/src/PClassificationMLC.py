import sys
import time
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from scipy.sparse import issparse
import pickle as pkl


def obj_pclassification(w, X, Y, C, p, weighting=True):
    """
        Objective with L2 regularisation and p-classification loss

        Input:
            - w: current weight vector, flattened L x D + 1 (bias)
            - X: feature matrix, N x D
            - Y: label matrix,   N x L
            - C: regularisation constant, is consistent with scikit-learn C = 1 / (N * \lambda)
            - p: constant for p-classification push loss
    """
    N, D = X.shape
    K = Y.shape[1]
    assert(w.shape[0] == K * D + 1)
    assert(p >= 1)
    assert(C > 0)

    W = w[1:].reshape(K, D)  # reshape weight matrix
    b = w[0]           # bias
    OneN = np.ones(N)  # N by 1
    OneK = np.ones(K)  # K by 1

    if weighting is True:
        # KPosAll = np.sum(Y, axis=1)
        KPosAll = np.dot(Y, OneK)   # number of positive labels for each example, N by 1
        KNegAll = K - KPosAll       # number of negative labels for each example, N by 1
    else:
        KPosAll = np.ones(N)
        KNegAll = np.ones(N)
    A_diag = np.divide(1, KPosAll)  # N by 1
    P_diag = np.divide(1, KNegAll)  # N by 1

    # T1 = np.dot(X, W.T)  # N by K
    T1 = np.dot(X, W.T) + b  # N by K

    T1p = np.multiply(Y, T1)
    T2 = np.multiply(Y, np.exp(-T1p))  # N by K
    T3 = T2 * A_diag[:, None]  # N by K

    # T1n = np.multiply(1-Y, T1)
    T1n = T1 - T1p
    T4 = np.multiply(1-Y, np.exp(p * T1n))  # N by K
    T5 = T4 * P_diag[:, None]  # N by K

    J = np.dot(W.ravel(), W.ravel()) * 0.5 / C
    J += (np.dot(OneN, np.dot(T3, OneK)) + np.dot(OneN, np.dot(T5/p, OneK))) / N
    # J = np.dot(W.ravel(), W.ravel()) * 0.5 / C + (np.dot(OneN, np.dot(T3 + T5/p, OneK))) / N  # not as efficient

    # G = W / C + (np.dot(T3.T, -X) + np.dot(T5.T, X)) / N
    G = W / C + (np.dot((-T3 + T5).T, X)) / N   # more efficient

    db = np.dot(OneN, np.dot(-T3 + T5, OneK)) / N

    gradients = np.concatenate(([db], G.ravel()), axis=0)

    return (J, gradients)


class PClassificationMLC(BaseEstimator):
    """All methods are necessary for a scikit-learn estimator"""

    def __init__(self, C=1, p=1, weighting=True):
        """Initialisation"""

        assert C > 0
        assert p >= 1
        self.C = C
        self.p = p
        self.weighting = weighting
        self.obj_func = obj_pclassification
        self.cost = [ ]
        self.trained = False

    def fit(self, X_train, Y_train):
        """Model fitting by optimising the objective"""
        opt_method = 'L-BFGS-B'  # 'BFGS' #'Newton-CG'
        options = {'disp': 1, 'maxiter': 10**5, 'maxfun': 10**5}  # , 'iprint': 99}
        sys.stdout.write('\nC: %g, p: %g, weighting: %s\n' % (self.C, self.p, self.weighting))
        sys.stdout.flush()

        N, D = X_train.shape
        K = Y_train.shape[1]
        # w0 = np.random.rand(K * D + 1) - 0.5  # initial guess in range [-1, 1]
        w0 = 0.001 * np.random.randn(K * D + 1)
        opt = minimize(self.obj_func, w0, args=(X_train, Y_train, self.C, self.p, self.weighting),
                       method=opt_method, jac=True, options=options)
        if opt.success is True:
            self.b = opt.x[0]
            self.W = np.reshape(opt.x[1:], (K, D))
            self.trained = True
        else:
            sys.stderr.write('Optimisation failed')
            print(opt.items())
            self.trained = False

    def fit_minibatch(self, X_train, Y_train, w0=None, learning_rate=0.001, batch_size=200, n_epochs=10, verbose=0):
        """Model fitting by mini-batch Gradient Descent"""
        # np.random.seed(918273645)
        N, D = X_train.shape
        K = Y_train.shape[1]
        if w0 is None:
            w = 0.001 * np.random.randn(K * D + 1)
        else:
            assert w0.shape[0] == K * D + 1
            w = w0
        n_batches = int((N-1) / batch_size) + 1
        objs = []
        decay = 0.9
        for epoch in range(n_epochs):
            if verbose > 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S'))
            indices = np.arange(N)
            np.random.shuffle(indices)
            for nb in range(n_batches):
                if verbose > 0:
                    sys.stdout.write('\r %d / %d' % (nb+1, n_batches))
                    sys.stdout.flush()
                ix_start = nb * batch_size
                ix_end = min((nb+1) * batch_size, N)
                ix = indices[ix_start:ix_end]
                X = X_train[ix]
                Y = Y_train[ix]
                if issparse(X):
                    X = X.toarray()
                if issparse(Y):
                    Y = Y.toarray()
                J, g = self.obj_func(w, X, Y, C=self.C, p=self.p, weighting=self.weighting)
                w = w - learning_rate * g
                if np.isnan(J):
                    print('J = NaN, training failed.')
                    return
                self.cost.append(J)
            print('\nepoch: %d / %d' % (epoch+1, n_epochs))
            learning_rate *= decay
        self.b = w[0]
        self.W = np.reshape(w[1:], (K, D))
        self.trained = True
        
    def resume_fit_minibatch(self, X_train, Y_train, learning_rate=0.001, batch_size=200, n_epochs=100):
        """Resume fitting the model using SGD"""
        assert self.trained is True, "Only trained model can be resumed"
        w0 = np.concatenate((self.b, self.W.ravel()), axis=-1)
        self.fit_minibatch(X_train=X_train, Y_train=Y_train, w=w0, \
                           learning_rate=learning_rate, batch_size=batch_size, n_epochs=n_epochs)

    def decision_function(self, X_test):
        """Make predictions (score is a real number)"""
        assert self.trained is True, "Can't make prediction before training"
        return np.dot(X_test, self.W.T) + self.b  # log of prediction score

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

    def dump_params(self):
        """Dump the parameters of trained model"""
        if self.trained is False:
            print('Model should be trained first! Do nothing.')
            return
        else:
            params = dict()
            params['C'] = self.C
            params['p'] = self.p
            params['weighting'] = self.weighting
            params['b'] = self.b
            params['W'] = self.W
            params['cost'] = self.cost
            return params
 
    def save_params(self, fname=None):
        """Dump the parameters of trained model"""
        if self.trained is False:
            print('Model should be trained first! Do nothing.')
            return
        else:
            if fname is None:
                fname = 'modelPC-' + time.strftime('%Y-%m-%d-%H-%M-%S') + '.pkl'
            param_dict = self.dump_params()
            pkl.dump(param_dict, open(fname, 'wb'))

    def load_params(self, fname):
        """Load the parameters of a trained model"""
        if self.trained is True:
            print('Cannot load to override trained model! Do nothing.')
            return
        else:
            params = pkl.load(open(fname, 'rb'))
            self.C = params['C']
            self.p = params['p']
            self.weighting = params['weighting']
            self.b = params['b']
            self.W = params['W']
            self.cost = params['cost']
            self.trained = True

