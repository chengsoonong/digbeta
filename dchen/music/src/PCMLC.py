import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from scipy.sparse import issparse
import pickle as pkl


def obj_pclassification(w, X, Y, C, p, weighting='samples', similarMat=None):
    """
        Objective with L2 regularisation and p-classification loss

        Input:
            - w: current weight vector, flattened L x D + 1 (bias)
            - X: feature matrix, N x D
            - Y: label matrix,   N x L
            - C: regularisation constant, is consistent with scikit-learn C = 1 / (N * \lambda)
            - p: constant for p-classification push loss
            - weighting: weight the exponential surrogate by the #positive or #negative labels or samples,
                  valid assignment: None, 'samples', 'labels', 'both'
                  - None: do not weight
                  - 'samples': weight by the #positive or #negative samples per label (weighting vertically)
                  - 'labels': weight by the #positive or #negative labels per example (weighting horizontally)
                  - 'both': equivalent to turn on both 'samples' and 'labels'
            - similarMat: square symmetric matrix, require the parameters of label_i and label_j should be similar
                          (by regularising their difference) if entry (i,j) is 1
    """
    N, D = X.shape
    K = Y.shape[1]
    assert w.shape[0] == K * D + 1
    assert p >= 1
    assert C > 0
    assert weighting in [None, 'samples', 'labels', 'both'], \
           'Valid assignment for "weighting" are: None, "samples", "labels", "both".'
    if similarMat is not None:
        assert similarMat.shape == (K, K)

    isnan = np.isnan(Y)
    if np.any(isnan):
        Yp = np.nan_to_num(Y)
        Yn = 1 - Yp - isnan
    else:
        Yp = Y
        Yn = 1 - Y
    Yp = Yp.astype(np.int)
    Yn = Yn.astype(np.int)

    W = w[1:].reshape(K, D)  # reshape weight matrix
    b = w[0]                 # bias

    T1 = np.dot(X, W.T) + b  # N by K
    T2 = np.multiply(-Yp + p * Yn, T1)
    T3 = np.multiply(Yp, -T1)
    T4 = np.multiply(Yn, p * T1)

    def _cost_grad(weight):
        if weight is None:
            Tp = 0
            Tn = np.log(p) * Yn
            P = Yp
            Q = Yn
            totalNum = 1
        else:
            if weight == 'samples':
                ax = 0
                shape = (1, K)
            elif weight == 'labels':
                ax = 1
                shape = (N, 1)
            numPosAll = np.sum(Yp, axis=ax)
            numNegAll = np.sum(Yn, axis=ax)
            pnumNegAll = p * numNegAll
            zero_pix = np.where(numPosAll == 0)[0]
            zero_nix = np.where(numNegAll == 0)[0]
            if zero_pix.shape[0] > 0:
                numPosAll[zero_pix] = 1
            if zero_nix.shape[0] > 0:
                numNegAll[zero_nix] = 1
                pnumNegAll[zero_nix] = 1
            Tp = Yp * np.log(numPosAll).reshape(shape)
            Tn = Yn * np.log(pnumNegAll).reshape(shape)
            P = Yp * np.divide(1, numPosAll).reshape(shape)
            Q = Yn * np.divide(1, numNegAll).reshape(shape)
            totalNum = np.prod(shape)

        _cost = logsumexp((T2 - Tp - Tn).ravel()) - np.log(totalNum)
        T5 = np.multiply(P, np.exp(T3 - _cost))
        T6 = np.multiply(Q, np.exp(T4 - _cost))

        _dW = np.dot((-T5 + T6).T, X) / totalNum
        _db = np.sum(-T5 + T6) / totalNum
        return _cost, _dW, _db

    if weighting != 'both':
        cost, dW, db = _cost_grad(weighting)
        J = np.dot(W.ravel(), W.ravel()) * 0.5 / C + cost
        dW += W / C
    else:
        cost1, dW1, db1 = _cost_grad(weight='samples')
        cost2, dW2, db2 = _cost_grad(weight='labels')
        J = np.dot(W.ravel(), W.ravel()) * 0.5 / C + cost1 + cost2
        dW = W / C + dW1 + dW2
        db = db1 + db2

    if similarMat is not None:
        M = -1. * similarMat
        np.fill_diagonal(M, np.sum(similarMat, axis=1))
        J += np.sum(np.multiply(np.dot(W, W.T), M)) * 0.5 / C
        dW += np.dot(M, W) / C

    grad = np.concatenate(([db], dW.ravel()), axis=0)
    return (J, grad)


class PCMLC(BaseEstimator):
    """All methods are necessary for a scikit-learn estimator"""

    def __init__(self, C=1, p=1, weighting=True, verticalWeighting=False, similarMat=None):
        """Initialisation"""

        assert C > 0
        assert p >= 1
        self.C = C
        self.p = p
        self.weighting = weighting
        self.verticalWeighting = verticalWeighting
        self.similarMat = similarMat
        self.obj_func = obj_pclassification
        self.cost = []
        self.trained = False

    def fit(self, X_train, Y_train):
        """Model fitting by optimising the objective"""
        opt_method = 'L-BFGS-B'  # 'BFGS' #'Newton-CG'
        options = {'disp': 1, 'maxiter': 10**5, 'maxfun': 10**5}  # 'eps': 1e-5}  # , 'iprint': 99}
        sys.stdout.write('\nC: %g, p: %g, weighting: %s\n' % (self.C, self.p, self.weighting))
        sys.stdout.flush()

        N, D = X_train.shape
        K = Y_train.shape[1]
        # w0 = np.random.rand(K * D + 1) - 0.5  # initial guess in range [-1, 1]
        # w0 = 0.001 * np.random.randn(K * D + 1)
        # w0 = 0.001 * np.random.randn(K * D + 1)
        w0 = np.zeros(K * D + 1)
        opt = minimize(self.obj_func, w0, args=(X_train, Y_train,
                       self.C, self.p, self.weighting, self.verticalWeighting, self.similarMat),
                       method=opt_method, jac=True, options=options)
        if opt.success is True:
            self.b = opt.x[0]
            self.W = np.reshape(opt.x[1:], (K, D))
            self.trained = True
        else:
            sys.stderr.write('Optimisation failed')
            print(opt.items())
            self.trained = False

    def fit_minibatch(self, X_train, Y_train, w0=None, learning_rate=0.1, batch_size=200, n_epochs=10, verbose=0):
        """Model fitting by mini-batch Gradient Descent"""
        # np.random.seed(918273645)
        N, D = X_train.shape
        K = Y_train.shape[1]
        if w0 is None:
            # w = 0.001 * np.random.randn(K * D + 1)
            w = np.zeros(K * D + 1)
        else:
            assert w0.shape[0] == K * D + 1
            w = w0
        n_batches = int((N-1) / batch_size) + 1
        decay = 0.8
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
                J, g = self.obj_func(w, X, Y, C=self.C, p=self.p, weighting=self.weighting,
                                     verticalWeighting=self.verticalWeighting, similarMat=self.similarMat)
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
        self.fit_minibatch(X_train=X_train, Y_train=Y_train, w=w0,
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