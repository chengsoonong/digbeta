import os
import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from scipy.sparse import issparse, csr_matrix
import pickle as pkl


def obj_pclassification(w, X, Y, p, C1=1, C2=1, C3=1, weighting='labels', similarMat=None):
    """
        Objective with L2 regularisation and p-classification loss

        Input:
            - w: current weight vector, flattened L x D + 1 (bias)
            - X: feature matrix, N x D
            - Y: label matrix,   N x L
            - p: constant for p-classification push loss
            - C1-C3: regularisation constants
            - weighting: weight the exponential surrogate by the #positive or #negative labels or samples,
                  valid assignment: None, 'samples', 'labels', 'both'
                  - None: do not weight
                  - 'samples': weight by the #positive or #negative samples per label (weighting vertically)
                  - 'labels': weight by the #positive or #negative labels per example (weighting horizontally)
                  - 'both': equivalent to turn on both 'samples' and 'labels'
            - similarMat: square symmetric matrix, require the parameters of label_i and label_j should be similar
                          (by regularising their difference) if entry (i,j) is 1.
                          This is the adjacent matrix of playlists (nodes), and playlists of the same user form a clique.
    """
    N, D = X.shape
    K = Y.shape[1]
    assert w.shape[0] == K * D + 1
    assert p >= 1
    assert C1 > 0
    assert C2 > 0
    assert C3 > 0
    assert weighting in [None, 'samples', 'labels', 'both'], \
        'Valid assignment for "weighting" are: None, "samples", "labels", "both".'
    if similarMat is not None:
        # assert similarMat.shape == (K, K)
        # assert np.isclose(np.sum(1. * similarMat - 1. * similarMat.T), 0)  # trust the input
        # assert np.isclose(np.sum(similarMat.diagonal()), 0)  # compatible to sparse matrix
        nrows, ncols = similarMat.shape
        assert nrows == ncols
        assert nrows >= K
        if nrows > K:
            similarMat = similarMat[:K, :K]

    isnan = np.isnan(Y)
    if np.any(isnan):
        Yp = np.nan_to_num(Y)
        Yn = 1 - Yp - isnan
        assert np.sum(Yp + Yn + isnan) == np.prod(Y.shape)
    else:
        Yp = Y
        Yn = 1 - Y
    Yp = Yp.astype(np.int)
    Yn = Yn.astype(np.int)

    W = w[1:].reshape(K, D)  # reshape weight matrix
    b = w[0]                 # bias

    T1 = np.dot(X, W.T) + b  # N by K
    # T2 = np.multiply(-Yp + p * Yn, T1)
    T2 = np.multiply(Yp, -T1)
    T3 = np.multiply(Yn, p * T1)

    def _cost_grad(weight):
        if weight is None:
            Tp = 0
            Tn = np.log(p) * Yn
            P = Yp
            Q = Yn
            num = 1
            return Tp, Tn, P, Q, num
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
        num = np.prod(shape)
        return Tp, Tn, P, Q, num

    if weighting == 'both':
        Tp1, Tn1, P1, Q1, num1 = _cost_grad(weight='samples')
        Tp2, Tn2, P2, Q2, num2 = _cost_grad(weight='labels')
        T5 = T2 + T3 - Tp1 - Tn1 - np.log(num1)
        T6 = T2 + T3 - Tp2 - Tn2 - np.log(num2) + np.log(C2)
        cost = logsumexp(np.concatenate([T5.ravel(), T6.ravel()], axis=-1))
        J = np.dot(W.ravel(), W.ravel()) * 0.5 / C1 + cost
        T7 = np.exp(T2 - cost)
        T8 = np.exp(T3 - cost)
        T11 = -np.multiply(P1, T7) + np.multiply(Q1, T8)
        T12 = -np.multiply(P2, T7) + np.multiply(Q2, T8)
        dW = W / C1 + np.dot(T11.T, X) / num1 + C2 * np.dot(T12.T, X) / num2
        db = np.sum(T11) / num1 + C2 * np.sum(T12) / num2
    else:
        Tp, Tn, P, Q, num = _cost_grad(weighting)
        cost = logsumexp((T2 + T3 - Tp - Tn).ravel()) - np.log(num)
        # cost = logsumexp((T2 + T3 - Tp - Tn - np.log(num)).ravel())
        J = np.dot(W.ravel(), W.ravel()) * 0.5 / C1 + cost
        T5 = -np.multiply(P, np.exp(T2 - cost)) + np.multiply(Q, np.exp(T3 - cost))
        dW = W / C1 + np.dot(T5.T, X) / num
        db = np.sum(T5) / num

    if similarMat is not None:
        if issparse(similarMat):
            sumVec = similarMat.sum(axis=1)  # K by 1: #playlists_of_u - 1
            M = similarMat.multiply(-1.)
            M = M.tolil()
            M.setdiag(sumVec)
            M = M.tocsr()
            nzcols = np.nonzero(sumVec)[0]
            extra_cost = 0.
            for k in nzcols:
                # nzc = M[k, :].nonzero()[1]  # non-zero columns in the k-th row, expensive
                # extra_cost += M[k, nzc].dot(np.dot(W[nzc, :], W[k, :]))
                # Mk = M[k, nzc].toarray()
                # extra_cost += np.dot(Mk, np.dot(W[nzc, :], W[k, :]))
                Mk = M[k, :].toarray()  # less expensive
                nzc = Mk.nonzero()[1]
                extra_cost += np.dot(Mk[0, nzc], np.dot(W[nzc, :], W[k, :]))
            normparam = 1. / (C3 * sumVec.sum())
            J += extra_cost * normparam
            dW += M.dot(W) * 2. * normparam
        else:
            M = -1. * similarMat
            sumVec = np.sum(similarMat, axis=1)
            np.fill_diagonal(M, sumVec)
            denom = C3 * np.sum(sumVec)
            J += np.sum(np.multiply(np.dot(W, W.T), M)) / denom
            dW += np.dot(M, W) * (2. / denom)

    grad = np.concatenate(([db], dW.ravel()), axis=0)
    return (J, grad)


class PCMLC(BaseEstimator):
    """All methods are necessary for a scikit-learn estimator"""

    def __init__(self, p=1, C1=1, C2=1, C3=1, weighting=True, similarMat=None):
        """Initialisation"""

        assert C1 > 0
        assert C2 > 0
        assert C3 > 0
        assert p >= 1
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.p = p
        self.weighting = weighting
        self.similarMat = similarMat
        self.obj_func = obj_pclassification
        self.cost = []
        self.trained = False

    def fit(self, X_train, Y_train, PUMat=None):
        """
            Model fitting by optimising the objective
            - PUMat: indicator matrix for additional labels with only positive observations.
                     If it is sparse, all missing entries are considered unobserved instead of negative;
                     if it is dense, it contains only 1 and NaN entries, and NaN entries are unobserved.
        """
        opt_method = 'L-BFGS-B'  # 'BFGS' #'Newton-CG'
        options = {'disp': 1, 'maxiter': 10**5, 'maxfun': 10**5}  # 'eps': 1e-5}  # , 'iprint': 99}
        sys.stdout.write('\nC: %g, %g, %g, p: %g, weighting: %s' %
                         (self.C1, self.C2, self.C3, self.p, self.weighting))
        sys.stdout.flush()

        if PUMat is not None:
            assert PUMat.shape[0] == Y_train.shape[0]
            if issparse(PUMat):
                PUMat = PUMat.toarray()
            Y_train = np.hstack([Y_train, PUMat])

        N, D = X_train.shape
        K = Y_train.shape[1]
        # w0 = np.random.rand(K * D + 1) - 0.5  # initial guess in range [-1, 1]
        # w0 = 0.001 * np.random.randn(K * D + 1)
        # w0 = 0.001 * np.random.randn(K * D + 1)
        w0 = np.zeros(K * D + 1)
        opt = minimize(self.obj_func, w0,
                       args=(X_train, Y_train, self.p, self.C1, self.C2, self.C3, self.weighting, self.similarMat),
                       method=opt_method, jac=True, options=options)
        if opt.success is True:
            self.b = opt.x[0]
            self.W = np.reshape(opt.x[1:], (K, D))
            self.trained = True
        else:
            sys.stderr.write('Optimisation failed')
            print(opt.items())
            self.trained = False

    def fit_minibatch(self, X_train, Y_train, PUMat=None, w0=None, learning_rate=0.1, batch_size=200, n_epochs=10, verbose=0):
        """
            Model fitting by mini-batch Gradient Descent
            - PUMat: indicator matrix for additional labels with only positive observations.
                     If it is sparse, all missing entries are considered unobserved instead of negative;
                     if it is dense, it contains only 1 and NaN entries, and NaN entries are unobserved.
        """
        if PUMat is not None:
            assert PUMat.shape[0] == Y_train.shape[0]
            assert not np.logical_xor(issparse(PUMat), issparse(Y_train))
            K = Y_train.shape[1] + PUMat.shape[1]
        else:
            K = Y_train.shape[1]

        fnpy = ('mlr' if PUMat is None else 'pla') + ('-N-' if self.similarMat is None else '-Y-') + \
               '%s-%g-%g-%g-%g-latest.npy' % (self.weighting, self.C1, self.C2, self.C3, self.p)
        # np.random.seed(918273645)
        N, D = X_train.shape
        if w0 is None:
            # w = 0.001 * np.random.randn(K * D + 1)
            if os.path.exists(fnpy):
                try:
                    w = np.load(fnpy, allow_pickle=False)
                    print('restore from %s' % fnpy)
                except:
                    w = np.zeros(K * D + 1)
            else:
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
                if PUMat is not None:
                    PU = PUMat[ix]
                    if issparse(PU):
                        PU = PU.toarray().astype(np.float)
                        PU[PU == 0] = np.nan
                    Y = np.hstack([Y, PU])
                J, g = self.obj_func(w, X, Y, p=self.p, C1=self.C1, C2=self.C2, C3=self.C3, weighting=self.weighting, 
                                     similarMat=self.similarMat)
                w = w - learning_rate * g
                if np.isnan(J):
                    print('J = NaN, training failed.')
                    return
                self.cost.append(J)
            print('\nepoch: %d / %d' % (epoch+1, n_epochs))
            learning_rate *= decay
            np.save(fnpy, w, allow_pickle=False)
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
