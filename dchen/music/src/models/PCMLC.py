import sys
import time
import numpy as np
from sklearn.base import BaseEstimator
from scipy.sparse import issparse
from lbfgs import fmin_lbfgs, LBFGSError  # pip install pylbfgs


def risk_pclassification(W, b, X, Y, N_all, K_all, p=1, loss_type='example'):
    """
        Empirical risk of p-classification loss for multilabel classification

        Input:
            - W: current weight matrix, K by D
            - b: current bias
            - X: feature matrix, N x D
            - Y: positive label matrix, N x K
            - N_all: total number of examples in training set
            - K_all: total number of labels in training set
            - p: constant for p-classification push loss
            - loss_type: valid assignment is 'example' or 'label'
                - 'example': compute a loss for each example, by the #positive or #negative labels per example
                - 'label'  : compute a loss for each label, by the #positive or #negative examples per label
                
        Output:
            - risk: empirical risk
            - db  : gradient of bias term
            - dW  : gradients of weights
    """
    assert p > 0
    assert N_all > 0
    assert K_all > 0
    assert Y.dtype == np.bool
    assert loss_type in ['example', 'label'], \
        'Valid assignment for "loss_type" are: "example", "label"'
    N, D = X.shape
    K = Y.shape[1]
    Yp = Y
    Yn = 1 - Y
    assert W.shape == (K, D)
    if loss_type == 'example':
        assert K == K_all
        ax = 1
        num = N_all
        shape = (N, 1)
    else:
        assert N == N_all
        ax = 0
        num = K_all
        shape = (1, K)
    numPos = np.sum(Yp, axis=ax)
    numNeg = np.sum(Yn, axis=ax)

    # deal with zeros
    # P = 1 / numPos, Q = 1 / numNeg
    nz_pix = np.nonzero(numPos)[0]
    nz_nix = np.nonzero(numNeg)[0]
    P = np.zeros_like(numPos, dtype=np.float)
    Q = np.zeros_like(numNeg, dtype=np.float)
    P[nz_pix] = 1. / numPos[nz_pix]
    Q[nz_nix] = 1. / numNeg[nz_nix]

    T1 = np.dot(X, W.T) + b
    T1p = np.multiply(Yp, T1)
    T1n = T1 - T1p
    T2 = np.multiply(Yp, np.exp(-T1p)   ) * P.reshape(shape)
    T3 = np.multiply(Yn, np.exp(p * T1n)) * Q.reshape(shape)

    risk = np.sum(T2 + T3 / p) / num
    db = np.sum(-T2 + T3) / num
    dW = np.dot((-T2 + T3).T, X) / num
    
    if np.isnan(risk) or np.isinf(risk):
        sys.stderr('risk_pclassification(): risk is NaN or inf!\n')
        sys.exit(0)

    return risk, db, dW


class DataIter:
    """
        SciPy sparse matrix slicing is slow, as stated here:
        https://stackoverflow.com/questions/42127046/fast-slicing-and-multiplication-of-scipy-sparse-csr-matrix
        Profiling confirms this inefficient slicing.
        This iterator aims to do slicing only once and cache the results.
    """
    def __init__(self, Y, ax=0, batch_size=256):
        assert ax in [0, 1]
        assert issparse(Y)
        self.ax = ax
        self.Yslices = []
        self.starts = []
        self.ends = []
        num = Y.shape[self.ax]
        bs = num if batch_size > num else batch_size
        self.n_batches = int((num-1) / bs) + 1
        Y = Y.tocsr() if self.ax == 0 else Y.tocsc()
        for nb in range(self.n_batches):
            ix_start = nb * bs
            ix_end = min((nb + 1) * bs, num)
            Yi = Y[ix_start:ix_end, :] if self.ax == 0 else Y[:, ix_start:ix_end]
            self.Yslices.append(Yi)
            self.starts.append(ix_start)
            self.ends.append(ix_end)
        self.nb = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        self.nb += 1
        if self.nb > self.n_batches:
            raise StopIteration
        i = self.nb - 1
        return self.starts[i], self.ends[i], self.Yslices[i]
    
    def reset(self):
        self.nb = 0


def accumulate_risk(Wt, bt, X, Y, p, loss, data_iter, verbose=0):
    assert loss in ['example', 'label']
    assert data_iter is not None
    assert Wt.shape == (Y.shape[1], X.shape[1])
    ax = 0 if loss == 'example' else 1
    assert data_iter.ax == ax
    risk = 0.
    db = 0.
    dW = np.zeros_like(Wt)
    data_iter.reset()
    nb = 0
    for ix_start, ix_end, Yi in data_iter:
        nb += 1
        if verbose > 1:
            sys.stdout.write('\r%d / %d' % (nb, data_iter.n_batches))
            sys.stdout.flush()
        Xi = X[ix_start:ix_end, :] if ax == 0 else X
        if issparse(Yi):
            Yi = Yi.toarray().astype(np.bool)
        Wb = Wt if ax == 0 else Wt[ix_start:ix_end, :]
        riski, dbi, dWi = risk_pclassification(Wb, bt, Xi, Yi, Y.shape[0], Y.shape[1], p=p, loss_type=loss)
        assert dWi.shape == Wb.shape
        risk += riski
        db += dbi
        if ax == 0:
            dW += dWi
        else:
            dW[ix_start:ix_end, :] = dWi
    if verbose > 1:
        print()
    return risk, db, dW


def multitask_regulariser(Wt, bt, cliques):
    assert cliques is not None
    denom = 0.
    cost_mt = 0.
    dW_mt = np.zeros_like(Wt)
    for clq in cliques:
        npl = len(clq)
        if npl < 2: 
            continue
        denom += npl * (npl - 1)
        M = -1 * np.ones((npl, npl), dtype=np.float)
        np.fill_diagonal(M, npl-1)
        Wu = Wt[clq, :]
        cost_mt += np.multiply(M, np.dot(Wu, Wu.T)).sum()
        dW_mt[clq, :] = np.dot(M, Wu)  # assume one playlist belongs to only one user
    cost_mt /= denom
    dW_mt = dW_mt * 2. / denom
    return cost_mt, dW_mt


def objective(w, dw, X, Y, C1=1, C2=1, C3=1, p=1, loss_type='example', cliques=None, data_iter_example=None, data_iter_label=None, verbose=0, fnpy=None):
        """
            - w : np.ndarray, current weights 
            - dw: np.ndarray, OUTPUT array for gradients of w
            - cliques: a list of arrays, each array is the indices of playlists of the same user.
                       To require the parameters of label_i and label_j be similar by regularising 
                       their diff if entry (i,j) is 1 (i.e. belong to the same user).
        """
        assert loss_type in ['example', 'label', 'both']
        assert C1 > 0
        assert C2 > 0
        assert C3 > 0
        assert p > 0
        t0 = time.time()
        
        N, D = X.shape
        K = Y.shape[1]
        assert w.shape[0] == K * D + 1
        b = w[0]
        W = w[1:].reshape(K, D)        
        
        if loss_type == 'both':
            assert data_iter_example is not None
            assert data_iter_label   is not None
            risk1, db1, dW1 = accumulate_risk(W, b, X, Y, p, loss='label',   data_iter=data_iter_label,   verbose=verbose)
            risk2, db2, dW2 = accumulate_risk(W, b, X, Y, p, loss='example', data_iter=data_iter_example, verbose=verbose)
            risk = risk1 + C2 * risk2
            db = db1 + C2 * db2
            dW = dW1 + C2 * dW2
        else:
            data_iter = data_iter_example if loss_type == 'example' else data_iter_label
            assert data_iter is not None
            risk, db, dW = accumulate_risk(W, b, X, Y, p, loss=loss_type, data_iter=data_iter, verbose=verbose)
            
        J = risk + np.dot(W.ravel(), W.ravel()) * 0.5 / C1
        dW += W / C1
        
        if cliques is not None:
            cost_mt, dW_mt = multitask_regulariser(W, b, cliques)
            J += cost_mt / C3
            dW += dW_mt / C3

        dw[:] = np.r_[db, dW.ravel()]  # in-place assignment
        
        if verbose > 1:
            print('Eval f, g: %.1f seconds used.' % (time.time() - t0))
        
        return J


def progress(x, g, f_x, xnorm, gnorm, step, k, ls, *args):
    """
        Report optimization progress.
        progress: callable(x, g, fx, xnorm, gnorm, step, k, num_eval, *args)
                  If not None, called at each iteration after the call to f with 
                  the current values of x, g and f(x), the L2 norms of x and g, 
                  the line search step, the iteration number, 
                  the number of evaluations at this iteration and args.
    """
    print('Iter {:3d}:  f = {:15.9f},  |g| = {:15.9f},  {}'.format(k, f_x, gnorm, time.strftime('%Y-%m-%d %H:%M:%S')))

    # save intermediate weights
    fnpy = args[-1]
    if fnpy is not None and k % 50 == 0:
        try:
            print(fnpy)
            #np.save(fnpy, x, allow_pickle=False)
        except (OSError, IOError, ValueError):
            sys.stderr.write('Save weights to .npy file failed\n')
    
        
class PCMLC(BaseEstimator):
    """All methods are necessary for a scikit-learn estimator"""

    def __init__(self, C1=1, C2=1, C3=1, p=1, loss_type='example'):
        """Initialisation"""
        assert C1 > 0
        assert C2 > 0
        assert C3 > 0
        assert p > 0
        assert loss_type in ['example', 'label', 'both'], \
            'Valid assignment for "loss_type" are: "example", "label", "both".'
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.p = p
        self.loss_type = loss_type 
        self.trained = False

    def fit(self, X_train, Y_train, user_playlist_indices=None, batch_size=256, verbose=0, w0=None, fnpy=None):
        assert X_train.shape[0] == Y_train.shape[0]
        N, D = X_train.shape
        K = Y_train.shape[1]

        if w0 is None:
            if fnpy is not None:
                try:
                    w0 = np.load(fnpy, allow_pickle=False)
                    assert w0.shape[0] == K * D + 1
                    print('Restore from %s' % fnpy)
                except (IOError, ValueError):
                    w0 = np.zeros(K * D + 1)
        else:
            assert w0.shape[0] == K * D + 1
            
        data_iter_example = None if self.loss_type == 'label' else DataIter(Y_train, ax=0, batch_size=batch_size)
        data_iter_label = None if self.loss_type == 'example' else DataIter(Y_train, ax=1, batch_size=batch_size)

        try:
            # fmin_lbfgs(f, x0, progress=None, args=())
            # f: callable(x, g, *args)
            res = fmin_lbfgs(objective, w0, progress, 
                             args=(X_train, Y_train, self.C1, self.C2, self.C3, self.p, self.loss_type, 
                                   user_playlist_indices, data_iter_example, data_iter_label, verbose, fnpy))
            self.b = res[0]
            self.W = res[1:].reshape(K, D)
            self.trained = True
        except (LBFGSError, MemoryError) as err:
            self.trained = False
            sys.stderr.write('LBFGS failed: {0}\n'.format(err))
            sys.stderr.flush()
        
    def decision_function(self, X_test):
        """Make predictions (score is a real number)"""
        assert self.trained is True, "Cannot make prediction before training"
        return np.dot(X_test, self.W.T) + self.b  # log of prediction score

    def predict(self, X_test):
        return self.decision_function(X_test)
    #    """Make predictions (score is boolean)"""
    #    preds = sigmoid(self.decision_function(X_test))
    #    return preds >= Threshold

    # inherit from BaseEstimator instead of re-implement
    # def get_params(self, deep = True):
    # def set_params(self, **params):

