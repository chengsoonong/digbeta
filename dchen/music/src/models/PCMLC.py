import sys
import time
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from scipy.sparse import issparse
from lbfgs import fmin_lbfgs, LBFGSError


def risk_pclassification(W, b, X, Y_pos, Y_neg, N_all, K_all, p=1, loss_type='example'):
    """
        Empirical risk of p-classification loss for multilabel classification

        Input:
            - W: current weight matrix, K by D
            - b: current bias
            - X: feature matrix, N x D
            - Y_pos: positive label matrix, N x K
            - Y_neg: negative label matrix, N x K
            - N_all: total number of examples in training set
            - K_all: total number of labels in training set
            - p: constant for p-classification push loss
            - loss_type: valid assignment is 'example' or 'label'
                - 'example': compute a loss for each example, by the #positive or #negative labels per example
                - 'label'  : compute a loss for each label, by the #positive or #negative examples per label
                
        Output:
            - cost: empirical cost
            - db  : gradient of bias term
            - dW  : gradients of weights
    """
    assert p > 0
    assert N_all > 0
    assert K_all > 0
    assert Y_pos.shape == Y_neg.shape
    assert loss_type in ['example', 'label'], \
        'Valid assignment for "loss_type" are: "example", "label"'
    N, D = X.shape
    K = Y_pos.shape[1]
    Yp = Y_pos
    Yn = Y_neg
    assert W.shape == (K, D)
    if loss_type == 'example':
        assert K == K_all
        ax = 1
        shape = (N, 1)
        num = N_all
    else:
        assert loss_type == 'label'
        assert N == N_all
        ax = 0
        shape = (1, K)
        num = K_all
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
    P = P.reshape(shape)
    Q = Q.reshape(shape)

    T1 = np.dot(X, W.T) + b
    T1p = np.multiply(Yp, T1)
    T1n = np.multiply(Yn, T1)
    T2 = np.multiply(Yp, np.exp(-T1p)) * P
    T3 = np.multiply(Yn, np.exp(p * T1n)) * Q

    cost = np.sum(T2 + T3 / p) / num
    db = np.sum(-T2 + T3) / num
    dW = np.dot((-T2 + T3).T, X) / num
    return cost, db, dW

def accumulate_example_loss(Wt, bt, X, Y, p, bs, PU=None, verbose=0):
    N, D = X.shape
    K = Y.shape[1] + (0 if PU is None else PU.shape[1])
    assert Wt.shape == (K, D)
    bs = N if bs > N else bs  # batch_size
    n_batches = int((N-1) / bs) + 1 
    risks = []
    db = 0.
    dW = np.zeros_like(Wt)
    for nb in range(n_batches):
        if verbose > 1:
            sys.stdout.write('\r%d / %d' % (nb+1, n_batches))
            sys.stdout.flush()
        ix_start = nb * bs
        ix_end = min((nb + 1) * bs, N)
        Xi = X[ix_start:ix_end, :]
        Yi = Y[ix_start:ix_end, :]
        Yi_pos = Yi.toarray().astype(np.bool) if issparse(Yi) else Yi
        Yi_neg = 1 - Yi_pos
        if PU is not None:
            PUi = PU[ix_start:ix_end, :]
            PUi = PUi.toarray().astype(np.bool) if issparse(PU) else PUi
            Yi_pos = np.hstack([Yi_pos, PUi])
            Yi_neg = np.hstack([Yi_neg, np.zeros_like(PUi, dtype=np.bool)])
        costi, dbi, dWi = risk_pclassification(Wt, bt, Xi, Yi_pos, Yi_neg, N, K, p=p, loss_type='example')
        risks.append(costi)
        db += dbi
        dW += dWi
    if verbose > 1:
        print()
    return risks, db, dW

def accumulate_label_loss(Wt, bt, X, Y, p, bs, YisPU=False, verbose=0):
    N, D = X.shape
    K = Y.shape[1]
    assert Wt.shape == (K, D)
    bs = K if bs > K else bs  # batch_size
    n_batches = int((K-1) / bs) + 1 
    risks = []
    db = 0.
    dW = np.zeros_like(Wt)
    for nb in range(n_batches):
        if verbose > 1:
            sys.stdout.write('\r%d / %d' % (nb+1, n_batches))
            sys.stdout.flush()
        ix_start = nb * bs
        ix_end = min((nb + 1) * bs, K)
        Xi = X
        Yi = Y[:, ix_start:ix_end]
        Yi_pos = Yi.toarray().astype(np.bool) if issparse(Yi) else Yi
        Yi_neg = np.zeros_like(Yi_pos, dtype=np.bool) if YisPU is True else 1 - Yi_pos
        Wb = Wt[ix_start:ix_end, :]
        costi, dbi, dWi = risk_pclassification(Wb, bt, Xi, Yi_pos, Yi_neg, N, K, p=p, loss_type='label')
        assert dWi.shape == Wb.shape
        risks.append(costi)
        db += dbi
        dW[ix_start:ix_end, :] = dWi
    if verbose > 1:
        print()
    return risks, db, dW

def multitask_regulariser(Wt, bt, C3, cliques):
    assert cliques is not None
    denom = 0.
    cost_mt = 0.
    dW_mt = np.zeros_like(Wt)
    for pls in cliques:
        # if len(pls) < 2: continue
        # assert len(pls) > 1  # trust the input
        npl = len(pls)
        denom += npl * (npl - 1)
        M = -1 * np.ones((npl, npl), dtype=np.float)
        np.fill_diagonal(M, npl-1)
        Wu = Wt[pls, :]
        cost_mt += np.multiply(M, np.dot(Wu, Wu.T)).sum()
        dW_mt[pls, :] = np.dot(M, Wu)  # assume one playlist belongs to only one user
    normparam = 1. / (C3 * denom)
    cost_mt *= normparam
    dW_mt *= 2. * normparam
    return cost_mt, dW_mt

        
def objective(w, dw, X, Y, C1=1, C2=1, C3=1, p=1, PU=None, cliques=None, loss_type='example', batch_size=256, verbose=0):
        """
            - w : np.ndarray, current weights 
            - dw: np.ndarray, OUTPUT array for gradients of w
            - PU: positive only matrix (with additional positive labels), PU.shape[0] = Y.shape[0]
            - cliques: a list of arrays, each array is the indices of playlists of the same user.
              To require require the parameters of label_i and label_j be similar by regularising the diff if 
              entry (i,j) is 1 (i.e. belong to the same user).
        """
        assert loss_type in ['example', 'label', 'both']
        assert C1 > 0
        assert C2 > 0
        assert C3 > 0
        assert p > 0
        assert batch_size > 0
        t0 = time.time()
        
        N, D = X.shape
        K = Y.shape[1]
        if PU is not None:
            assert PU.shape[0] == N
            K += PU.shape[1]
        assert w.shape[0] == K * D + 1
        b = w[0]
        W = w[1:].reshape(K, D)        
        
        if loss_type == 'example':
            risks, db, dW = accumulate_example_loss(W, b, X, Y, p, batch_size, PU, verbose=verbose)
        elif loss_type == 'label':
            if PU is None:
                risks, db, dW = accumulate_label_loss(W, b, X, Y, p, batch_size, YisPU=False, verbose=verbose)
            else:
                K1, K2 = Y.shape[1], PU.shape[1]
                risks1, db1, dW1 = accumulate_label_loss(W[:K1, :], b, X, Y, p, batch_size, YisPU=False, verbose=verbose)
                risks2, db2, dW2 = accumulate_label_loss(W[K1:, :], b, X, PU, p, batch_size, YisPU=True, verbose=verbose)
                risks = risks1 + risks2
                db = db1 + db2
                dW = np.vstack([dW1, dW2])
        else:
            assert loss_type == 'both'
            if PU is None:
                risks, db, dW = accumulate_label_loss(W, b, X, Y, p, batch_size, YisPU=False, verbose=verbose)
            else:
                K1 = Y.shape[1]
                risks1, db1, dW1 = accumulate_label_loss(W[:K1, :], b, X, Y, p, batch_size, YisPU=False, verbose=verbose)
                risks2, db2, dW2 = accumulate_label_loss(W[K1:, :], b, X, PU, p, batch_size, YisPU=True, verbose=verbose)
                risks = risks1 + risks2
                db = db1 + db2
                dW = np.vstack([dW1, dW2])
            risks3, db3, dW3 = accumulate_example_loss(W, b, X, Y, p, batch_size, PU, verbose=verbose)
            risks = np.r_[risks, C2 * np.asarray(risks3)]
            db += C2 * db3
            dW += C2 * dW3
        J = np.sum(risks) + np.dot(W.ravel(), W.ravel()) * 0.5 / C1
        dW += W / C1
        
        if cliques is not None:
            cost_mt, dW_mt = multitask_regulariser(W, b, C3, cliques)
            J += cost_mt
            dW += dW_mt

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
        self.cost = []
        self.trained = False

    def fit(self, X_train, Y_train, PUMat=None, user_playlist_indices=None, batch_size=256, w0=None, rand_init=False, verbose=0):
        """
            Model fitting by mini-batch Gradient Descent.
            - PUMat: indicator matrix for additional labels with only positive observations.
                     If it is sparse, all missing entries are considered unobserved instead of negative;
                     if it is dense, it contains only 1 and NaN entries, and NaN entries are unobserved.
            First consume batches without labels in PUMat, then consume batches with positive labels in PUMat. 
            The former does not touch the weights corresponding to PUMat, and is the same as fit_minibatch_mlr().
        """
        assert X_train.shape[0] == Y_train.shape[0]
        assert PUMat is None  # do not use this param
        N, D = X_train.shape
        K = Y_train.shape[1]
        if PUMat is not None:
            assert PUMat.shape[0] == Y_train.shape[0]
            assert not np.logical_xor(issparse(PUMat), issparse(Y_train))
            K += PUMat.shape[1]
        if w0 is not None:
            assert w0.shape[0] == K * D + 1
        else:
            if rand_init is True:
                if PUMat is None:
                    w0 = 0.001 * np.random.randn(K * D + 1)
                else:
                    K1, K2 = Y_train.shape[1], PUMat.shape[1]
                    w0 = np.r_[np.zeros(K1 * D + 1), 0.001 * np.random.randn(K2 * D)]
            else:
                w0 = np.zeros(K * D + 1)
        try:
            # fmin_lbfgs(f, x0, progress=None, args=())
            # f: callable(x, g, *args)
            # objective(w, dw, X, Y, C1, C2, C3, p, PU, cliques, loss_type, batch_size, verbose)
            res = fmin_lbfgs(objective, w0, progress=progress if verbose > 0 else None, 
                             args=(X_train, Y_train, self.C1, self.C2, self.C3, self.p, PUMat, 
                                   user_playlist_indices, self.loss_type, batch_size, verbose))
            self.b = res[0]
            self.W = res[1:].reshape(K, D)
            self.trained = True
        except (LBFGSError, MemoryError) as err:
            sys.stderr.write('LBFGS failed: {0}\n'.format(err))
            sys.stderr.flush()   
        
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
