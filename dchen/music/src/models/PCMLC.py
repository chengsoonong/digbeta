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

    T1 = np.dot(X, W.T) + b  # N by K
    T2 = np.multiply(Yp, -T1)
    T3 = np.multiply(Yn, p * T1)
    
    lognum = np.log(num)
    # cost = logsumexp((T2 + T3 - Tp - Tn).ravel()) - lognum
    # cost = logsumexp((T2 + T3 - Tp - Tn - lognum).ravel())  # the same as above
    # T5 = -np.multiply(P, np.exp(T2 - cost)) + np.multiply(Q, np.exp(T3 - cost))
    costs = (T2 + T3 - Tp - Tn).ravel().tolist()
    T5 = -np.multiply(P, np.exp(T2 - lognum)) + np.multiply(Q, np.exp(T3 - lognum))
    dW = np.dot(T5.T, X)
    db = np.sum(T5)
    return costs, db, dW


def objective(w, dw, X, Y, C1=1, C2=1, C3=1, p=1, PU=None, cliques=None, loss_type='example', batch_size=256):
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
        
        N, D = X.shape
        K = Y.shape[1]
        if PU is not None:
            assert PU.shape[0] == N
            K += PU.shape[1]
        assert w.shape[0] == K * D + 1
        b = w[0]
        W = w[1:].reshape(K, D)
        
        def _accumulate_example_loss(Wt, bt):
            bs = N if batch_size > N else batch_size
            n_batches = int((N-1) / bs) + 1 
            risks = []
            db = 0.
            dW = np.zeros_like(Wt)
            for nb in range(n_batches):  
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
                costsi, dbi, dWi = risk_pclassification(Wt, bt, Xi, Yi_pos, Yi_neg, N, K, p=p, loss_type='example')
                risks += costsi
                db += dbi
                dW += dWi
            return risks, db, dW
        
        def _accumulate_label_loss(Wt, bt, PU_only=False):
            if PU_only is True:
                assert PU is not None
                assert Wt.shape[0] == PU.shape[1]
                NUM = PU.shape[1]
            else:
                assert Wt.shape[0] == Y.shape[1]
                NUM = Y.shape[1]
            bs = NUM if batch_size > NUM else batch_size
            n_batches = int((NUM-1) / bs) + 1 
            risks = []
            db = 0.
            dW = np.zeros_like(Wt)
            indices = np.arange(NUM)
            for nb in range(n_batches):  
                ix_start = nb * bs
                ix_end = min((nb + 1) * bs, NUM)
                ix = indices[ix_start:ix_end]
                Xi = X
                Yi = PU[:, ix] if PU_only is True else Y[:, ix]
                Yi_pos = Yi.toarray().astype(np.bool) if issparse(Yi) else Yi
                Yi_neg = np.zeros_like(Yi_pos, dtype=np.bool) if PU_only is True else 1 - Yi_pos
                costsi, dbi, dWi = risk_pclassification(Wt, bt, Xi, Yi_pos, Yi_neg, N, K, p=p, loss_type='label')
                risks += costsi
                db += dbi
                dW[ix, :] = dWi
            return risks, db, dW

        def _multitask_regulariser(Wt, bt, C3=1):
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

        
        if loss_type == 'example':
            risks, db, dW = _accumulate_example_loss(W, b)
            risks = np.asarray(risks) - np.log(N)
        elif loss_type == 'label':
            if PU is None:
                risks, db, dW = _accumulate_label_loss(W, b, PU_only=False)
                risks = np.asarray(risks) - np.log(K)
            else:
                K1, K2 = Y.shape[1], PU.shape[1]
                risks1, db1, dW1 = _accumulate_label_loss(W[:K1, :], b, PU_only=False)
                risks2, db2, dW2 = _accumulate_label_loss(W[K1:, :], b, PU_only=True)
                risks = np.asarray(risks1 + risks2) - np.log(K1 + K2)
                db = db1 + db2
                dW = np.vstack([dW1, dW2])
        else:
            assert loss_type == 'both'
            if PU is None:
                risks, db, dW = _accumulate_label_loss(W, b, PU_only=False)
                risks = np.asarray(risks) - np.log(K)
            else:
                K1, K2 = Y.shape[1], PU.shape[1]
                risks1, db1, dW1 = _accumulate_label_loss(W[:K1, :], b, PU_only=False)
                risks2, db2, dW2 = _accumulate_label_loss(W[K1:, :], b, PU_only=True)
                risks = np.asarray(risks1 + risks2) - np.log(K1 + K2)
                db = db1 + db2
                dW = np.vstack([dW1, dW2])
            risks3, db3, dW3 = _accumulate_example_loss(W, b)
            risks = np.r_[risks, np.asarray(risks3) - np.log(N) + np.log(C2)]
            db += C2 * db3
            dW += C2 * dW3
        J = logsumexp(risks)
        denom = np.exp(J)
        db /= denom
        dW = W / C1 + dW / denom
        J += np.dot(W.ravel(), W.ravel()) * 0.5 / C1
        
        if cliques is not None:
            cost_mt, dW_mt = _multitask_regulariser(W, b, C3=C3)
            J += cost_mt
            dW += dW_mt
        
        dw[:] = np.r_[db, dW.ravel()]
        return J


def progress(x, g, f_x, xnorm, gnorm, step, k, ls):
    """Report optimization progress."""
    print("x = %g  |  f(x) = %g  |  f'(x) = %g" % (x, f_x, g))
    
        
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

    def fit(self, X_train, Y_train, PUMat=None, user_playlist_indices=None, w0=None, batch_size=256, verbose=0):
        """
            Model fitting by mini-batch Gradient Descent.
            - PUMat: indicator matrix for additional labels with only positive observations.
                     If it is sparse, all missing entries are considered unobserved instead of negative;
                     if it is dense, it contains only 1 and NaN entries, and NaN entries are unobserved.
            First consume batches without labels in PUMat, then consume batches with positive labels in PUMat. 
            The former does not touch the weights corresponding to PUMat, and is the same as fit_minibatch_mlr().
        """
        assert X_train.shape[0] == Y_train.shape[0]
        N, D = X_train.shape
        K = Y_train.shape[1]
        if PUMat is not None:
            assert PUMat.shape[0] == Y_train.shape[0]
            assert not np.logical_xor(issparse(PUMat), issparse(Y_train))
            K += PUMat.shape[1]
        if w0 is None:
            try:
                w0 = np.load(fnpy, allow_pickle=False)
                print('Restore from %s' % fnpy)
            except (IOError, ValueError):
                w0 = np.zeros(K * D + 1)
        else:
            assert w0.shape[0] == K * D + 1
        try:
            # fmin_lbfgs(f, x0, progress=None, args=())
            # f: callable(x, g, *args)
            # objective(w, dw, X, Y, C1=1, C2=1, C3=1, p=1, PU=None, cliques=None, loss_type='example', batch_size=256)
            res = fmin_lbfgs(objective, w0, progress=progress if verbose > 0 else None, 
                             args=(X_train, Y_train, self.C1, self.C2, self.C3, self.p, 
                                   PUMat, user_playlist_indices, self.loss_type, batch_size))
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
