import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from scipy.sparse import issparse


def obj_pclassification(w, X, Y, p=1, C1=1, C2=1, C3=1, loss_type='example', PU=None, user_playlist_indices=None):
    """
        Objective with L2 regularisation and p-classification loss

        Input:
            - w: current weight vector, flattened K x D + 1 (bias)
            - X: feature matrix, N x D
            - Y: label matrix,   N x K
            - p: constant for p-classification push loss
            - C1-C3: regularisation constants
            - loss_type: weighting the exponential surrogate by the #positive or #negative labels or samples,
                  valid assignment: None, 'example', 'label', 'both'
                  - None: do not weight
                  - 'example': weighting each example, by the #positive or #negative labels per example
                  - 'label': weighting each label, by the #positive or #negative samples per label
                  - 'both': equivalent to turn on both 'example' and 'label'
            - PU: positive only matrix (with additional positive labels), PU.shape[0] = Y.shape[0]
            - user_playlist_indices: a list of arrays, each array is the indices of playlists of the same user.
              To require require the parameters of label_i and label_j be similar by regularising the diff if 
              entry (i,j) is 1 (i.e. belong to the same user).
    """
    N, D = X.shape
    K = Y.shape[1]
    Yp = Y
    Yn = 1 - Y

    if PU is not None:
        assert PU.shape[0] == Y.shape[0]
        K += PU.shape[1]
        Yp = np.hstack([Yp, PU])
        Yn = np.hstack([Yn, np.zeros_like(PU, dtype=Yn.dtype)])

    assert w.shape[0] == K * D + 1
    assert p >= 1
    assert C1 > 0
    assert C2 > 0
    assert C3 > 0
    assert loss_type in [None, 'example', 'label', 'both'], \
        'Valid assignment for "loss_type" are: None, "example", "label", "both".'

    def _cost_grad(loss_type):
        if loss_type is None:
            Tp = 0
            Tn = np.log(p) * Yn
            P = Yp
            Q = Yn
            num = 1
            return Tp, Tn, P, Q, num
        if loss_type == 'label':
            ax = 0
            shape = (1, K)
        elif loss_type == 'example':
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

    W = w[1:].reshape(K, D)  # reshape weight matrix
    b = w[0]                 # bias
    T1 = np.dot(X, W.T) + b  # N by K
    T2 = np.multiply(Yp, -T1)
    T3 = np.multiply(Yn, p * T1)

    if loss_type == 'both':
        Tp1, Tn1, P1, Q1, num1 = _cost_grad(loss_type='label')
        Tp2, Tn2, P2, Q2, num2 = _cost_grad(loss_type='example')
        T5 = T2 + T3 - Tp1 - Tn1 - np.log(num1)
        T6 = T2 + T3 - Tp2 - Tn2 - np.log(num2) + np.log(C2)
        cost = logsumexp(np.concatenate([T5.ravel(), T6.ravel()], axis=-1))
        T7 = np.exp(T2 - cost)
        T8 = np.exp(T3 - cost)
        T11 = -np.multiply(P1, T7) + np.multiply(Q1, T8)
        T12 = -np.multiply(P2, T7) + np.multiply(Q2, T8)
        dW = W / C1 + np.dot(T11.T, X) / num1 + C2 * np.dot(T12.T, X) / num2
        db = np.sum(T11) / num1 + C2 * np.sum(T12) / num2
    else:
        Tp, Tn, P, Q, num = _cost_grad(loss_type=loss_type)
        cost = logsumexp((T2 + T3 - Tp - Tn).ravel()) - np.log(num)
        # cost = logsumexp((T2 + T3 - Tp - Tn - np.log(num)).ravel())
        T5 = -np.multiply(P, np.exp(T2 - cost)) + np.multiply(Q, np.exp(T3 - cost))
        dW = W / C1 + np.dot(T5.T, X) / num
        db = np.sum(T5) / num

    J = np.dot(W.ravel(), W.ravel()) * 0.5 / C1 + cost

    # Multi-task regularisation
    if user_playlist_indices is not None:
        denom = 0.
        extra_cost = 0.
        dW1 = np.zeros_like(dW)
        for pls in user_playlist_indices:
            # if len(pls) < 2: continue
            # assert len(pls) > 1  # trust the input
            npl = len(pls)
            denom += npl * (npl - 1)
            M = -1 * np.ones((npl, npl), dtype=np.float)
            np.fill_diagonal(M, npl-1)
            Wu = W[pls, :]
            extra_cost += np.multiply(M, np.dot(Wu, Wu.T)).sum()
            dW1[pls, :] = np.dot(M, Wu)  # assume one playlist belongs to only one user
        normparam = 1. / (C3 * denom)
        J += extra_cost * normparam
        dW += dW1 * 2. * normparam
        
    grad = np.r_[db, dW.ravel()]
    return (J, grad)


class PCMLC(BaseEstimator):
    """All methods are necessary for a scikit-learn estimator"""

    def __init__(self, p=1, C1=1, C2=1, C3=1, loss_type='example'):
        """Initialisation"""

        assert C1 > 0
        assert C2 > 0
        assert C3 > 0
        assert p >= 1
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.p = p
        self.loss_type = loss_type 
        self.obj_func = obj_pclassification
        self.cost = []
        self.trained = False

    def fit(self, X_train, Y_train, PUMat=None, user_playlist_indices=None):
        """
            Model fitting by optimising the objective
            - PUMat: indicator matrix for additional labels with only positive observations.
                     If it is sparse, all missing entries are considered unobserved instead of negative;
                     if it is dense, it contains only 1 and NaN entries, and NaN entries are unobserved.
        """
        sys.stdout.write('\nC: %g, %g, %g, p: %g, loss_type: %s\n' %
                         (self.C1, self.C2, self.C3, self.p, self.loss_type))
        sys.stdout.flush()

        N, D = X_train.shape
        K = Y_train.shape[1]

        if PUMat is not None:
            assert PUMat.shape[0] == Y_train.shape[0]
            K += PUMat.shape[1]
            if issparse(PUMat):
                PUMat = PUMat.toarray()

        if PUMat is None:
            w0 = np.zeros(K * D + 1)
        else:
            K1 = PUMat.shape[1]
            w0 = np.r_[np.zeros((K-K1) * D + 1), 0.001 * np.random.randn(K1 * D)]

        opt_method = 'L-BFGS-B'  # 'BFGS' #'Newton-CG'
        options = {'disp': 1, 'maxiter': 10**5, 'maxfun': 10**5}  # 'eps': 1e-5}  # , 'iprint': 99}
        opt = minimize(self.obj_func, w0,
                       args=(X_train, Y_train, self.p, self.C1, self.C2, self.C3, self.loss_type, PUMat, user_playlist_indices),
                       method=opt_method, jac=True, options=options)
        if opt.success is True:
            self.b = opt.x[0]
            self.W = np.reshape(opt.x[1:], (K, D))
            self.trained = True
        else:
            sys.stderr.write('Optimisation failed')
            print(opt.items())
            self.trained = False

    def fit_minibatch_mlr(self, X_train, Y_train, user_playlist_indices=None, w0=None,
                          learning_rate=0.001, lr_decay=0.9, batch_size=256, n_epochs=10, verbose=0):
        """
            Model fitting by mini-batch Gradient Descent.
        """
        N, D = X_train.shape
        K = Y_train.shape[1]
        fnpy = 'mlr-' + ('N' if user_playlist_indices is None else 'Y') + \
            '-%s-%g-%g-%g-%g-latest.npy' % (self.loss_type, self.C1, self.C2, self.C3, self.p)

        if w0 is None:
            try:
                w = np.load(fnpy, allow_pickle=False)
                print('Restore from %s' % fnpy)
            except (IOError, ValueError):
                w = np.zeros(K * D + 1)
        else:
            assert w0.shape[0] == K * D + 1
            w = w0

        n_batches = int((N-1) / batch_size) + 1
        alpha = learning_rate
        decay = lr_decay
        np.random.seed(91827365)
        for epoch in range(n_epochs):
            if verbose > 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S'))

            indices = np.arange(N)
            np.random.shuffle(indices)

            for nb in range(n_batches):
                if verbose > 0:
                    sys.stdout.write('%d / %d' % (nb + 1, n_batches))
                ix_start = nb * batch_size
                ix_end = min((nb + 1) * batch_size, N)
                ix = indices[ix_start:ix_end]
                X = X_train[ix]
                Y = Y_train[ix]
                if issparse(Y):
                    Y = Y.toarray().astype(np.bool)

                J, dw = self.obj_func(w=w, X=X, Y=Y, p=self.p, C1=self.C1, C2=self.C2, C3=self.C3,
                                      loss_type=self.loss_type, user_playlist_indices=user_playlist_indices)
                w -= alpha * dw 

                if (nb + 1) % 30 == 0:
                    alpha *= decay

                if np.isnan(J) or np.isinf(J):
                    print('\nJ = NaN or INF, training failed.')
                    return

                self.cost.append(J)
                if verbose > 0:
                    print(' | alpha: %.6f, |dw|: %.6f, objective: %.6f' % (alpha, np.sqrt(np.dot(dw, dw)), J))
                    sys.stdout.flush()

            print('\nepoch: %d / %d' % (epoch + 1, n_epochs))
            alpha *= decay
            if verbose > 1:
                try:
                    np.save(fnpy, w, allow_pickle=False)
                except (OSError, IOError, ValueError):
                    sys.stderr.write('Save params to .npy file failed\n')

        self.b = w[0]
        self.W = np.reshape(w[1:], (K, D))
        self.trained = True


    def fit_minibatch_pla(self, X_train, Y_train, PUMat, user_playlist_indices=None, w0=None,
                      learning_rate=0.001, lr_decay=0.9, batch_size=256, n_epochs=10, verbose=0):
        """
            Model fitting by mini-batch Gradient Descent.
            - PUMat: indicator matrix for additional labels with only positive observations.
                     If it is sparse, all missing entries are considered unobserved instead of negative;
                     if it is dense, it contains only 1 and NaN entries, and NaN entries are unobserved.
            First consume batches without labels in PUMat, then consume batches with positive labels in PUMat. 
            The former does not touch the weights corresponding to PUMat, and is the same as fit_minibatch_mlr().
        """
        assert PUMat.shape[0] == Y_train.shape[0] == X_train.shape[0]
        assert not np.logical_xor(issparse(PUMat), issparse(Y_train))
        D = X_train.shape[1]
        K1 = Y_train.shape[1]
        K2 = PUMat.shape[1]
        fnpy = 'pla-' + ('N' if user_playlist_indices is None else 'Y') + \
            '-%s-%g-%g-%g-%g-latest.npy' % (self.loss_type, self.C1, self.C2, self.C3, self.p)

        cliques_train = None
        if user_playlist_indices is not None:
            cliques_train = []
            for clq in user_playlist_indices:
                cliques_train.append(clq[clq < Y_train.shape[1]])

        if w0 is None:
            try:
                w = np.load(fnpy, allow_pickle=False)
                print('Restore from %s' % fnpy)
            except (IOError, ValueError):
                w = np.r_[np.zeros(K1 * D + 1), 0.001 * np.random.randn(K2 * D)]
        else:
            assert w0.shape[0] == (K1 + K2) * D + 1
            w = w0

        pusum = np.asarray(PUMat.sum(axis=1)).reshape(-1)
        prows = []
        urows = []
        for row in range(PUMat.shape[0]):
            if pusum[row] > 0:
                prows.append(row)
            else:
                urows.append(row)
        nbu = int((len(urows)-1) / batch_size) + 1
        nbp = int((len(prows)-1) / batch_size) + 1
        n_batches = nbp + nbu
        if verbose > 0:
            print('U batches: %d, P batches: %d' % (nbu, nbp))

        alpha = learning_rate
        decay = lr_decay
        np.random.seed(91827365)
        for epoch in range(n_epochs):
            if verbose > 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S'))

            uindices = np.random.permutation(urows)
            pindices = np.random.permutation(prows)
            
            for nb in range(n_batches):
                if verbose > 0:
                    sys.stdout.write('%d / %d' % (nb + 1, n_batches))
                
                # first consume batches without labels in PUMat then batches with positive labels
                PU = None
                if nb < nbu:
                    ix_start = nb * batch_size
                    ix_end = min((nb + 1) * batch_size, len(urows))
                    ix = uindices[ix_start:ix_end]
                else:
                    ix_start = (nb - nbu) * batch_size
                    ix_end = min((nb - nbu + 1) * batch_size, len(prows))
                    ix = pindices[ix_start:ix_end]
                    PU = PUMat[ix]
                X = X_train[ix]
                Y = Y_train[ix]
                if issparse(Y):
                    Y = Y.toarray().astype(np.bool)
                if PU is not None and issparse(PU):
                    PU = PU.toarray().astype(np.bool)

                nparam = K1 * D + 1 if PU is None else (K1 + K2) * D + 1
                cliques = cliques_train if PU is None else user_playlist_indices
                J, dw = self.obj_func(w=w[:nparam], X=X, Y=Y, p=self.p, C1=self.C1, C2=self.C2, C3=self.C3, PU=PU,
                                      loss_type=self.loss_type, user_playlist_indices=cliques)
                assert len(dw) == nparam
                w[:nparam] -= alpha * dw
                
                if (nb + 1) % 30 == 0:
                    alpha *= decay

                if np.isnan(J) or np.isinf(J):
                    print('\nJ = NaN or INF, training failed.')
                    return
                self.cost.append(J)
                if verbose > 0:
                    print(' | alpha: %.6f, |dw|: %.6f, objective: %.6f' % (alpha, np.sqrt(np.dot(dw, dw)), J))
                    sys.stdout.flush()
            print('\nepoch: %d / %d' % (epoch + 1, n_epochs))
            alpha *= decay
            if verbose > 1:
                try:
                    np.save(fnpy, w, allow_pickle=False)
                except (OSError, IOError, ValueError):
                    sys.stderr.write('Save params to .npy file failed\n')
        self.b = w[0]
        self.W = np.reshape(w[1:], ((K1 + K2), D))
        self.trained = True

    def fit_minibatch_pla2(self, X_train, Y_train, PUMat, user_playlist_indices=None, w0=None,
                      learning_rate=0.001, lr_decay=0.95, batch_size=256, n_epochs=10, verbose=0):
        """
            Model fitting by mini-batch Gradient Descent.
            - PUMat: indicator matrix for additional labels with only positive observations.
                     If it is sparse, all missing entries are considered unobserved instead of negative;
                     if it is dense, it contains only 1 and NaN entries, and NaN entries are unobserved.
            First consume batches without labels in PUMat, then consume batches with positive labels in PUMat. 
            The former does not touch the weights corresponding to PUMat, and is the same as fit_minibatch_mlr().
        """
        assert PUMat.shape[0] == Y_train.shape[0] == X_train.shape[0]
        assert not np.logical_xor(issparse(PUMat), issparse(Y_train))
        D = X_train.shape[1]
        K1 = Y_train.shape[1]
        K2 = PUMat.shape[1]
        fnpy = 'pla-' + ('N' if user_playlist_indices is None else 'Y') + \
            '-%s-%g-%g-%g-%g-latest.npy' % (self.loss_type, self.C1, self.C2, self.C3, self.p)

        cliques_train = None
        if user_playlist_indices is not None:
            cliques_train = []
            for clq in user_playlist_indices:
                cliques_train.append(clq[clq < Y_train.shape[1]])

        if w0 is None:
            try:
                w = np.load(fnpy, allow_pickle=False)
                print('Restore from %s' % fnpy)
            except (IOError, ValueError):
                w = np.r_[np.zeros(K1 * D + 1), 0.001 * np.random.randn(K2 * D)]
        else:
            assert w0.shape[0] == (K1 + K2) * D + 1
            w = w0

        pusum = np.asarray(PUMat.sum(axis=1)).reshape(-1)
        prows = []
        urows = []
        for row in range(PUMat.shape[0]):
            if pusum[row] > 0:
                prows.append(row)
            else:
                urows.append(row)
        nbu = int((len(urows)-1) / batch_size) + 1
        nbp = int((len(prows)-1) / batch_size) + 1
        n_batches = nbp + nbu
        if verbose > 0:
            print('U batches: %d, P batches: %d' % (nbu, nbp))

        alpha = learning_rate
        decay = lr_decay
        np.random.seed(91827365)
        for epoch in range(n_epochs):
            if verbose > 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S'))

            uindices = np.random.permutation(urows)
            pindices = np.random.permutation(prows)
            
            for nb in range(n_batches):
                if verbose > 0:
                    sys.stdout.write('%d / %d' % (nb + 1, n_batches))
                
                # first consume batches without labels in PUMat then batches with positive labels
                PU = None
                if nb < nbp:
                    ix_start = nb * batch_size
                    ix_end = min((nb + 1) * batch_size, len(prows))
                    ix = pindices[ix_start:ix_end]
                    PU = PUMat[ix]
                else:
                    ix_start = (nb - nbp) * batch_size
                    ix_end = min((nb - nbp + 1) * batch_size, len(urows))
                    ix = uindices[ix_start:ix_end]
                X = X_train[ix]
                Y = Y_train[ix]
                if issparse(Y):
                    Y = Y.toarray().astype(np.bool)
                if PU is not None and issparse(PU):
                    PU = PU.toarray().astype(np.bool)

                nparam = K1 * D + 1 if PU is None else (K1 + K2) * D + 1
                cliques = cliques_train if PU is None else user_playlist_indices
                J, dw = self.obj_func(w=w[:nparam], X=X, Y=Y, p=self.p, C1=self.C1, C2=self.C2, C3=self.C3, PU=PU,
                                      loss_type=self.loss_type, user_playlist_indices=cliques)
                assert len(dw) == nparam
                w[:nparam] -= alpha * dw
                
                if (nb + 1) % 10 == 0:
                    alpha *= decay

                if np.isnan(J) or np.isinf(J):
                    print('\nJ = NaN or INF, training failed.')
                    return
                self.cost.append(J)
                if verbose > 0:
                    print(' | alpha: %.6f, |dw|: %.6f, objective: %.6f' % (alpha, np.sqrt(np.dot(dw, dw)), J))
                    sys.stdout.flush()
            print('\nepoch: %d / %d' % (epoch + 1, n_epochs))
            alpha *= decay
            if verbose > 1:
                try:
                    np.save(fnpy, w, allow_pickle=False)
                except (OSError, IOError, ValueError):
                    sys.stderr.write('Save params to .npy file failed\n')
        self.b = w[0]
        self.W = np.reshape(w[1:], ((K1 + K2), D))
        self.trained = True

    def fit_minibatch_pla3(self, X_train, Y_train, PUMat, user_playlist_indices=None, w0=None,
                      learning_rate=0.001, lr_decay=0.9, batch_size=256, n_epochs=10, verbose=0):
        """
            Model fitting by mini-batch Gradient Descent.
            - PUMat: indicator matrix for additional labels with only positive observations.
                     If it is sparse, all missing entries are considered unobserved instead of negative;
                     if it is dense, it contains only 1 and NaN entries, and NaN entries are unobserved.
            First consume batches without labels in PUMat, then consume batches with positive labels in PUMat. 
            The former does not touch the weights corresponding to PUMat, and is the same as fit_minibatch_mlr().
        """
        assert PUMat.shape[0] == Y_train.shape[0] == X_train.shape[0]
        assert not np.logical_xor(issparse(PUMat), issparse(Y_train))
        D = X_train.shape[1]
        K1 = Y_train.shape[1]
        K2 = PUMat.shape[1]
        fnpy = 'pla-' + ('N' if user_playlist_indices is None else 'Y') + \
            '-%s-%g-%g-%g-%g-latest.npy' % (self.loss_type, self.C1, self.C2, self.C3, self.p)

        cliques_train = None
        if user_playlist_indices is not None:
            cliques_train = []
            for clq in user_playlist_indices:
                cliques_train.append(clq[clq < Y_train.shape[1]])

        if w0 is None:
            try:
                w = np.load(fnpy, allow_pickle=False)
                print('Restore from %s' % fnpy)
            except (IOError, ValueError):
                w = np.r_[np.zeros(K1 * D + 1), 0.001 * np.random.randn(K2 * D)]
        else:
            assert w0.shape[0] == (K1 + K2) * D + 1
            w = w0

        pusum = np.asarray(PUMat.sum(axis=1)).reshape(-1)
        prows = []
        urows = []
        for row in range(PUMat.shape[0]):
            if pusum[row] > 0:
                prows.append(row)
            else:
                urows.append(row)
        nbu = int((len(urows)-1) / batch_size) + 1
        nbp = int((len(prows)-1) / batch_size) + 1
        n_batches = nbp + nbu
        if verbose > 0:
            print('U batches: %d, P batches: %d' % (nbu, nbp))

        alpha = learning_rate
        np.random.seed(91827365)
        for epoch in range(n_epochs):
            if verbose > 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S'))

            uindices = np.random.permutation(urows)
            
            for nb in range(nbu):
                if verbose > 0:
                    sys.stdout.write('%d / %d' % (nb + 1, nbu))
                
                ix_start = nb * batch_size
                ix_end = min((nb + 1) * batch_size, len(urows))
                ix = uindices[ix_start:ix_end]
                PU = None
                X = X_train[ix]
                Y = Y_train[ix]
                if issparse(Y):
                    Y = Y.toarray().astype(np.bool)

                nparam = K1 * D + 1 if PU is None else (K1 + K2) * D + 1
                cliques = cliques_train if PU is None else user_playlist_indices
                J, dw = self.obj_func(w=w[:nparam], X=X, Y=Y, p=self.p, C1=self.C1, C2=self.C2, C3=self.C3, PU=PU,
                                      loss_type=self.loss_type, user_playlist_indices=cliques)
                assert len(dw) == nparam
                w[:nparam] -= alpha * dw
                
                if (nb + 1) % 10 == 0:
                    alpha *= lr_decay

                if np.isnan(J) or np.isinf(J):
                    print('\nJ = NaN or INF, training failed.')
                    return
                self.cost.append(J)
                if verbose > 0:
                    print(' | alpha: %.6f, |dw|: %.6f, objective: %.6f' % (alpha, np.sqrt(np.dot(dw, dw)), J))
                    sys.stdout.flush()
            print('\nepoch: %d / %d' % (epoch + 1, n_epochs))
            alpha *= lr_decay

        alpha = learning_rate
        for epoch in range(n_epochs):
            if verbose > 0:
                print(time.strftime('%Y-%m-%d %H:%M:%S'))

            pindices = np.random.permutation(prows)
            
            for nb in range(nbp):
                if verbose > 0:
                    sys.stdout.write('%d / %d' % (nb + 1, n_batches))
                
                ix_start = nb * batch_size
                ix_end = min((nb + 1) * batch_size, len(prows))
                ix = pindices[ix_start:ix_end]
                PU = PUMat[ix]
                X = X_train[ix]
                Y = Y_train[ix]
                if issparse(Y):
                    Y = Y.toarray().astype(np.bool)
                if PU is not None and issparse(PU):
                    PU = PU.toarray().astype(np.bool)

                nparam = K1 * D + 1 if PU is None else (K1 + K2) * D + 1
                cliques = cliques_train if PU is None else user_playlist_indices
                J, dw = self.obj_func(w=w[:nparam], X=X, Y=Y, p=self.p, C1=self.C1, C2=self.C2, C3=self.C3, PU=PU,
                                      loss_type=self.loss_type, user_playlist_indices=cliques)
                assert len(dw) == nparam
                w[:nparam] -= alpha * dw
                
                if (nb + 1) % 10 == 0:
                    alpha *= lr_decay

                if np.isnan(J) or np.isinf(J):
                    print('\nJ = NaN or INF, training failed.')
                    return
                self.cost.append(J)
                if verbose > 0:
                    print(' | alpha: %.6f, |dw|: %.6f, objective: %.6f' % (alpha, np.sqrt(np.dot(dw, dw)), J))
                    sys.stdout.flush()
            print('\nepoch: %d / %d' % (epoch + 1, n_epochs))
            alpha *= lr_decay
 
            if verbose > 1:
                try:
                    np.save(fnpy, w, allow_pickle=False)
                except (OSError, IOError, ValueError):
                    sys.stderr.write('Save params to .npy file failed\n')
        self.b = w[0]
        self.W = np.reshape(w[1:], ((K1 + K2), D))
        self.trained = True

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
