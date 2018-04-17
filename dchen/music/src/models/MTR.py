import sys
import time
import numpy as np
from scipy.sparse import isspmatrix_coo, isspmatrix_csc
import cyipopt


def risk_primal(mu, v, Wu, xiu, X, Yu, Pu, N, U, C):
    assert N > 0
    assert U > 0
    assert C > 0
    assert Yu.dtype == np.bool
    assert isspmatrix_coo(Yu)  # scipy.sparse.coo_matrix type
    M, D = X.shape
    Nu = Yu.shape[1]
    assert v.shape == mu.shape == (D,)
    assert Wu.shape == (Nu, D)
    assert xiu.shape == Pu.shape == (Nu,)

    Wt = Wu + (v + mu).reshape(1, D)
    T1 = xiu.reshape(1, Nu) - np.dot(X, Wt.T)
    T1p = np.zeros(T1.shape)
    T1p[Yu.row, Yu.col] = T1[Yu.row, Yu.col]

    T2p = np.exp(T1p)
    T2 = np.zeros(T1.shape)
    T2[Yu.row, Yu.col] = T2p[Yu.row, Yu.col]
    T2 *= Pu

    risk = np.dot(v, v) / U + np.sum([np.dot(Wu[k, :], Wu[k, :]) for k in range(Nu)]) / N
    risk = risk * C / 2 + T2.sum()

    if np.isnan(risk) or np.isinf(risk):
        sys.stderr.write('risk_pclassification(): risk is NaN or inf!\n')
        sys.exit(0)
    return risk


def grad_primal(mu, v, Wu, xiu, X, Yu, Pu):
    assert Yu.dtype == np.bool
    assert isspmatrix_coo(Yu)  # scipy.sparse.coo_matrix type
    M, D = X.shape
    Nu = Yu.shape[1]
    assert v.shape == mu.shape == (D,)
    assert Wu.shape == (Nu, D)
    assert xiu.shape == Pu.shape == (Nu,)

    Wt = Wu + (v + mu).reshape(1, D)
    T1 = xiu.reshape(1, Nu) - np.dot(X, Wt.T)
    T1p = np.zeros(T1.shape)
    T1p[Yu.row, Yu.col] = T1[Yu.row, Yu.col]

    T2p = np.exp(T1p)
    T2 = np.zeros(T1.shape)
    T2[Yu.row, Yu.col] = T2p[Yu.row, Yu.col]
    T2 *= Pu

    T3 = np.dot(T2.T, X)
    dW = T3
    dv = T3.sum(axis=0)
    dmu = dv
    dxi = T2.sum(axis=0)

    return dmu, dv, dW, dxi


class DataHelper:
    """
        SciPy sparse matrix slicing is slow, as stated here:
        https://stackoverflow.com/questions/42127046/fast-slicing-and-multiplication-of-scipy-sparse-csr-matrix
        Profiling confirms this inefficient slicing.
        This iterator aims to do slicing only once and cache the results.
    """
    def __init__(self, Y, cliques):
        assert isspmatrix_csc(Y)
        M, N = Y.shape
        assert np.all(np.arange(N) == np.asarray(sorted([k for clq in cliques for k in clq])))
        U = len(cliques)
        self.init = False
        self.Ys = []
        self.Ps = []
        Mplus = Y.sum(axis=0).A.reshape(-1)
        P = 1. / (N * Mplus)
        for u in range(U):
            clq = cliques[u]
            Yu = Y[:, clq]
            self.Ys.append(Yu.tocoo())
            self.Ps.append(P[clq])
        self.init = True

    def get_data(self):
        assert self.init is True
        return self.Ys, self.Ps


class MTR(cyipopt.problem):
    """Multitask ranking"""

    def __init__(self, X_train, Y_train, C, user_playlist_indices):
        assert isspmatrix_csc(Y_train)
        assert C > 0
        self.X, self.Y = X_train, Y_train
        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        assert self.M == self.Y.shape[0]
        self.C = C

        self.cliques = user_playlist_indices
        self.U = len(self.cliques)
        self.u2pl = self.cliques
        self.pl2u = np.zeros(self.N, dtype=np.int32)
        for u in range(self.U):
            clq = self.cliques[u]
            self.pl2u[clq] = u

        self.data_helper = DataHelper(self.Y, self.cliques)

        # self.constraints:
        # - a list of N lists
        # - let n = self.constraints[k][j], then x_n violates constraint f_{k,n} <= 0
        # - it will be updated by self.update_constraints() given model parameters
        # - self.constraints(), self.jacobianstructure() and self.jacobian() depend on this data structure
        self.constraints = [list() for _ in range(self.N)]
        self.all_constraints_satisfied = False
        self.n_constraints = 0

    def fit(self, w0=None, verbose=0, fnpy='_'):
        N, U, D = self.N, self.U, self.D

        if verbose > 0:
            t0 = time.time()

        if verbose > 0:
            print('\nC: %g' % self.C)

        if w0 is not None:
            assert w0.shape == ((U + N + 1) * D + N,)
        else:
            if fnpy is not None:
                try:
                    w0 = np.load(fnpy, allow_pickle=False)
                    assert w0.shape == ((U + N + 1) * D + N,)
                    print('Restore from %s' % fnpy)
                except (IOError, ValueError):
                    w0 = 0.001 * np.random.randn((U + N + 1) * D + N)
        self.n_vars = (U + N + 1) * D + N
        LB = np.full(self.n_vars, -1e3, dtype=np.float)
        UB = np.full(self.n_vars,  1e3, dtype=np.float)

        # solve using IPOPT and cutting-plane method
        n_cp_iter = 1
        w = w0
        self.update_constraints(w)
        while self.all_constraints_satisfied is False:
            print('\nThe %3d-th iteration of cutting plane\n' % n_cp_iter)
            # create an optimisation problem (same objective, updated constraints) and solve it
            super(MTR, self).__init__(
                n=self.n_vars,            # number of variables
                m=self.n_constraints,     # number of constraints
                lb=LB,                    # lower bounds on variables
                ub=UB,                    # upper bounds on variables
                # cl=np.zeros(N),         # lower bounds on constraints
                cu=np.zeros(self.n_vars)  # upper bounds on constraints
            )

            # Set solver options, https://www.coin-or.org/Ipopt/documentation/node51.html
            self.addOption(b'mu_strategy', b'adaptive')
            self.addOption(b'max_iter', int(1e4))
            self.addOption(b'tol', 1e-7)
            self.addOption(b'acceptable_tol', 1e-5)
            self.addOption(b'linear_solver', b'ma57')
            # self.addOption(b'derivative_test', b'first-order')
            # self.addOption(b'acceptable_constr_viol_tol', 1e-6)
            w, info = super(MTR, self).solve(w)
            print(info['status'], info['status_msg'])
            self.update_constraints(w)
            n_cp_iter += 1

        assert self.all_constraints_satisfied is True
        self.mu = w[:D]
        self.V = w[D:(U + 1) * D].reshape(U, D)
        self.W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        self.xi = w[(U + N + 1) * D:]
        self.trained = True

        if self.verbose > 0:
            print('Training finished in %.1f seconds' % (time.time() - t0))

    def objective(self, w):
        """
            The callback for calculating the objective
        """
        t0 = time.time()
        N, D, C, U = self.N, self.D, self.C, self.U
        assert w.shape == ((U + N + 1) * D + N,)
        mu = w[:D]
        V = w[D:(U + 1) * D].reshape(U, D)
        W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        xi = w[(U + N + 1) * D:]
        assert xi.shape == (N,)

        Ys, Ps = self.data_helper.get_data()
        assert U == len(Ys) == len(Ps)

        J = np.dot(mu, mu) * C / 2
        for u in range(U):
            clq = self.cliques[u]
            risk = risk_primal(mu, V[u, :], W[clq, :], xi[clq], self.X, Ys[u], Ps[u], N, U, C)
            J += risk

        if self.verbose > 0:
            print('Eval f: %.1f seconds used.' % (time.time() - t0))

        return J

    def gradient(self, w):
        """
            The callback for calculating the gradient
        """
        t0 = time.time()
        N, D, C, U = self.N, self.D, self.C, self.U
        assert w.shape == ((U + N + 1) * D + N,)
        mu = w[:D]
        V = w[D:(U + 1) * D].reshape(U, D)
        W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        xi = w[(U + N + 1) * D:]
        assert xi.shape == (N,)

        Ys, Ps = self.data_helper.get_data()
        assert U == len(Ys) == len(Ps)

        dV_slices = []
        dW_slices = []
        dmu = C * mu
        dxi_slices = []
        for u in range(U):
            clq = self.cliques[u]
            results = grad_primal(mu, V[u, :], W[clq, :], xi[clq], self.X, Ys[u], Ps[u])
            dmu += results[0]
            dV_slices.append(results[1])
            dW_slices.append(results[2])
            dxi_slices.append(results[3])
        dV = V * C / U + np.vstack(dV_slices)
        dW = W * C / N + np.vstack(dW_slices)
        dxi = np.concatenate(dxi_slices, axis=-1)

        if self.verbose > 0:
            print('Eval g: %.1f seconds used.' % (time.time() - t0))

        return np.r_[dmu, dV.ravel(), dW.ravel(), dxi]

    def constraints(self, w):
        """
            The callback for calculating the (function value of) constraints
        """
        N, D, U = self.N, self.D, self.U
        assert w.shape == ((U + N + 1) * D + N,)
        mu = w[:D]
        V = w[D:(U + 1) * D].reshape(U, D)
        W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        xi = w[(U + N + 1) * D:]
        assert xi.shape == (N,)

        cons_values = []
        for k in range(N):
            u = self.pl2u[k]
            v = V[u, :]
            wk = W[k, :]
            if len(self.constraints[k]) > 0:
                ix = self.constraints[k]
                values = self.X[ix, :].dot(v + wk + mu)
                assert values.shape == (len(ix),)
                cons_values += values.tolist()

        return np.asarray(cons_values)

    def jacobianstructure(self):
        """
            The sparse structure (i.e., rows, cols) of the Jacobian matrix
        """
        N, D = self.N, self.D
        rows = []
        cols = []

        ix = 0
        for k in range(N):
            u = self.pl2u[k]
            if len(self.constraints[k]) > 0:
                for n in self.constraints[k]:
                    # indices of derivatives of constraint f_{k, n} <= 0
                    # dmu: D
                    # rows += np.full(D, ix, dtype=np.int32).tolist()
                    cols += list(range(D))

                    # dV: U by D
                    # rows += np.full(D, ix, dtype=np.int32).tolist()
                    cols += list(range((u + 1) * D, (u + 2) * D))

                    # dW: N by D
                    # rows += np.full(D, ix, dtype=np.int32).tolist()
                    cols += list(range((u + 2 + k) * D, (u + 3 + k) * D))

                    # dxi: N
                    # rows.append(ix)
                    cols.append((u + 3 + k) * D + k)
                    rows += np.full(3 * D + 1, ix, dtype=np.int32).tolist()
                    ix += 1
        return np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)

    def jacobian(self, w):
        """
            The callback for calculating the Jacobian of constraints
        """
        N = self.N
        X = self.X
        jac = []

        for k in range(N):
            if len(self.constraints[k]) > 0:
                for n in self.constraints[k]:
                    # derivatives of constraints f_{k, n} <= 0
                    # dmu
                    jac.append(X[n, :])

                    # dV
                    jac.append(X[n, :])

                    # dW
                    jac.append(X[n, :])

                    # dxi
                    jac.append([-1.])
        return np.concatenate(jac, axis=-1)

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm,
                     regularization_size, alpha_du, alpha_pr, ls_trials):
        """
            The intermediate callback
        """
        print('Iter %5d: %g' % (iter_count, obj_value))
        # save intermediate weights
#         fnpy = self.fnpy
#         assert type(fnpy) == str
#         if fnpy.endswith('.npy') and k > 20 and k % 10 == 0:
#             try:
#                 np.save(fnpy, x, allow_pickle=False)
#                 if verbose > 0:
#                     print('Save to %s' % fnpy)
#             except (OSError, IOError, ValueError) as err:
#                 sys.stderr.write('Save intermediate weights failed: {0}\n'.format(err))

    def update_constraints(self, w):
        N, D, U = self.N, self.D, self.U
        X = self.X
        assert w.shape == ((U + N + 1) * D + N,)
        mu = w[:D]
        V = w[D:(U + 1) * D].reshape(U, D)
        W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        xi = w[(U + N + 1) * D:]
        assert xi.shape == (N,)

        Ys, _ = self.data_helper.get_data()
        assert U == len(Ys)

        all_satisfied = True
        for u in range(U):
            clq = self.cliques[u]
            v = V[u, :]
            Wu = W[clq, :]
            Yu = Ys[u]

            Wt = Wu + (v + mu).reshape(1, D)
            T1 = np.dot(X, Wt.T)
            T1[Yu.row, Yu.col] = -np.inf  # mask entry (m,i) if y_m^i = 1
            max_ix = T1.argmax(axis=0)

            assert max_ix.shape[0] == len(clq)
            for j in range(max_ix.shape[0]):
                k = clq[j]
                row, col = max_ix[j], j
                if xi[k] < T1[row, col]:
                    all_satisfied = False
                    self.n_constraints += 1
                    self.constraints[k].append(row)
        self.all_constraints_satisfied = all_satisfied

    def predict(self, X_test):
        """Make predictions (score is a real number)"""
        assert self.trained is True, 'Cannot make prediction before training'
        U, D = self.U, self.D
        assert D == X_test.shape[1]
        preds = []
        for u in range(U):
            clq = self.cliques[u]
            Wt = self.W[clq, :] + (self.V[u, :] + self.mu).reshape(1, D)
            preds.append(np.dot(X_test, Wt.T))
        return np.hstack(preds)
