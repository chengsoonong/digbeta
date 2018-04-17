import sys
import time
import numpy as np
from scipy.sparse import isspmatrix_csc
import cyipopt


class MTR(object):
    def __init__(self, X_train, Y_train, C, cliques, verbose=0):
        self.problem = RankPrimal(X=X_train, Y=Y_train, C=C, cliques=cliques, verbose=verbose)
        self.trained = False

    def fit(self, w0=None, fnpy=None):
        N, U, D = self.problem.N, self.problem.U, self.problem.D
        verbose = self.problem.verbose
        if verbose > 0:
            t0 = time.time()
            print('\nC: %g' % self.C)

        if w0 is None:
            if fnpy is None:
                w0 = 0.001 * np.random.randn((U + N + 1) * D + N)
            else:
                try:
                    w0 = np.load(fnpy, allow_pickle=False)
                    print('Restore from %s' % fnpy)
                except (IOError, ValueError):
                    w0 = 0.001 * np.random.randn((U + N + 1) * D + N)
        assert w0.shape == ((U + N + 1) * D + N,)

        # solve using IPOPT and cutting-plane method
        self.problem.update_constraints(w0)
        n_cp_iter = 1
        w = w0
        n_vars = self.problem.n_vars
        n_cons = self.problem.n_constraints
        while self.problem.all_constraints_satisfied is False:
            # create an optimisation problem (same objective, updated constraints) and solve it
            print('\nIteration %d of cutting plane\n' % n_cp_iter)
            nlp = cyipopt.problem(
                problem_obj=self.problem,
                n=n_vars,            # number of variables
                m=n_cons,            # number of constraints
                lb=None,             # lower bounds on variables
                ub=None,             # upper bounds on variables
                cl=None,             # lower bounds on constraints
                cu=np.zeros(n_cons)  # upper bounds on constraints
            )

            # Set solver options, https://www.coin-or.org/Ipopt/documentation/node51.html
            nlp.addOption(b'mu_strategy', b'adaptive')
            nlp.addOption(b'max_iter', int(1e4))
            nlp.addOption(b'tol', 1e-7)
            nlp.addOption(b'acceptable_tol', 1e-5)
            nlp.addOption(b'linear_solver', b'ma57')
            # nlp.addOption(b'derivative_test', b'first-order')
            # nlp.addOption(b'acceptable_constr_viol_tol', 1e-6)
            w, info = nlp.solve(w)
            print(info['status'], info['status_msg'])
            self.problem.update_constraints(w)
            n_cp_iter += 1

            if fnpy is not None:
                assert type(fnpy) == str
                assert fnpy.endswith('.npy')
                try:
                    np.save(fnpy, w, allow_pickle=False)
                    if verbose > 0:
                        print('Save to %s' % fnpy)
                except (OSError, IOError, ValueError) as err:
                    sys.stderr.write('Save intermediate weights failed: {0}\n'.format(err))

        assert self.problem.all_constraints_satisfied is True
        self.mu = w[:D]
        self.V = w[D:(U + 1) * D].reshape(U, D)
        self.W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        self.xi = w[(U + N + 1) * D:]
        self.trained = True

        if verbose > 0:
            print('Training finished in %.1f seconds' % (time.time() - t0))

    def predict(self, X_test):
        """Make predictions (score is a real number)"""
        assert self.trained is True, 'Cannot make prediction before training'
        U, D = self.primal.U, self.primal.D
        assert D == X_test.shape[1]
        preds = []
        for u in range(U):
            clq = self.cliques[u]
            Wt = self.W[clq, :] + (self.V[u, :] + self.mu).reshape(1, D)
            preds.append(np.dot(X_test, Wt.T))
        return np.hstack(preds)


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


class RankPrimal(object):
    """Primal Problem of Multitask Ranking"""

    def __init__(self, X, Y, C, cliques, verbose=0):
        assert isspmatrix_csc(Y)
        assert C > 0
        self.X, self.Y = X, Y
        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        assert self.M == self.Y.shape[0]
        self.C = C

        self.cliques = cliques
        self.U = len(self.cliques)
        self.u2pl = self.cliques
        self.pl2u = np.zeros(self.N, dtype=np.int32)
        for u in range(self.U):
            clq = self.cliques[u]
            self.pl2u[clq] = u

        # self.all_constraints:
        # - a list of N lists
        # - let n = self.all_constraints[k][j], then x_n violates constraint f_{k,n} <= 0
        # - it will be updated by self.update_constraints() given model parameters
        # - self.constraints(), self.jacobianstructure() and self.jacobian() depend on this data structure
        self.all_constraints = [list() for _ in range(self.N)]
        self.all_constraints_satisfied = False
        self.n_constraints = 0

        self.n_vars = (self.U + self.N + 1) * self.D + self.N
        self.data_helper = DataHelper(self.Y, self.cliques)
        self.verbose = verbose

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

        J = np.dot(mu, mu)
        J += np.sum([np.dot(V[u, :], V[u, :]) for u in range(U)]) / U
        J += np.sum([np.dot(W[k, :], W[k, :]) for k in range(N)]) / N
        J *= C / 2
        for u in range(U):
            clq = self.cliques[u]
            Nu = len(clq)
            Yu = Ys[u]

            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = xi[clq].reshape(1, Nu) - np.dot(self.X, Wt.T)
            T1p = np.zeros(T1.shape)
            T1p[Yu.row, Yu.col] = T1[Yu.row, Yu.col]

            T2p = np.exp(T1p)
            T2 = np.zeros(T1.shape)
            T2[Yu.row, Yu.col] = T2p[Yu.row, Yu.col]
            T2 *= Ps[u]

            J += T2.sum()

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

        dmu = C * mu
        dV = V * C / U
        dW = W * C / N
        dxi = np.zeros_like(xi)
        for u in range(U):
            clq = self.cliques[u]
            Nu = len(clq)
            Yu = Ys[u]

            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = xi[clq].reshape(1, Nu) - np.dot(self.X, Wt.T)
            T1p = np.zeros(T1.shape)
            T1p[Yu.row, Yu.col] = T1[Yu.row, Yu.col]

            T2p = np.exp(T1p)
            T2 = np.zeros(T1.shape)
            T2[Yu.row, Yu.col] = T2p[Yu.row, Yu.col]
            T2 *= Ps[u]

            T3 = np.dot(T2.T, self.X)

            dv = T3.sum(axis=0)
            dmu -= dv
            dV[u, :] -= dv
            dW[clq, :] -= T3
            dxi[clq] = T2.sum(axis=0)

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
            if len(self.all_constraints[k]) > 0:
                indices = self.all_constraints[k]
                values = self.X[indices, :].dot(v + wk + mu)
                assert values.shape == (len(indices),)
                cons_values.append(values)
        return np.concatenate(cons_values, axis=-1)

    def jacobianstructure(self):
        """
            The sparse structure (i.e., rows, cols) of the Jacobian matrix
        """
        N, D, U = self.N, self.D, self.U
        rows = []
        cols = []
        ix = 0
        for k in range(N):
            u = self.pl2u[k]
            if len(self.all_constraints[k]) > 0:
                for n in self.all_constraints[k]:
                    # indices of derivatives of constraint f_{k, n} <= 0
                    rows += np.full(3 * D + 1, ix, dtype=np.int32).tolist()
                    cols += list(range(D))                                 # dmu: D
                    cols += list(range((u + 1) * D, (u + 2) * D))          # dV: U by D
                    cols += list(range((U + 1 + k) * D, (U + 2 + k) * D))  # dW: N by D
                    cols.append((U + N + 1) * D + k)                       # dxi: N
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
            if len(self.all_constraints[k]) > 0:
                for n in self.all_constraints[k]:
                    # derivatives of constraints f_{k, n} <= 0
                    jac.append(X[n, :])  # dmu
                    jac.append(X[n, :])  # dV
                    jac.append(X[n, :])  # dW
                    jac.append([-1.])    # dxi
        return np.concatenate(jac, axis=-1)

    # def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm,
    #                  regularization_size, alpha_du, alpha_pr, ls_trials):
    #     """
    #         The intermediate callback
    #     """
    #     print('Iter %5d: %g' % (iter_count, obj_value))

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
                    self.all_constraints[k].append(row)
        self.all_constraints_satisfied = all_satisfied
