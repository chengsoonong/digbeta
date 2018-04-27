import sys
import time
import numpy as np
from scipy.sparse import isspmatrix_csc, coo_matrix
import cyipopt


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
        self.Minus = M - Y.sum(axis=0).A.reshape(-1)
        for u in range(U):
            clq = cliques[u]
            Yu = Y[:, clq]
            self.Ys.append(Yu.tocoo())
        self.init = True

    def get_data(self):
        assert self.init is True
        return self.Ys, self.Minus


class QPMTR(object):
    def __init__(self, X_train, Y_train, C1, C2, C3, cliques):
        if not isspmatrix_csc(Y_train):
            raise ValueError('ERROR: %s\n' % 'Y_train should be a parse csc_matrix.')
        if not np.all(np.array([C1, C2, C3]) > 0):
            raise ValueError('ERROR: %s\n' % 'Regularisation parameters should be positive.')
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError('X_train.shape[0] != Y_train.shape[0]')

        self.X = X_train
        self.Y = Y_train
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        self.cliques = cliques
        self.U = len(self.cliques)
        self.num_vars = (self.U + self.N + 1) * self.D + self.N * 2
        self.num_cons = int(self.Y.sum()) + self.N
        self.data_helper = DataHelper(self.Y, self.cliques)
        self.trained = False

    def _init_vars(self):
        np.random.seed(0)
        return 0.001 * np.random.randn(self.num_vars)
        N, U, D = self.N, self.U, self.D
        # w0 = np.zeros((U + N + 1) * D + N)
        # w0 = 1e-5 * np.random.rand((U + N + 1) * D + N)
        # w0 = np.r_[1e-3 * np.random.randn((U + N + 1) * D), np.random.rand(N)]
        # w0 = np.r_[1e-3 * np.random.randn((U + N + 1) * D), np.zeros(N)]
        # w0 = np.r_[1e-3 * np.random.rand((U + 1) * D), np.zeros(N * (D + 1))]
        mu = 1e-3 * np.random.randn(D)
        V = 1e-3 * np.random.randn(U, D)
        W = 1e-3 * np.random.randn(N, D)
        xi = np.zeros(N)
        delta = np.zeros(N)
        Ys, Minus = self.data_helper.get_data()
        assert U == len(Ys)
        for u in range(U):
            clq = self.cliques[u]
            Yu = Ys[u]
            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = np.dot(self.X, Wt.T)
            T2 = np.full(T1.shape, np.inf, dtype=np.float)
            T2[Yu.row, Yu.col] = T1[Yu.row, Yu.col]  # mask entry (n,i) if y_n^i = 0
            xi[clq] = T2.min(axis=0)
            T1p = np.zeros(T1.shape)
            T1p[Yu.row, Yu.col] = T1[Yu.row, Yu.col]
            T1n = T1 - T1p
            delta[clq] = T1n.sum(axis=0) / Minus[clq]
        return np.r_[mu, V.ravel(), W.ravel(), xi, delta]

    def fit(self, w0=None, verbose=0):
        N, U, D = self.N, self.U, self.D
        if verbose > 0:
            t0 = time.time()
            print('\nC: %g, %g, %g' % (self.C1, self.C2, self.C3))

        problem = RankPrimal(X=self.X, Y=self.Y, C1=self.C1, C2=self.C2, C3=self.C3, cliques=self.cliques,
                             data_helper=self.data_helper, verbose=verbose)
        problem.compute_constraint_info()
        LB = np.full(self.num_vars, -cyipopt.INF, dtype=np.float)
        LB[-N:] = 0.
        # print(type(self.num_cons))

        nlp = cyipopt.problem(
            problem_obj=problem,        # problem instance
            n=self.num_vars,            # number of variables
            m=self.num_cons,            # number of constraints
            lb=LB,                      # lower bounds on variables
            ub=None,                    # upper bounds on variables
            cl=None,                    # lower bounds on constraints
            cu=np.zeros(self.num_cons)  # upper bounds on constraints
        )

        # ROOT_URL = www.coin-or.org/Ipopt/documentation
        # set solver options: $ROOT_URL/node40.html
        nlp.addOption(b'max_iter', int(1e4))
        # nlp.addOption(b'mu_strategy', b'adaptive')
        # nlp.addOption(b'tol', 1e-7)
        # nlp.addOption(b'acceptable_tol', 1e-5)
        # nlp.addOption(b'acceptable_constr_viol_tol', 1e-6)

        # linear solver for the augmented linear system: $ROOT_URL/node51.html, www.hsl.rl.ac.uk/ipopt/
        # nlp.addOption(b'linear_solver', b'ma27')  # default
        nlp.addOption(b'linear_solver', b'ma57')  # small/medium problem
        # nlp.addOption(b'linear_solver', b'ma86')  # large problem

        # gradient checking for objective and constraints: $ROOT_URL/node30.html
        # nlp.addOption(b'derivative_test', b'first-order')
        # nlp.addOption(b'derivative_test_tol', 0.0009)  # default is 0.0001
        # nlp.addOption(b'print_level', 6)

        w0 = self._init_vars()
        w, info = nlp.solve(w0)
        # print(info['status'], info['status_msg'])
        print('\n[IPOPT] %s\n' % info['status_msg'].decode('utf-8'))

        self.mu = w[:D]
        self.V = w[D:(U + 1) * D].reshape(U, D)
        self.W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        # self.xi = w[(U + N + 1) * D:(U + N + 1) * D + N]
        # self.delta = w[self.num_vars - N:self.num_vars]
        self.trained = True

        if verbose > 0:
            print('Training finished in %.1f seconds' % (time.time() - t0))

    def predict(self, X_test):
        """
            Make predictions (score is a real number)
        """
        assert self.trained is True, 'Cannot make prediction before training'
        U, D = self.U, self.D
        assert D == X_test.shape[1]
        preds = []
        for u in range(U):
            clq = self.cliques[u]
            Wt = self.W[clq, :] + (self.V[u, :] + self.mu).reshape(1, D)
            preds.append(np.dot(X_test, Wt.T))
        return np.hstack(preds)


class RankPrimal(object):
    """Primal Problem of Multitask Ranking"""

    def __init__(self, X, Y, C1, C2, C3, cliques, data_helper, verbose=0):
        assert isspmatrix_csc(Y)
        assert C1 > 0
        assert C2 > 0
        assert C3 > 0
        assert X.shape[0] == Y.shape[0]
        self.X = X
        self.Y = Y
        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.cliques = cliques
        self.U = len(self.cliques)
        self.u2pl = self.cliques
        self.pl2u = np.zeros(self.N, dtype=np.int32)
        for u in range(self.U):
            clq = self.cliques[u]
            self.pl2u[clq] = u

        self.num_vars = (self.U + self.N + 1) * self.D + self.N * 2
        self.data_helper = data_helper
        self.verbose = verbose

        self.jac = None
        self.js_rows = None
        self.js_cols = None

    def objective(self, w):
        """
            The callback for calculating the objective
        """
        t0 = time.time()
        N, D, U, C1, C2, C3 = self.N, self.D, self.U, self.C1, self.C2, self.C3
        assert w.shape == ((U + N + 1) * D + N * 2,)
        mu = w[:D]
        V = w[D:(U + 1) * D].reshape(U, D)
        W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        # xi = w[(U + N + 1) * D:(U + N + 1) * D + N]
        delta = w[(U + N + 1) * D + N:]
        assert delta.shape == (N,)

        Ys, Minus = self.data_helper.get_data()
        assert U == len(Ys)
        assert Minus.shape == (N,)

        J = np.dot(mu, mu) * 0.5 * C3
        J += np.sum([np.dot(V[u, :], V[u, :]) for u in range(U)]) * 0.5 * C1 / U
        J += np.sum([np.dot(W[k, :], W[k, :]) for k in range(N)]) * 0.5 * C2 / N
        J += delta.sum() / N

        if np.isnan(J) or np.isinf(J):
            sys.stderr.write('objective(): objective is NaN or inf!\n')
            sys.exit(0)

        if self.verbose > 2:
            print('Eval f: %.1f seconds used.' % (time.time() - t0))

        return J

    def gradient(self, w):
        """
            The callback for calculating the gradient
        """
        t0 = time.time()
        N, D, U, C1, C2, C3 = self.N, self.D, self.U, self.C1, self.C2, self.C3
        assert w.shape == ((U + N + 1) * D + N * 2,)
        mu = w[:D]
        V = w[D:(U + 1) * D].reshape(U, D)
        W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        # xi = w[(U + N + 1) * D:(U + N + 1) * D + N]
        delta = w[(U + N + 1) * D + N:]
        assert delta.shape == (N,)

        Ys, Minus = self.data_helper.get_data()
        assert U == len(Ys)
        assert Minus.shape == (N,)

        dmu = mu * C3
        dV = V * C1 / U
        dW = W * C2 / N
        dxi = np.zeros(N)
        ddelta = np.ones(N) / N

        if self.verbose > 2:
            print('Eval g: %.1f seconds used.' % (time.time() - t0))

        return np.r_[dmu, dV.ravel(), dW.ravel(), dxi, ddelta]

    def constraints(self, w):
        """
            The callback for calculating the (function value of) constraints
        """
        N, D, U = self.N, self.D, self.U
        assert w.shape == ((U + N + 1) * D + N * 2,)
        mu = w[:D]
        V = w[D:(U + 1) * D].reshape(U, D)
        W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        xi = w[(U + N + 1) * D:(U + N + 1) * D + N]
        delta = w[(U + N + 1) * D + N:]
        assert delta.shape == (N,)

        Ys, Minus = self.data_helper.get_data()
        assert U == len(Ys)
        assert Minus.shape == (N,)

        cons_values = []
        delta_cons_values = np.zeros(N)
        for u in range(U):
            # constraint: 1 - \xi_k - \delta_k + f(u(k), k, n) / M_-^k <= 0
            clq = self.cliques[u]
            Yu = Ys[u]
            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = np.dot(self.X, Wt.T)
            T1p = np.zeros(T1.shape)
            T1p[Yu.row, Yu.col] = T1[Yu.row, Yu.col]
            T1n = T1 - T1p
            delta_cons_values[clq] = 1. - xi[clq] - delta[clq] + T1n.sum(axis=0) / Minus[clq]
        cons_values += delta_cons_values.tolist()

        for k in range(N):
            # constraint: \xi_k - f(u(k), k, m) <= 0
            u = self.pl2u[k]
            v = V[u, :]
            wk = W[k, :]
            for m in range(self.M):
                if self.Y[m, k] == 1:
                    cons_values.append(xi[k] - np.dot(self.X[m, :], v + wk + mu))

        # for k in range(N):
        #     u = self.pl2u[k]
        #     v = V[u, :]
        #     wk = W[k, :]
        #     if len(self.current_constraints[k]) > 0:
        #         indices = sorted(self.current_constraints[k])
        #         values = self.X[indices, :].dot(v + wk + mu) - xi[k]
        #         assert values.shape == (len(indices),)
        #         cons_values.append(values)
        # return np.concatenate(cons_values, axis=-1)
        return np.asarray(cons_values)

    def jacobianstructure(self):
        """
            The sparse structure (i.e., rows, cols) of the Jacobian matrix
        """
        assert self.js_rows is not None
        assert self.js_cols is not None
        return self.js_rows, self.js_cols

    def jacobian(self, w):
        """
            The callback for calculating the Jacobian of constraints
        """
        assert self.jac is not None
        return self.jac

    # def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm,
    #                  regularization_size, alpha_du, alpha_pr, ls_trials):
    #     """
    #         The intermediate callback
    #     """
    #     print('Iter %5d: %g' % (iter_count, obj_value))

    def jacobian_sanity_check(self):
        """
            Check if compute_constraint_info() compute the correct jacobian and jacobianstructure
            NOTE the follow results of coo_matrix:
                mat = coo_matrix(data, (rows, cols))
                assert np.all(mat.row == rows)  # NOT TRUE
                assert np.all(mat.col == cols)  # NOT TRUE
        """
        if (self.jac is None) or (self.js_rows is None) or (self.js_cols is None):
            self.compute_constraint_info()

        U, N, D = self.U, self.N, self.D
        jac = []
        for k in range(N):
            u = self.pl2u[k]
            ny = 1 - self.Y[:, k].A.reshape(-1)
            vec = ny.dot(self.X) / ny.sum()
            dmu = vec
            dV = np.zeros((U, D), dtype=np.float)
            dV[u, :] = vec
            dW = np.zeros((N, D), dtype=np.float)
            dW[k, :] = vec
            dxi = np.zeros(N, dtype=np.float)
            dxi[k] = -1.
            ddelta = np.zeros(N, dtype=np.float)
            ddelta[k] = -1.
            jac.append(np.r_[dmu, dV.ravel(), dW.ravel(), dxi, ddelta])
        for k in range(N):
            u = self.pl2u[k]
            for m in range(self.M):
                if self.Y[m, k] == 1:
                    vec = -self.X[m, :]
                    dmu = vec
                    dV = np.zeros((U, D), dtype=np.float)
                    dV[u, :] = vec
                    dW = np.zeros((N, D), dtype=np.float)
                    dW[k, :] = vec
                    dxi = np.zeros(N, dtype=np.float)
                    dxi[k] = 1.
                    ddelta = np.zeros(N, dtype=np.float)
                    jac.append(np.r_[dmu, dV.ravel(), dW.ravel(), dxi, ddelta])
        jac_coo = coo_matrix(np.vstack(jac))

        # self.jac = jac_coo.data
        # self.js_rows = jac_coo.row
        # self.js_cols = jac_coo.col

        print('checking jacobianstructure ...')
        assert np.all(jac_coo.row == self.js_rows)
        assert np.all(jac_coo.col == self.js_cols)

        print('checking jacobian ...')
        assert np.all(jac_coo.data == self.jac)

    def compute_constraint_info(self):
        """
            compute jacobian of constraints and its sparse structure (rows, cols of non-zero elements)
        """
        N, D, U = self.N, self.D, self.U
        rows, cols, jac = [], [], []
        ix = 0
        for k in range(N):
            # constraint: 1 - \xi_k - \delta_k + \sum_{n: y_n^k=0} f(u(k), k, n) / M_-^k <= 0
            u = self.pl2u[k]
            ny = 1 - self.Y[:, k].A.reshape(-1)
            vec = ny.dot(self.X) / ny.sum()
            jac += [vec, vec, vec, [-1., -1.]]
            rows.append(np.full(3 * D + 2, ix, dtype=np.int32))
            cols.append(np.arange(D, dtype=np.int32))
            cols.append(np.arange((u + 1) * D, (u + 2) * D, dtype=np.int32))
            cols.append(np.arange((U + 1 + k) * D, (U + 2 + k) * D, dtype=np.int32))
            cols.append([(U + N + 1) * D + k])
            cols.append([(U + N + 1) * D + N + k])
            ix += 1
        for k in range(N):
            u = self.pl2u[k]
            for m in range(self.M):
                if self.Y[m, k] == 1:
                    # constraint: \xi_k - f(u(k), k, m) <= 0
                    vec = -self.X[m, :]
                    jac += [vec, vec, vec, [1.]]
                    rows.append(np.full(3 * D + 1, ix, dtype=np.int32))
                    cols.append(np.arange(D))
                    cols.append(np.arange((u + 1) * D, (u + 2) * D))
                    cols.append(np.arange((U + 1 + k) * D, (U + 2 + k) * D))
                    cols.append([(U + N + 1) * D + k])
                    ix += 1
        self.jac = np.concatenate(jac, axis=-1)
        self.js_rows = np.concatenate(rows, axis=-1)
        self.js_cols = np.concatenate(cols, axis=-1)
