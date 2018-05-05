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
        self.Pindices = [list() for k in range(N)]
        Minus = M - Y.sum(axis=0).A.reshape(-1)
        self.Q = 1. / (N * Minus)
        for u in range(U):
            clq = cliques[u]
            Yu = Y[:, clq]
            self.Ys.append(Yu.tocoo())
            for k in clq:
                y = Y[:, k].A.reshape(-1)
                self.Pindices[k][:] = np.where(y > 0)[0]
        self.init = True

    def get_data(self):
        assert self.init is True
        return self.Ys, self.Q, self.Pindices


class MTR(object):
    def __init__(self, X_train, Y_train, C1, C2, C3, cliques):
        assert C1 > 0
        assert C2 > 0
        assert C3 > 0
        assert X_train.shape[0] == Y_train.shape[0]
        assert isspmatrix_csc(Y_train)
        self.X = X_train
        self.Y = Y_train
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.cliques = cliques
        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        self.U = len(self.cliques)
        self.data_helper = DataHelper(self.Y, self.cliques)
        self.trained = False

    def _init_vars(self):
        np.random.seed(0)
        N, U, D = self.N, self.U, self.D
        w0 = 1e-4 * np.random.randn((U + N + 1) * D + N)
        return w0

    def fit(self, loss='exponential', w0=None, verbose=0):
        t0 = time.time()
        if loss not in ['exponential', 'squared_hinge']:
            raise ValueError("'loss' should be either 'exponential' or 'squared_hinge'")
        N, U, D = self.N, self.U, self.D
        if verbose > 0:
            print('\nC: {:g}, {:g}, {:g}'.format(self.C1, self.C2, self.C3))

        num_vars = (U + N + 1) * D + N
        num_cons = int(self.Y.sum())
        if w0 is None:
            w0 = self._init_vars()
        if w0.shape != (num_vars,):
            raise ValueError('ERROR: incorrect dimention for initial weights.')

        # solve using IPOPT
        problem = RankPrimal(X=self.X, Y=self.Y, C1=self.C1, C2=self.C2, C3=self.C3, cliques=self.cliques,
                             data_helper=self.data_helper, loss=loss, verbose=verbose)

        if verbose > 0:
            print('[IPOPT] {:,d} variables, {:,d} constraints.'.format(num_vars, num_cons))

        problem.compute_constraint_info()

        nlp = cyipopt.problem(
            problem_obj=problem,    # problem instance
            n=num_vars,             # number of variables
            m=num_cons,             # number of constraints
            lb=None,                # lower bounds on variables
            ub=None,                # upper bounds on variables
            cl=np.zeros(num_cons),  # lower bounds on constraints
            cu=None                 # upper bounds on constraints
        )

        # ROOT_URL = www.coin-or.org/Ipopt/documentation
        # set solver options: $ROOT_URL/node40.html
        # nlp.addOption(b'max_iter', int(1e3))
        # nlp.addOption(b'mu_strategy', b'adaptive')
        # nlp.addOption(b'tol', 1e-7)
        # nlp.addOption(b'acceptable_tol', 1e-5)
        # nlp.addOption(b'acceptable_constr_viol_tol', 1e-6)

        # linear solver for the augmented linear system: $ROOT_URL/node51.html, www.hsl.rl.ac.uk/ipopt/
        # nlp.addOption(b'linear_solver', b'ma27')  # default
        # nlp.addOption(b'linear_solver', b'ma57')  # small/medium problem
        nlp.addOption(b'linear_solver', b'ma86')  # large problem

        # gradient checking for objective and constraints: $ROOT_URL/node30.html
        # nlp.addOption(b'derivative_test', b'first-order')
        # nlp.addOption(b'derivative_test_print_all', b'yes')
        nlp.addOption(b'print_level', 1)

        w, info = nlp.solve(w0)
        # print(info['status'], info['status_msg'])
        print('\n[IPOPT] %s\n' % info['status_msg'].decode('utf-8'))

        self.V = w[:U * D].reshape(U, D)
        self.W = w[U * D:(U + N) * D].reshape(N, D)
        self.mu = w[(U + N) * D:(U + N + 1) * D]
        self.xi = w[(U + N + 1) * D:]
        self.pl2u = problem.pl2u
        self.trained = True
        self.status = info['status']
        self.status_msg = info['status_msg']

        if verbose > 0:
            print('Training finished in %.1f seconds' % (time.time() - t0))


class RankPrimal(object):
    """Primal Problem of Multitask Ranking"""

    def __init__(self, X, Y, C1, C2, C3, cliques, data_helper, loss, verbose=0):
        assert isspmatrix_csc(Y)
        assert C1 > 0
        assert C2 > 0
        assert C3 > 0
        assert X.shape[0] == Y.shape[0]
        assert loss in ['exponential', 'squared_hinge'], "'loss' should be either 'exponential' or 'squared_hinge'"
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
        self.data_helper = data_helper
        self.loss = loss
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
        assert w.shape == ((U + N + 1) * D + N,)
        V = w[:U * D].reshape(U, D)
        W = w[U * D:(U + N) * D].reshape(N, D)
        mu = w[(U + N) * D:(U + N + 1) * D]
        xi = w[(U + N + 1) * D:]
        assert xi.shape == (N,)
        Ys, Q, Pindices = self.data_helper.get_data()
        assert U == len(Ys)
        assert Q.shape == (N,)

        J = 0.
        J += np.sum([np.dot(V[u, :], V[u, :]) for u in range(U)]) * C1 / 2
        J += np.abs(W).sum() * C2
        J += np.abs(mu).sum() * C3

        def risk_exponential(u):
            clq = self.cliques[u]
            Nu = len(clq)
            Yu = Ys[u]
            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = np.dot(self.X, Wt.T)  # M by Nu
            T1[Yu.row, Yu.col] = -np.inf  # mask entries (m, k) that y_m^k = 1
            T2 = np.exp(T1)
            T2 *= (np.exp(-xi[clq]) * Q[clq]).reshape(1, Nu)
            return T2.sum()

        def risk_squared_hinge(u):
            clq = self.cliques[u]
            Nu = len(clq)
            Yu = Ys[u]
            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = np.dot(self.X, Wt.T) + (1. - xi[clq]).reshape(1, Nu)  # M by Nu
            T1[Yu.row, Yu.col] = 0.   # mask entries (m, k) that y_m^k = 1
            T1[T1 < 0] = 0.
            T2 = np.square(T1)
            T2 *= Q[clq].reshape(1, Nu)
            return T2.sum()

        for u in range(U):
            if self.loss == 'exponential':
                J += risk_exponential(u)
            else:
                J += risk_squared_hinge(u)

        if np.isnan(J) or np.isinf(J):
            raise ValueError('objective(): objective is NaN or inf!\n')

        if self.verbose > 1:
            print('Eval f: %.1f seconds used.' % (time.time() - t0))

        return J

    def gradient(self, w):
        """
            The callback for calculating the gradient
        """
        t0 = time.time()
        N, D, U, C1, C2, C3 = self.N, self.D, self.U, self.C1, self.C2, self.C3
        assert w.shape == ((U + N + 1) * D + N,)
        V = w[:U * D].reshape(U, D)
        W = w[U * D:(U + N) * D].reshape(N, D)
        mu = w[(U + N) * D:(U + N + 1) * D]
        xi = w[(U + N + 1) * D:]
        assert xi.shape == (N,)
        Ys, Q, Pindices = self.data_helper.get_data()
        assert U == len(Ys)
        assert Q.shape == (N,)
        assert N == len(Pindices)

        def grad_exponential(u):
            clq = self.cliques[u]
            Nu = len(clq)
            Yu = Ys[u]
            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = np.dot(self.X, Wt.T)  # M by Nu
            T1[Yu.row, Yu.col] = -np.inf  # mask entries (m, k) that y_m^k = 1
            T2 = np.exp(T1)
            T3 = np.dot(T2.T, self.X)  # Nu by D
            xivec = np.exp(-xi[clq]) * Q[clq]
            T3 *= xivec.reshape(Nu, 1)
            vec = T3.sum(axis=0)
            dV = vec
            dW = T3
            dmu = vec
            dxi = -T2.sum(axis=0) * xivec
            return dV, dW, dmu, dxi

        def grad_squred_hinge(u):
            clq = self.cliques[u]
            Nu = len(clq)
            Yu = Ys[u]
            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = np.dot(self.X, Wt.T) + (1. - xi[clq]).reshape(1, Nu)  # M by Nu
            T1[Yu.row, Yu.col] = 0.   # mask entries (m, k) that y_m^k = 1
            T1[T1 < 0] = 0.
            T2 = 2 * np.dot(T1.T, self.X)  # Nu by D
            T2 *= Q[clq].reshape(Nu, 1)
            vec = T2.sum(axis=0)
            dV = vec
            dW = T2
            dmu = vec
            dxi = -2. * T1.sum(axis=0) * Q[clq]
            return dV, dW, dmu, dxi

        dV = V * C1
        dW = np.sign(W) * C2
        dmu = np.sign(mu) * C3
        dxi = np.zeros_like(xi)
        for u in range(U):
            clq = self.cliques[u]
            if self.loss == 'exponential':
                grads = grad_exponential(u)
            else:
                grads = grad_squred_hinge(u)
            dV[u, :] += grads[0]
            dW[clq, :] += grads[1]
            dmu += grads[2]
            dxi[clq] = grads[3]

        if self.verbose > 1:
            print('Eval g: %.1f seconds used.' % (time.time() - t0))

        return np.r_[dV.ravel(), dW.ravel(), dmu, dxi]

    def constraints(self, w):
        """
            The callback for calculating the (function value of) constraints
        """
        N, D, U = self.N, self.D, self.U
        assert w.shape == ((U + N + 1) * D + N,)
        V = w[:U * D].reshape(U, D)
        W = w[U * D:(U + N) * D].reshape(N, D)
        mu = w[(U + N) * D:(U + N + 1) * D]
        xi = w[(U + N + 1) * D:]
        assert xi.shape == (N,)
        _, _, Pindices = self.data_helper.get_data()
        assert len(Pindices) == N

        cons_values = []
        for k in range(N):
            u = self.pl2u[k]
            v = V[u, :]
            wk = W[k, :]
            indices = Pindices[k]
            assert len(indices) > 0
            values = self.X[indices, :].dot(v + wk + mu) - xi[k]
            assert values.shape == (len(indices),)
            cons_values.append(values)
        return np.concatenate(cons_values, axis=-1)

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

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm,
                     regularization_size, alpha_du, alpha_pr, ls_trials):
        """
            The intermediate callback
        """
        if self.verbose > 0:
            print('Iter {:5d}: {:.9f}, {:s}'.format(iter_count, obj_value, time.strftime('%Y-%m-%d %H:%M:%S')))

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
        _, _, Pindices = self.data_helper.get_data()
        assert len(Pindices) == N
        jac = []
        for k in range(N):
            u = self.pl2u[k]
            for m in Pindices[k]:
                dV = np.zeros((U, D), dtype=np.float)
                dV[u, :] = self.X[m, :]
                dW = np.zeros((N, D), dtype=np.float)
                dW[k, :] = self.X[m, :]
                dmu = self.X[m, :]
                dxi = np.zeros(N, dtype=np.float)
                dxi[k] = -1
                jac.append(np.r_[dmu, dV.ravel(), dW.ravel(), dxi])
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
        _, _, Pindices = self.data_helper.get_data()
        assert len(Pindices) == N
        rows, cols, jac = [], [], []
        ix = 0
        for k in range(N):
            u = self.pl2u[k]
            for m in Pindices[k]:
                jac.append(np.tile(self.X[m, :].reshape(1, D), (1, 3)))
                jac.append(np.array([-1.]).reshape(1, 1))
                rows.append(np.full(3 * D + 1, ix, dtype=np.int32))
                cols.append(np.arange(D, dtype=np.int32))
                cols.append(np.arange((u + 1) * D, (u + 2) * D, dtype=np.int32))
                cols.append(np.arange((U + 1 + k) * D, (U + 2 + k) * D, dtype=np.int32))
                cols.append([(U + N + 1) * D + k])
                ix += 1
        self.jac = np.concatenate(jac, axis=-1)
        self.js_rows = np.concatenate(rows, axis=-1)
        self.js_cols = np.concatenate(cols, axis=-1)
