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


class MTR(object):
    def __init__(self, X_train, Y_train, C, cliques, verbose=0):
        assert C > 0
        assert X_train.shape[0] == Y_train.shape[0]
        assert isspmatrix_csc(Y_train)

        self.X = X_train
        self.Y = Y_train
        self.C = C
        self.cliques = cliques
        self.verbose = verbose

        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        self.U = len(self.cliques)
        self.data_helper = DataHelper(self.Y, self.cliques)
        self.trained = False

    def _init_vars(self):
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
        Ys, _ = self.data_helper.get_data()
        assert U == len(Ys)
        for u in range(U):
            clq = self.cliques[u]
            Yu = Ys[u]
            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = np.dot(self.X, Wt.T)
            T1[Yu.row, Yu.col] = -np.inf  # mask entry (m,i) if y_m^i = 1
            xi[clq] = T1.max(axis=0)
        return np.r_[mu, V.ravel(), W.ravel(), xi]

    def fit(self, loss='exponential', regu='l2', max_iter=1e3, use_all_constraints=False, w0=None, fnpy=None):
        assert loss in ['exponential', 'squared_hinge']
        assert regu in ['l1', 'l2']
        M, N, U, D = self.M, self.N, self.U, self.D
        verbose = self.verbose
        if verbose > 0:
            t0 = time.time()
            print('\nC: %g' % self.C)

        num_vars = (U + N + 1) * D + N
        max_num_cons = M * N - self.Y.sum()
        if w0 is None:
            if fnpy is None:
                w0 = self._init_vars()
            else:
                try:
                    assert type(fnpy) == str
                    assert fnpy.endswith('.npy')
                    w0 = np.load(fnpy, allow_pickle=False)
                    print('Restore from %s' % fnpy)
                except (IOError, ValueError):
                    w0 = self._init_vars()
        assert w0.shape == (num_vars,)

        # solve using IPOPT and cutting-plane method
        # first create an optimisation problem and solve it,
        # then add constraints violated by current solution,
        # create a new optimisation problem (same objective, update constraints)
        # keep doing this until termination criteria satisfied.
        w = w0
        cp_iter = 0
        current_constraints = None
        print('[CUTTING-PLANE] %d variables, %d maximum possible constraints.' % (num_vars, max_num_cons))
        while cp_iter < max_iter:
            cp_iter += 1
            problem = RankPrimal(X=self.X, Y=self.Y, C=self.C, cliques=self.cliques, data_helper=self.data_helper,
                                 loss=loss, regu=regu, verbose=verbose)

            if use_all_constraints is True:
                problem.generate_all_constraints()
                print('[CUTTING-PLANE] All constraints will be used.')

            if current_constraints is not None:
                problem.add_constraints(current_constraints)  # restore constraints

            num_cons = problem.get_num_constraints()
            if num_cons > 0:
                problem.compute_constraint_info()
            else:
                problem.jacobian = None
                problem.jacobianstructure = None

            print('[CUTTING-PLANE] Iter %d: %d constraints.' % (cp_iter, num_cons))
            nlp = cyipopt.problem(
                problem_obj=problem,   # problem instance
                n=num_vars,            # number of variables
                m=num_cons,            # number of constraints
                lb=None,               # lower bounds on variables
                ub=None,               # upper bounds on variables
                cl=None,               # lower bounds on constraints
                cu=np.zeros(num_cons)  # upper bounds on constraints
            )

            # ROOT_URL = www.coin-or.org/Ipopt/documentation
            # set solver options: $ROOT_URL/node40.html
            nlp.addOption(b'max_iter', int(1e5))
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
            # nlp.addOption(b'print_level', 6)

            # w = self._init_vars()  # cold start, comment this line for warm start
            w, info = nlp.solve(w)
            # print(info['status'], info['status_msg'])
            print('\n[IPOPT] %s\n' % info['status_msg'].decode('utf-8'))

            if use_all_constraints is True:
                break

            problem.update_constraints(w)
            if problem.all_constraints_satisfied is True:
                print('[CUTTING-PLANE] All constraints satisfied.')
                break
            elif num_cons == problem.get_num_constraints():
                print('[CUTTING-PLANE] No more effective constraints, violations are considered acceptable by IPOPT.')
                break
            else:
                current_constraints = problem.get_current_constraints()

            if fnpy is not None:
                try:
                    np.save(fnpy, w, allow_pickle=False)
                    if verbose > 0:
                        print('Save to %s' % fnpy)
                except (OSError, IOError, ValueError) as err:
                    sys.stderr.write('Save intermediate weights failed: {0}\n'.format(err))

        if cp_iter >= max_iter:
            print('[CUTTING-PLANE] Reaching max number of iterations.')

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

    def __init__(self, X, Y, C, cliques, data_helper, loss, regu, verbose=0):
        assert isspmatrix_csc(Y)
        assert C > 0
        assert X.shape[0] == Y.shape[0]
        assert loss in ['exponential', 'squared_hinge'], "'loss' should be either 'exponential' or 'squared_hinge'"
        assert regu in ['l1', 'l2'], "'regu' should be either 'l1' or 'l2'"
        self.X = X
        self.Y = Y
        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        self.C = C
        self.loss = loss
        self.regu = regu

        self.cliques = cliques
        self.U = len(self.cliques)
        self.u2pl = self.cliques
        self.pl2u = np.zeros(self.N, dtype=np.int32)
        for u in range(self.U):
            clq = self.cliques[u]
            self.pl2u[clq] = u

        # self.current_constraints:
        # - a list of N sets
        # - if `n` \in self.current_constraints[k], it means `x_n` violates constraint `f_{k,n} <= 0`
        # - it will be modified by self.update_constraints() and self.restore_constraints()
        self.current_constraints = [set() for _ in range(self.N)]
        self.all_constraints_satisfied = False

        self.num_vars = (self.U + self.N + 1) * self.D + self.N
        self.data_helper = data_helper
        self.verbose = verbose
        # self.tol = 1e-8

        self.jac = None
        self.js_rows = None
        self.js_cols = None

    def get_num_vars(self):
        return self.num_vars

    def get_num_constraints(self):
        return int(np.sum([len(cons) for cons in self.current_constraints]))

    def get_current_constraints(self):
        current_constraints = []
        for k in range(self.N):
            current_constraints.append(self.current_constraints[k].copy())
        return current_constraints

    def add_constraints(self, additional_constraints):
        if additional_constraints is not None:
            assert type(additional_constraints) == list
            for k in range(self.N):
                assert type(additional_constraints[k]) == set
                self.current_constraints[k] = self.current_constraints[k] | additional_constraints[k]

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

        J = np.dot(mu, mu) * 0.5 * C
        J += np.sum([np.dot(V[u, :], V[u, :]) for u in range(U)]) * 0.5 * C / U
        if self.regu == 'l2':
            J += np.sum([np.dot(W[k, :], W[k, :]) for k in range(N)]) * 0.5 * C / N
        else:
            J += np.abs(W).sum() * C / N

        def risk_exponential(u):
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
            return T2.sum()

        def risk_squared_hinge(u):
            clq = self.cliques[u]
            Nu = len(clq)
            Yu = Ys[u]

            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = (1 + xi[clq]).reshape(1, Nu) - np.dot(self.X, Wt.T)
            v1p = T1[Yu.row, Yu.col]
            v1p[v1p < 0] = 0

            T2 = np.zeros(T1.shape)
            T2[Yu.row, Yu.col] = v1p * v1p  # square it
            T2 *= Ps[u]
            return T2.sum()

        for u in range(U):
            if self.loss == 'exponential':
                J += risk_exponential(u)
            else:
                J += risk_squared_hinge(u)

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
        N, D, C, U = self.N, self.D, self.C, self.U
        assert w.shape == ((U + N + 1) * D + N,)
        mu = w[:D]
        V = w[D:(U + 1) * D].reshape(U, D)
        W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        xi = w[(U + N + 1) * D:]
        assert xi.shape == (N,)

        Ys, Ps = self.data_helper.get_data()
        assert U == len(Ys) == len(Ps)

        def grad_exponential(u):
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

            dmuu = dv
            dVu = dv
            dWu = T3
            dxiu = T2.sum(axis=0)
            return dmuu, dVu, dWu, dxiu

        def grad_squred_hinge(u):
            clq = self.cliques[u]
            Nu = len(clq)
            Yu = Ys[u]

            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = (1 + xi[clq]).reshape(1, Nu) - np.dot(self.X, Wt.T)
            v1p = T1[Yu.row, Yu.col]
            v1p[v1p < 0] = 0

            T2 = np.zeros(T1.shape)
            T2[Yu.row, Yu.col] = v1p
            T2 *= Ps[u]

            T3 = 2 * np.dot(T2.T, self.X)
            dv = T3.sum(axis=0)

            dmuu = dv
            dVu = dv
            dWu = T3
            dxiu = 2 * T2.sum(axis=0)
            return dmuu, dVu, dWu, dxiu

        dmu = C * mu
        dV = V * C / U
        dxi = np.zeros_like(xi)
        if self.regu == 'l2':
            dW = W * C / N
        else:
            dW = np.sign(W) * C / N

        for u in range(U):
            clq = self.cliques[u]
            if self.loss == 'exponential':
                dmuu, dVu, dWu, dxiu = grad_exponential(u)
            else:
                dmuu, dVu, dWu, dxiu = grad_squred_hinge(u)
            dmu -= dmuu
            dV[u, :] -= dVu
            dW[clq, :] -= dWu
            dxi[clq] = dxiu

        if self.verbose > 2:
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
            if len(self.current_constraints[k]) > 0:
                indices = sorted(self.current_constraints[k])
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
        X = self.X
        jac = []
        for k in range(N):
            u = self.pl2u[k]
            for n in sorted(self.current_constraints[k]):
                dmu = X[n, :]
                dV = np.zeros((U, D), dtype=np.float)
                dV[u, :] = X[n, :]
                dW = np.zeros((N, D), dtype=np.float)
                dW[k, :] = X[n, :]
                dxi = np.zeros(N, dtype=np.float)
                dxi[k] = -1
                # jac.append(np.hstack([dmu.reshape(1, -1), dV.reshape(1, -1), dW.reshape(1, -1), dxi.reshape(1, -1)]))
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
        X = self.X
        rows, cols, jac = [], [], []
        ix = 0
        for k in range(N):
            u = self.pl2u[k]
            for n in sorted(self.current_constraints[k]):
                jac.append(np.tile(X[n, :].reshape(1, -1), (1, 3)))
                jac.append(np.array([-1.]).reshape(1, 1))
                rows.append(np.full(3 * D + 1, ix, dtype=np.int32))
                cols.append(np.arange(D))
                cols.append(np.arange((u + 1) * D, (u + 2) * D))
                cols.append(np.arange((U + 1 + k) * D, (U + 2 + k) * D))
                cols.append([(U + N + 1) * D + k])
                ix += 1
        self.jac = np.concatenate(jac, axis=-1)
        self.js_rows = np.concatenate(rows, axis=-1)
        self.js_cols = np.concatenate(cols, axis=-1)

    def generate_all_constraints(self):
        """
            generate all possible constraints
        """
        sys.stdout.write('Generating all possible constraints ... ')
        M, N = self.M, self.N
        Y = self.Y
        assert Y.dtype == np.bool
        for n in range(M):
            for k in range(N):
                if Y[n, k] < 1:
                    self.current_constraints[k].add(n)
        print("%d constraints in total." % self.get_num_constraints())

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

            if self.verbose > 1:
                print(np.max(T1, axis=0))
                print(xi[clq])

            assert max_ix.shape[0] == len(clq)
            for j in range(max_ix.shape[0]):
                k = clq[j]
                row, col = max_ix[j], j
                # if xi[k] + self.tol < T1[row, col]:
                if xi[k] < T1[row, col] or (xi[k] == T1[row, col] == 0):
                    all_satisfied = False
                    self.current_constraints[k].add(row)
        self.all_constraints_satisfied = all_satisfied
