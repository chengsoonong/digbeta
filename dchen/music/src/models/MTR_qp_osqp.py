import sys
import time
import numpy as np
from scipy.sparse import isspmatrix_csc, csc_matrix, vstack
import osqp


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
        self.Mplus = Y.sum(axis=0).A.reshape(-1)
        self.Q = 1. / (N * self.Mplus)
        for u in range(U):
            clq = cliques[u]
            Yu = Y[:, clq]
            self.Ys.append(Yu.tocoo())
        self.init = True

    def get_data(self):
        assert self.init is True
        return self.Ys, self.Q, self.Mplus


class MTR(object):
    def __init__(self, X_train, Y_train, C1, C2, C3, cliques, verbose=0):
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
        self.cliques = cliques
        self.verbose = verbose

        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        self.U = len(self.cliques)
        self.pl2u = np.zeros(self.N, dtype=np.int32)
        for u in range(self.U):
            clq = self.cliques[u]
            self.pl2u[clq] = u

        self.num_vars = (self.U + self.N + 1) * self.D + self.N * 2
        self.data_helper = DataHelper(self.Y, self.cliques)
        self.P = self._create_P()
        self.q = self._create_q()
        self.A, self.lb = self._create_Alb()

        # self.current_constraints:
        # - a list of N sets
        # - if `n` \in self.current_constraints[k], it means `x_n` violates constraint `f_{k,n} <= 0`
        # - it will be modified by self.update_constraints() and self.restore_constraints()
        self.current_constraints = [set() for _ in range(self.N)]
        self.all_constraints_satisfied = False
        self.trained = False

    def get_num_vars(self):
        return self.num_vars

    def get_num_constraints(self):
        return int(np.sum([len(cons) for cons in self.current_constraints])) + self.N * 2

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

    def update_constraints(self, w):
        N, D, U = self.N, self.D, self.U
        X = self.X
        assert w.shape == ((U + N + 1) * D + N * 2,)
        mu = w[:D]
        V = w[D:(U + 1) * D].reshape(U, D)
        W = w[(U + 1) * D:(U + N + 1) * D].reshape(N, D)
        xi = w[(U + N + 1) * D:(U + N + 1) * D + N]
        Ys, _, _ = self.data_helper.get_data()
        assert U == len(Ys)

        all_satisfied = True
        for u in range(U):
            clq = self.cliques[u]
            Yu = Ys[u]
            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
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
        di = np.zeros(N)
        Ys, _, _ = self.data_helper.get_data()
        assert U == len(Ys)
        for u in range(U):
            clq = self.cliques[u]
            Yu = Ys[u]
            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = np.dot(self.X, Wt.T)
            T2 = np.zeros(T1.shape)
            T2[Yu.row, Yu.col] = T1[Yu.row, Yu.col]
            T1[Yu.row, Yu.col] = -np.inf  # mask entry (m,i) if y_m^i = 1
            xi[clq] = T1.max(axis=0)
            di[clq] = T2.sum(axis=0)
        return np.r_[mu, V.ravel(), W.ravel(), xi, di]

    def _create_P(self):
        N, U, D, C1, C2, C3 = self.N, self.U, self.D, self.C1, self.C2, self.C3
        rows, cols, values = [], [], []
        values.append(np.full(D, C3, dtype=np.float))
        values.append(np.full(U * D, C1 / U, dtype=np.float))
        values.append(np.full(N * D, C2 / N, dtype=np.float))
        ix_mu = np.arange(D)
        rows.append(ix_mu)
        cols.append(ix_mu)
        ix_V = np.arange(D, (U + 1) * D)
        rows.append(ix_V)
        cols.append(ix_V)
        ix_W = np.arange((U + 1) * D, (U + 1 + N) * D)
        rows.append(ix_W)
        cols.append(ix_W)
        # csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
        P = csc_matrix((np.concatenate(values, axis=-1),
                        (np.concatenate(rows, axis=-1), np.concatenate(cols, axis=-1))),
                       shape=(self.num_vars, self.num_vars))
        return P

    def _create_q(self):
        N = self.N
        _, Q, _ = self.data_helper.get_data()
        # rows = np.arange(self.num_vars - N, self.num_vars)
        # cols = np.zeros(N, dtype=np.int32)
        # q = cvxopt.spmatrix(Q, rows, cols, size=(self.num_vars, 1))
        q = np.zeros(self.num_vars)  # should be dense matrix
        q[-N:] = Q
        return q

    def _create_Alb(self):
        N, D, U = self.N, self.D, self.U
        _, _, Mplus = self.data_helper.get_data()
        rows, cols, A = [], [], []
        ix = 0
        for k in range(N):
            # constraint: \sum_{m: y_m^k=1} (1 - f(u(k), k, m) + \xi_k) - \delta_k <= 0
            # which is equivalent to
            # (v_{u(k)} + w_k + \mu)^T \sum_{m: y_m^k=1} x_m - M_+^k * \xi_k + \delta_k >= M_+^k
            u = self.pl2u[k]
            yk = self.Y[:, k].A.reshape(-1)
            vec = yk.dot(self.X)
            A += [vec, vec, vec, [-yk.sum(), 1.]]
            rows.append(np.full(3 * D + 2, ix, dtype=np.int32))
            cols.append(np.arange(D))
            cols.append(np.arange((u + 1) * D, (u + 2) * D))
            cols.append(np.arange((U + 1 + k) * D, (U + 2 + k) * D))
            cols.append([(U + N + 1) * D + k])
            cols.append([(U + N + 1) * D + N + k])
            ix += 1

        # constraint: \delta_k >= 0
        A.append(np.full(N, 1., dtype=np.float))
        rows.append(np.arange(ix, ix + N))
        cols.append(np.arange(self.num_vars - N, self.num_vars))

        A = csc_matrix((np.concatenate(A, axis=-1),
                        (np.concatenate(rows, axis=-1), np.concatenate(cols, axis=-1))),
                       shape=(2 * N, self.num_vars))
        lb = np.zeros((2 * N))
        lb[:N] = Mplus
        return A, lb

    def _update_Alb(self):
        N, D, U = self.N, self.D, self.U
        if self.get_num_constraints() <= 2 * N:
            return self.A, self.lb

        rows, cols, values = [], [], []
        ix = 0
        for k in range(N):
            u = self.pl2u[k]
            for n in sorted(self.current_constraints[k]):
                # constraint: \xi_k - f(u(k), k, n) >= 0
                vec = -self.X[n, :]
                values += [vec, vec, vec, [1.]]
                rows.append(np.full(3 * D + 1, ix, dtype=np.int32))
                cols.append(np.arange(D))
                cols.append(np.arange((u + 1) * D, (u + 2) * D))
                cols.append(np.arange((U + 1 + k) * D, (U + 2 + k) * D))
                cols.append([(U + N + 1) * D + k])
                ix += 1
        A = vstack([self.A,
                    csc_matrix((np.concatenate(values, axis=-1),
                                (np.concatenate(rows, axis=-1), np.concatenate(cols, axis=-1))),
                               shape=(ix, self.num_vars))]).tocsc()
        lb = np.zeros(A.shape[0])
        lb[:N] = self.lb[:N]
        return A, lb

    def fit(self, max_iter=1e3, use_all_constraints=False, w0=None, fnpy=None):
        M, N, U, D = self.M, self.N, self.U, self.D
        verbose = self.verbose
        if verbose > 0:
            t0 = time.time()
            print('\nC: %g, %g, %g' % (self.C1, self.C2, self.C3))

        num_vars = (U + N + 1) * D + N * 2
        max_num_cons = M * N - self.Y.sum() + N
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

        # solve using CVXOPT and cutting-plane method
        # first create an optimisation problem and solve it,
        # then add constraints violated by current solution,
        # create a new optimisation problem (same objective, update constraints)
        # keep doing this until termination criteria satisfied.
        w = w0
        cp_iter = 0
        current_constraints = None
        num_cons = 0
        print('[CUTTING-PLANE] %d variables, %d maximum possible constraints.' % (num_vars, max_num_cons))
        while cp_iter < max_iter:
            cp_iter += 1
            A, lb = self._update_Alb()
            num_cons = self.get_num_constraints()
            print('[CUTTING-PLANE] Iter %d: %d constraints.' % (cp_iter, num_cons))
            qp = osqp.OSQP()
            qp.setup(self.P, self.q, A, lb, verbose=False)
            # qp.update_verbose(False)
            # qp.update_settings(verbose=False)
            results = qp.solve()
            w = results.x
            print('\n[OSQP] %s\n' % results.info.status)

            # if use_all_constraints is True:
            #     problem.generate_all_constraints()
            #     print('[CUTTING-PLANE] All constraints will be used.')

            if current_constraints is not None:
                self.add_constraints(current_constraints)  # restore constraints

            # if use_all_constraints is True:
            #     break

            self.update_constraints(w)
            if self.all_constraints_satisfied is True:
                print('[CUTTING-PLANE] All constraints satisfied.')
                break
            elif num_cons == self.get_num_constraints():
                print('[CUTTING-PLANE] No more effective constraints, violations are considered acceptable by IPOPT.')
                break
            else:
                current_constraints = self.get_current_constraints()

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
        # self.xi = w[(U + N + 1) * D:(U + N + 1) * D + N]
        # self.di = w[self.num_vars - N:self.num_vars]
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

    # def generate_all_constraints(self):
    #     """
    #         generate all possible constraints
    #     """
    #     sys.stdout.write('Generating all possible constraints ... ')
    #     M, N = self.M, self.N
    #     Y = self.Y
    #     assert Y.dtype == np.bool
    #     for n in range(M):
    #         for k in range(N):
    #             if Y[n, k] < 1:
    #                 self.current_constraints[k].add(n)
    #     print("%d constraints in total." % self.get_num_constraints())
