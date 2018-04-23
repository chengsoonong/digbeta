import sys
import time
import gzip
import numpy as np
import pickle as pkl
from scipy.sparse import isspmatrix_csc
from gurobipy import quicksum, Model, GRB


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

        self.num_vars = (self.U + self.N + 1) * self.D + self.N
        self.data_helper = DataHelper(self.Y, self.cliques)

        # self.current_constraints:
        # - a list of N sets
        # - if `n` \in self.current_constraints[k], it means `x_n` violates constraint `f_{k,n} <= 0`
        # - it will be modified by self.update_constraints() and self.restore_constraints()
        self.current_constraints = [set() for _ in range(self.N)]
        self.num_constraints = self.N
        self.all_constraints_satisfied = False
        self.trained = False

        self._create_model()
        self.mu = np.zeros(self.D)
        self.V = np.zeros((self.U, self.D))
        self.W = np.zeros((self.N, self.D))
        self.xi = np.zeros(self.N)

    def _create_model(self):
        N, U, D, C1, C2, C3 = self.N, self.U, self.D, self.C1, self.C2, self.C3
        _, Q, Mplus = self.data_helper.get_data()
        assert Q.shape == Mplus.shape == (N,)

        # create model
        qp = Model()

        # create variables
        V = qp.addVars(U, D, name='V')
        W = qp.addVars(N, D, name='W')
        mu = qp.addVars(D, name='mu')
        xi = qp.addVars(N, name='xi')
        delta = qp.addVars(N, lb=0, name='delta')

        # create objective
        # more efficient way: QuadExpr.addTerms() and LinExpr.addTerms()
        obj = quicksum(V[u, d] * V[u, d] for u in range(U) for d in range(D)) * 0.5 * C1 / U
        obj += quicksum(W[k, d] * W[k, d] for k in range(N) for d in range(D)) * 0.5 * C2 / N
        obj += quicksum(mu[d] * mu[d] for d in range(D)) * 0.5 * C3
        obj += quicksum(Q[k] * delta[k] for k in range(N))
        qp.setObjective(obj, GRB.MINIMIZE)

        # create constraints
        for k in range(N):
            # constraint: \delta_k >= \sum_{m: y_m^k=1} (1 - f(u(k), k, m) + \xi_k)
            # which is equivalent to
            # M_+^k + M_+^k * \xi_k - (v_{u(k)} + w_k + \mu)^T \sum_{m: y_m^k=1} x_m <= \delta_k
            u = self.pl2u[k]
            yk = self.Y[:, k].A.reshape(-1)
            vec = yk.dot(self.X)
            qp.addConstr(Mplus[k] * (1 + xi[k]) - quicksum((V[u, d] + W[k, d] + mu[d]) * vec[d] for d in range(D))
                         <= delta[k])
        self.grb_qp = qp
        self.grb_obj = obj
        self.grb_V = V
        self.grb_W = W
        self.grb_mu = mu
        self.grb_xi = xi
        self.grb_delta = delta

        # control verbose output
        if self.verbose > 0:
            self.grb_qp.Params.OutputFlag = 1
        else:
            self.grb_qp.Params.OutputFlag = 0

    def _update_constraints(self):
        N, D, U = self.N, self.D, self.U
        grb_V, grb_W, grb_mu, grb_xi = self.grb_V, self.grb_W, self.grb_mu, self.grb_xi
        for d in range(D):
            self.mu[d] = grb_mu[d].x
        for u in range(U):
            for d in range(D):
                self.V[u, d] = grb_V[u, d].x
        for k in range(N):
            for d in range(D):
                self.W[k, d] = grb_W[k, d].x
        for k in range(N):
            self.xi[k] = grb_xi[k].x

        mu, V, W, xi = self.mu, self.V, self.W, self.xi
        Ys, _, _ = self.data_helper.get_data()
        assert U == len(Ys)

        all_satisfied = True
        for u in range(U):
            clq = self.cliques[u]
            Yu = Ys[u]
            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = np.dot(self.X, Wt.T)
            T1[Yu.row, Yu.col] = -np.inf  # mask entry (m,i) if y_m^i = 1
            max_ix = T1.argmax(axis=0)

            if self.verbose > 1:
                print(np.max(T1, axis=0))
                print(xi[clq])

            assert max_ix.shape[0] == len(clq)
            for j in range(max_ix.shape[0]):
                k = clq[j]
                n = max_ix[j]
                # if xi[k] + self.tol < T1[row, col]:
                if xi[k] < T1[n, j] or (xi[k] == T1[n, j] == 0):
                    all_satisfied = False
                    if n not in self.current_constraints[k]:
                        self.current_constraints[k].add(n)
                        self.num_constraints += 1
                        self.grb_qp.addConstr(quicksum((grb_V[u, d] + grb_W[k, d] + grb_mu[d]) * self.X[n, d]
                                                       for d in range(D)) <= grb_xi[k])
        self.all_constraints_satisfied = all_satisfied

    def fit(self, max_iter=1e3, use_all_constraints=False, w0=None, fpkl=None):
        t0 = time.time()
        M, N, = self.M, self.N
        verbose = self.verbose
        if verbose > 0:
            print('\nC: %g, %g, %g' % (self.C1, self.C2, self.C3))

        # solve QP using a QP solver and cutting-plane method
        # first create an optimisation problem and solve it,
        # then add constraints violated by current solution,
        # create a new optimisation problem (same objective, update constraints)
        # keep doing this until termination criteria satisfied.
        cp_iter = 0
        max_num_cons = M * N - self.Y.sum() + N
        print('[CUTTING-PLANE] %d variables, %d maximum possible constraints.' % (self.num_vars, max_num_cons))
        while cp_iter < max_iter:
            cp_iter += 1
            num_cons = self.num_constraints
            print('[CUTTING-PLANE] Iter %d: %d constraints.' % (cp_iter, num_cons))
            self.grb_qp.optimize()

            self._update_constraints()
            if self.all_constraints_satisfied is True:
                print('[CUTTING-PLANE] All constraints satisfied.')
                break
            if num_cons == self.num_constraints:
                print('[CUTTING-PLANE] No more effective constraints, violations are considered acceptable.')
                break

        if cp_iter >= max_iter:
            print('[CUTTING-PLANE] Reaching max number of iterations.')

        self.trained = True

        if fpkl is not None:
            try:
                pkl.dump([self.mu, self.V, self.W], gzip.open(fpkl, 'wb'))
                if verbose > 0:
                    print('Save to %s' % fpkl)
            except (OSError, IOError, ValueError) as err:
                sys.stderr.write('Save weights failed: {0}\n'.format(err))

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
