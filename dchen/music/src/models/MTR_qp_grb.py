import sys
import time
import gzip
import numpy as np
import pickle as pkl
from scipy.sparse import isspmatrix_csc
from gurobipy import quicksum, Model, GRB, QuadExpr, LinExpr


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
        # self.current_constraints = [set() for _ in range(self.N)]
        self.grb_cons = []
        self.inactive_cnt = []  # number of survives from constraints pruning
        self.inactive_threshold = 10
        self.all_constraints_satisfied = False

        self.mu = np.zeros(self.D)
        self.V = np.zeros((self.U, self.D))
        self.W = np.zeros((self.N, self.D))
        self.xi = np.zeros(self.N)
        self.delta = np.zeros(self.N)
        self.tol = 1e-6

        self._create_model()
        self.trained = False

    def _get_num_vars(self):
        return self.num_vars

    def _get_num_constraints(self):
        return len(self.grb_cons)

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
        # most efficient approach: use QuadExpr.addTerms() and LinExpr.addTerms()
        qexpr = QuadExpr()
        lexpr = LinExpr()

        coefU = np.ones(U * D) * 0.5 * C1 / U
        varsU = [V[u, d] for u in range(U) for d in range(D)]
        qexpr.addTerms(coefU, varsU, varsU)

        coefW = np.ones(N * D) * 0.5 * C2 / N
        varsW = [W[k, d] for k in range(N) for d in range(D)]
        qexpr.addTerms(coefW, varsW, varsW)

        coefmu = np.ones(D) * 0.5 * C3
        varsmu = [mu[d] for d in range(D)]
        qexpr.addTerms(coefmu, varsmu, varsmu)

        lexpr.addTerms(Q, [delta[k] for k in range(N)])
        obj = qexpr + lexpr

        # alternative approach
#         obj = quicksum(V[u, d] * V[u, d] for u in range(U) for d in range(D)) * 0.5 * C1 / U
#         obj += quicksum(W[k, d] * W[k, d] for k in range(N) for d in range(D)) * 0.5 * C2 / N
#         obj += quicksum(mu[d] * mu[d] for d in range(D)) * 0.5 * C3
#         obj += quicksum(Q[k] * delta[k] for k in range(N))

        qp.setObjective(obj, GRB.MINIMIZE)

        # create constraints
        for k in range(N):
            # constraint: \delta_k >= \sum_{m: y_m^k=1} (1 - f(u(k), k, m) + \xi_k)
            # which is equivalent to
            # M_+^k * (1 + \xi_k) - \delta_k - (v_{u(k)} + w_k + \mu)^T \sum_{m: y_m^k=1} x_m <= 0
            u = self.pl2u[k]
            y = self.Y[:, k].A.reshape(-1)
            vec = y.dot(self.X)
            c_k = qp.addConstr(Mplus[k] * (1 + xi[k]) - delta[k]
                               - quicksum((V[u, d] + W[k, d] + mu[d]) * vec[d] for d in range(D)) <= 0)
            self.grb_cons.append(c_k)
            self.inactive_cnt.append(0)
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

    def _generate_all_constraints(self):
        """
            generate all possible constraints
        """
        sys.stdout.write('Generating all possible constraints ... ')
        M, N, D = self.M, self.N, self.D
        grb_V, grb_W, grb_mu, grb_xi = self.grb_V, self.grb_W, self.grb_mu, self.grb_xi
        grb_qp = self.grb_qp
        assert self.Y.dtype == np.bool
        for k in range(N):
            u = self.pl2u[k]
            grb_qp.addConstrs((quicksum((grb_V[u, d] + grb_W[k, d] + grb_mu[d]) * self.X[n, d]
                                        for d in range(D)) - grb_xi[k] <= 0
                              for n in range(M) if self.Y[n, k] < 1))
        print('%d constraints in total.' % grb_qp.getAttr('NumConstrs'))

    def _update_constraints(self):
        N, D, U = self.N, self.D, self.U
        grb_qp, grb_V, grb_W, grb_mu, grb_xi = self.grb_qp, self.grb_V, self.grb_W, self.grb_mu, self.grb_xi
        # grb_delta = self.grb_delta

        self.mu[:] = [grb_mu[d].x for d in range(D)]
        for u in range(U):
            self.V[u, :] = [grb_V[u, d].x for d in range(D)]
        for k in range(N):
            self.W[k, :] = [grb_W[k, d].x for d in range(D)]
        self.xi[:] = [grb_xi[k].x for k in range(N)]
        # self.delta[:] = [grb_delta[k].x for k in range(N)]

        mu, V, W, xi = self.mu, self.V, self.W, self.xi
        Ys, _, Mplus = self.data_helper.get_data()
        assert U == len(Ys)
        assert Mplus.shape == (N,)

        all_satisfied = True
        # for k in range(N):
        #     # constraint: \delta_k >= \sum_{m: y_m^k=1} (1 - f(u(k), k, m) + \xi_k)
        #     u = self.pl2u[k]
        #     y = self.Y[:, k].A.reshape(-1)
        #     vec = y.dot(self.X)
        #     if delta[k] < Mplus[k] * (1 + xi[k]) - vec.dot(V[u, :] + W[k, :] + mu):
        #         all_satisfied = False
        #         c_k = grb_qp.addConstr(Mplus[k] * (1 + grb_xi[k]) - grb_delta[k]
        #                                - quicksum((grb_V[u, d] + grb_W[k, d] + grb_mu[d]) * vec[d]
        #                                   for d in range(D)) <= 0)
        #         self.grb_cons.append(c_k)
        #         self.inactive_cnt.append(0)

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
                    c_kn = grb_qp.addConstr(quicksum((grb_V[u, d] + grb_W[k, d] + grb_mu[d]) * self.X[n, d]
                                            for d in range(D)) - grb_xi[k] <= 0)
                    self.grb_cons.append(c_kn)
                    self.inactive_cnt.append(0)
        self.all_constraints_satisfied = all_satisfied

    def _prune_constraints(self):
        prune_ix = []
        num_cons = len(self.grb_cons)
        # assert num_cons == len(self.inactive_cnt)
        if len(self.inactive_cnt) != num_cons:
            raise ValueError('len(inactive_cnt) != num_cons')
        for j in range(self.N, num_cons):  # keep the first N constraints
            cons = self.grb_cons[j]
            if abs(cons.slack) < 1e-10:
                self.inactive_cnt[j] += 1
                if self.inactive_cnt[j] > self.inactive_threshold:
                    prune_ix.append(j)
        if len(prune_ix) > 0:
            print('[CUTTING-PLANE] Pruning %d inactive constraints.' % len(prune_ix))
            for j in prune_ix:
                self.grb_qp.remove(self.grb_cons[j])
            prune_ix_set = set(prune_ix)
            self.grb_cons[:] = [self.grb_cons[ix] for ix in range(num_cons) if ix not in prune_ix_set]
            self.inactive_cnt[:] = [self.inactive_cnt[ix] for ix in range(num_cons) if ix not in prune_ix_set]

    def fit(self, max_iter=1e3, njobs=10, use_all_constraints=False, w0=None, fpkl=None):
        t0 = time.time()
        print(time.strftime('%Y-%m-%d %H:%M:%S'))

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
        self.grb_qp.Params.Threads = njobs
        self.grb_qp.Params.Presolve = 2
        print('[CUTTING-PLANE] %d variables, %d maximum possible constraints.' % (self.num_vars, max_num_cons))
        while cp_iter < max_iter:
            cp_iter += 1

            if use_all_constraints is True:
                self._generate_all_constraints()

            num_cons = self._get_num_constraints()
            print('[CUTTING-PLANE] Iter %d: %d constraints.' % (cp_iter, num_cons))

            self.grb_qp.optimize()

            # pruning constraints
            # if cp_iter > 50 and cp_iter % 10 == 0:
            #    self._prune_constraints()

            if use_all_constraints is True:
                break
            else:
                self._update_constraints()

            if self.all_constraints_satisfied is True:
                print('[CUTTING-PLANE] All constraints satisfied.')
                break
            if num_cons == self._get_num_constraints():
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
