import sys
import time
import gzip
import numpy as np
import pickle as pkl
from scipy.sparse import isspmatrix_csc
from gurobipy import quicksum, Model, GRB
# from gurobipy import QuadExpr, LinExpr
# from joblib import Parallel, delayed


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
        self.M, self.D = self.X.shape
        self.N = self.Y.shape[1]
        self.U = len(self.cliques)
        self.pl2u = np.zeros(self.N, dtype=np.int32)
        for u in range(self.U):
            clq = self.cliques[u]
            self.pl2u[clq] = u

        self.num_vars = (self.U + self.N + 1) * self.D + self.N
        self.data_helper = DataHelper(self.Y, self.cliques)

        # self.constraints_dict:
        # - a dictionary of N sets
        # - if `n` \in self.constraints_dict[k], it means `x_n` violates constraint `xi_k >= f(u(k), k, n)`
        # - it will be modified by _update_constraints() and _prune_constraints()
        self.constraints_dict = {k: set() for k in range(self.N)}

        # self.inactive_cnts:
        # - a dictionary with key (k, n), k \in {0,...,N-1}, n \in {0,...,M-1} and integer values
        # - inactive_cnts[(k, n)] represents the number of survives (from pruning) of constraint (k, n)
        # - it will be modified by _update_constraints() and _prune_constraints()
        self.inactive_cnts = dict()
        self.inactive_threshold = 10
        self.slack_threshold = 0.001
        self.gap = 0.1
        self.tol = 1e-6

        self.mu = np.zeros(self.D)
        self.V = np.zeros((self.U, self.D))
        self.W = np.zeros((self.N, self.D))
        self.xi = np.zeros(self.N)
        self.delta = np.zeros(self.N)

        self._create_model()
        self.all_constraints_satisfied = False
        self.trained = False

    def _get_num_vars(self):
        # return self.grb_qp.numVars
        return self.num_vars

    def _get_num_constraints(self):
        # return self.grb_qp.numConstrs  # could be incorrect due to gurobi lazy evaluation
        return self.N + np.sum([len(v) for k, v in self.constraints_dict.items()])

    def _create_model(self):
        """
            Create a QP model.
        """
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
        obj = quicksum(V[u, d] * V[u, d] for u in range(U) for d in range(D)) * 0.5 * C1 / U
        obj += quicksum(W[k, d] * W[k, d] for k in range(N) for d in range(D)) * 0.5 * C2 / N
        obj += quicksum(mu[d] * mu[d] for d in range(D)) * 0.5 * C3
        obj += quicksum(Q[k] * delta[k] for k in range(N))

        # alterative approach: use QuadExpr.addTerms() and LinExpr.addTerms()
        # it could be more efficient but less readable
        # qexpr = QuadExpr()
        # lexpr = LinExpr()
        # coefU = np.ones(U * D) * 0.5 * C1 / U
        # varsU = [V[u, d] for u in range(U) for d in range(D)]
        # qexpr.addTerms(coefU, varsU, varsU)
        # coefW = np.ones(N * D) * 0.5 * C2 / N
        # varsW = [W[k, d] for k in range(N) for d in range(D)]
        # qexpr.addTerms(coefW, varsW, varsW)
        # coefmu = np.ones(D) * 0.5 * C3
        # varsmu = [mu[d] for d in range(D)]
        # qexpr.addTerms(coefmu, varsmu, varsmu)
        # lexpr.addTerms(Q, [delta[k] for k in range(N)])
        # obj = qexpr + lexpr

        qp.setObjective(obj, GRB.MINIMIZE)

        # create constraints
        for k in range(N):
            # constraint: \delta_k >= \sum_{m: y_m^k=1} (1 - f(u(k), k, m) + \xi_k)
            # which is equivalent to
            # M_+^k * (1 + \xi_k) - \delta_k - (v_{u(k)} + w_k + \mu)^T \sum_{m: y_m^k=1} x_m <= 0
            u = self.pl2u[k]
            y = self.Y[:, k].A.reshape(-1)
            vec = y.dot(self.X)
            qp.addConstr(Mplus[k] * (1 + xi[k]) - delta[k]
                         - quicksum((V[u, d] + W[k, d] + mu[d]) * vec[d] for d in range(D)) <= 0)
        self.grb_qp = qp
        self.grb_obj = obj
        self.grb_V = V
        self.grb_W = W
        self.grb_mu = mu
        self.grb_xi = xi
        self.grb_delta = delta

    def _generate_all_constraints(self):
        """
            Generate all possible constraints:
                xi_k >= f(u(k), k, n), k \in {0,...N-1}, y_n^k = 0
        """
        sys.stdout.write('Generating all possible constraints ... ')
        M, N, D = self.M, self.N, self.D
        grb_qp, grb_V, grb_W, grb_mu, grb_xi = self.grb_qp, self.grb_V, self.grb_W, self.grb_mu, self.grb_xi
        assert self.Y.dtype == np.bool
        for k in range(N):
            u = self.pl2u[k]
            grb_qp.addConstrs((quicksum((grb_V[u, d] + grb_W[k, d] + grb_mu[d]) * self.X[n, d] for d in range(D))
                               - grb_xi[k] <= 0 for n in range(M) if self.Y[n, k] < 1))
        print('%d constraints in total.' % grb_qp.getAttr('NumConstrs'))

    def _update_constraints(self):
        """
            Check if current solution satisfies constraint:
                xi_k >= f(u(k), k, n), k \in {0,...N-1}, y_n^k = 0
            If not, add constraint:
                f(u(k), k, n^*) - xi_k <= 0
            where
                n^* = argmax_n f(u(k), k, n)
        """
        N, D, U = self.N, self.D, self.U
        grb_qp, grb_V, grb_W, grb_mu, grb_xi = self.grb_qp, self.grb_V, self.grb_W, self.grb_mu, self.grb_xi

        self.mu[:] = [grb_mu[d].x for d in range(D)]
        for u in range(U):
            self.V[u, :] = [grb_V[u, d].x for d in range(D)]
        for k in range(N):
            self.W[k, :] = [grb_W[k, d].x for d in range(D)]
        self.xi[:] = [grb_xi[k].x for k in range(N)]

        mu, V, W, xi = self.mu, self.V, self.W, self.xi
        Ys, _, Mplus = self.data_helper.get_data()
        assert U == len(Ys)
        assert Mplus.shape == (N,)

        # updated_dict = dict()

        all_satisfied = True
        for u in range(U):
            clq = self.cliques[u]
            Yu = Ys[u]
            Wt = W[clq, :] + (V[u, :] + mu).reshape(1, D)
            T1 = np.dot(self.X, Wt.T)
            T1[Yu.row, Yu.col] = -np.inf  # mask entry (m,i) if y_m^i = 1
            max_ix = T1.argmax(axis=0)
            assert max_ix.shape[0] == len(clq)
            for j in range(len(clq)):
                k = clq[j]
                n = max_ix[j]
                # if xi[k] < T1[n, j] or (xi[k] == T1[n, j] == 0):
                if xi[k] + self.tol < T1[n, j]:
                    all_satisfied = False
                    # feasibility cut at query point q for constraint f(z) <= 0:
                    # f(q) + f'(q)^T (z - q) <= 0
                    grb_qp.addConstr(
                        quicksum(T1[n, j] - xi[k]
                                 + (grb_V[u, d] - V[u, d] + grb_W[k, d] - W[k, d] + grb_mu[d] - mu[d]) * self.X[n, d]
                                 - (grb_xi[k] - xi[k]) for d in range(D)) <= 0, name='ckn_%d_%d' % (k, n))
                    self.constraints_dict[k].add(n)
                    self.inactive_cnts[(k, n)] = 0
            # sort_ix = (-T1).argsort(axis=0)
            # for i in range(2):
            #     for j in range(len(clq)):
            #         k = clq[j]
            #         n = sort_ix[i, j]
            #         if xi[k] + self.tol < T1[n, j]:
            #             all_satisfied = False
            #             grb_qp.addConstr(quicksum((grb_V[u, d] + grb_W[k, d] + grb_mu[d]) * self.X[n, d]
            #                                       for d in range(D)) - grb_xi[k] <= 0, name='ckn_%d_%d' % (k, n))
            #             self.constraints_dict[k].add(n)
            #             self.inactive_cnts[(k, n)] = 0

                    # try:
                    #     updated_dict[k].append(n)
                    # except KeyError:
                    #     updated_dict[k] = [n]
        # print(updated_dict)

        self.all_constraints_satisfied = all_satisfied

    def _prune_constraints(self, check_only=False):
        """
            Pruning inactive constraints:
                f(u(k), k, n) - xi_k <= 0
        """
        # linear_constrs = self.grb_qp.getConstrs()
        # num_cons = len(linear_constrs)
        #
        # def _check_inactive_cons(cons):
        #     if cons.slack > self.slack_threshold:
        #         cname = cons.constrName
        #         if cname.startswith('ckn_'):
        #             _, k, n = cname.split('_')
        #             k, n = int(k), int(n)
        #             return True, k, n
        #     return False, None, None
        #
        # if num_cons <= 0:
        #     return 0
        # tuples = Parallel(n_jobs=-1)(delayed(_check_inactive_cons)(linear_constrs[i]) for i in range(num_cons))
        # assert len(tuples) == len(num_cons)
        # inactive_keys = [(ix, tuples[1], tuples[2]) for ix in range(num_cons) if tuples[0] is True]
        # prune_keys = []
        # for ix, k, n in inactive_keys:
        #     try:
        #         self.inactive_cnts[(k, n)] += 1
        #         if check_only is False:
        #             if self.inactive_cnts[(k, n)] > self.inactive_threshold:
        #                 self.grb_qp.remove(linear_constrs[ix])
        #                 prune_keys.append((k, n))
        #     except KeyError:
        #         raise ValueError('ERROR: %s' % ('NO inactive counts for constraint (%d, %d)!' % (k, n)))
        # if check_only is False and len(prune_keys) > 0:
        #     for k, n in prune_keys:
        #         try:
        #             self.constraints_dict[k].remove(n)
        #             self.inactive_cnts.pop((k, n))
        #         except KeyError:
        #             raise ValueError('ERROR: %s' % ('constraint (%d, %d) not found!' % (k, n)))
        # return len(prune_keys)
        pruned_dict = dict()

        num_pruned = 0
        for cons in self.grb_qp.getConstrs():
            if cons.slack > self.slack_threshold:
                cname = cons.constrName
                if cname.startswith('ckn_'):
                    _, k, n = cname.split('_')
                    k, n = int(k), int(n)
                    try:
                        self.inactive_cnts[(k, n)] += 1
                        if check_only is False:
                            if self.inactive_cnts[(k, n)] > self.inactive_threshold:
                                self.grb_qp.remove(cons)
                                self.constraints_dict[k].remove(n)
                                self.inactive_cnts.pop((k, n))
                                num_pruned += 1

                                try:
                                    pruned_dict[k].append(n)
                                except KeyError:
                                    pruned_dict[k] = [n]
                    except KeyError:
                        raise ValueError('ERROR: %s' % ('NO inactive counts for constraint (%d, %d)!' % (k, n)))
        if check_only is False:
            print(pruned_dict)

        return num_pruned

    def fit(self, max_iter=1e3, njobs=10, use_all_constraints=False, verbose=0, fpkl=None):
        """
            Model fitting by solving a QP using cutting plane method in addition to a QP solver.
        """
        t0 = time.time()
        print(time.strftime('%Y-%m-%d %H:%M:%S'))
        M, N, = self.M, self.N
        if verbose > 0:
            self.grb_qp.Params.OutputFlag = 1
            print('\nC: %g, %g, %g' % (self.C1, self.C2, self.C3))
        else:
            self.grb_qp.Params.OutputFlag = 0

        # solve QP using a QP solver and cutting-plane method
        # first create an optimisation problem and solve it,
        # then add constraints violated by current solution,
        # create a new optimisation problem (same objective, update constraints)
        # keep doing this until termination criteria satisfied.
        cp_iter = 0
        max_num_cons = M * N - self.Y.sum() + N
        self.grb_qp.Params.Threads = njobs
        # self.grb_qp.Params.Presolve = 2
        print('[CUTTING-PLANE] %d variables, %d maximum possible constraints.' % (self._get_num_vars(), max_num_cons))
        while cp_iter < max_iter:
            cp_iter += 1

            if use_all_constraints is True:
                self._generate_all_constraints()

            num_cons = self._get_num_constraints()
            print('[CUTTING-PLANE] Iter %d: %d constraints.' % (cp_iter, num_cons))

            self.grb_qp.optimize()

            # pruning constraints
            # if cp_iter % 5 == 0:
            #     self._prune_constraints(check_only=True)
            # if cp_iter % 50 == 0:
            #     sys.stdout.write('[CUTTING-PLANE] Pruning inactive constraints ... ')
            #     num_pruned = self._prune_constraints(check_only=False)
            #     print('%d constaints pruned.' % num_pruned)

            if use_all_constraints is True:
                break
            else:
                self._update_constraints()

            if self.all_constraints_satisfied is True:
                print('[CUTTING-PLANE] All constraints satisfied.')
                break
            if num_cons == self._get_num_constraints():
                break
            #    print('[CUTTING-PLANE] No more strong cuts, starts to use weaker cuts')
            #    self.gap /= 2
            #    self._update_constraints()
            #     print('[CUTTING-PLANE] No more effective constraints, violations are considered acceptable.')
            #     break

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
        """
            Making predictions (score is a real number)
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
