import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from scipy.sparse import coo_matrix


def obj_toppush(w, X, Y, C, r=1, weighting=True):
    """
        Objective with L2 regularisation and top push loss
        
        Input:
            - w: current weight vector, flattened L x D
            - X: feature matrix, N x D
            - Y: label matrix,   N x K
            - C: regularisation constant, C = 1 / lambda
            - r: parameter for log-sum-exp approximation
            - weighting: if True, divide K+ in top-push loss
    """
    N, D = X.shape
    K = Y.shape[1]
    assert(w.shape[0] == K * D)
    assert(r > 0)
    assert(C > 0)
    
    W = w.reshape(K, D)  # theta
    
    J = 0.0  # cost
    G = np.zeros_like(W)  # gradient matrix
    
    # instead of using diagonal matrix to scale each row of a matrix with a different factor,
    # we use Mat * Vec[:, None] which is more memory efficient
    
    if weighting is True:
        KPosAll = np.sum(Y, axis=1)  # number of positive labels for each example, N by 1
    else:
        KPosAll = np.ones(N)
        
    A_diag = 1.0 / KPosAll
    #AY = Y * A_diag[:, None]
    A = coo_matrix(([0], ([0], [0])), shape=(N, N), dtype=np.float).tolil()
    A.setdiag(A_diag)
    AY = np.dot(A, Y.tocsr())
    
    T1 = np.dot(X, W.T)  # N by K
    #m0 = np.max(T1)  # underflow in np.exp(r*T1 - m1)
    m0 = 0.5 * (np.max(T1) + np.min(T1))
    m1 = r * m0
    #print('----------------')
    #print(m0, np.min(T1))
    
    T2 = np.multiply(1 - Y, np.exp(r * T1 - m1))  # N by K
    B_tilde_diag = np.dot(T2, np.ones(K))
    #print(np.max(B_tilde_diag), np.min(B_tilde_diag))  # big numbers here, can cause overflow in T3
    
    #T3 = np.exp(-T1 + m0) * np.power(B_tilde_diag, 1.0 / r)[:, None]
    #T4 = np.multiply(AY, np.log1p(T3))
    T3 = (-T1 + m0) + (1.0 / r) * np.log(B_tilde_diag)[:, None]
    #print(np.min(T3), np.max(T3))
    m2 = 0.5 * (np.min(T3) + np.max(T3))
    #T4 = np.logaddexp(0, T3)
    T4 = np.logaddexp(-m2, T3-m2) + m2
    T5 = np.multiply(AY, T4)  
    
    J = np.dot(w, w) * 0.5 / C + np.dot(np.ones(N), np.dot(T5, np.ones(K))) / N
    
    #T5 = 1.0 / (1.0 + np.divide(1.0, T3))
    #T5 = np.divide(T3, 1 + T3)
    T6 = np.exp(T3 - T4)
    O_diag = np.dot(np.multiply(Y, T6), np.ones(K))
    T7 = A_diag * (1.0 / B_tilde_diag) * O_diag
    
    G1 = np.dot(np.multiply(AY, T6).T, -X)
    G2 = np.dot((T2 * T7[:, None]).T, X)
    
    G = W / C + (G1 + G2) / N
    
    return (J, G.ravel())

class TopPushMLC(BaseEstimator):
    """All methods are necessary for a scikit-learn estimator"""
    
    def __init__(self, C=1, r=1, weighting=True):
        """Initialisation"""
        
        assert C > 0
        assert r > 0
        assert type(weighting) == bool
        self.C = C
        self.r = r
        self.weighting = weighting
        self.trained = False
        
    def fit(self, X_train, Y_train):
        """Model fitting by optimising the objective"""
        opt_method = 'L-BFGS-B' #'BFGS' #'Newton-CG'
        options = {'disp': 1, 'maxiter': 10**5, 'maxfun': 10**5} # , 'iprint': 99}
        print('\nC: %g, r: %g' % (self.C, self.r))
            
        N, D = X_train.shape
        K = Y_train.shape[1]
        #w0 = np.random.rand(K * D) - 0.5  # initial guess in range [-1, 1]
        w0 = 0.001 * np.random.randn(K * D)
        opt = minimize(obj_toppush, w0, args=(X_train, Y_train, self.C, self.r, self.weighting), \
                       method=opt_method, jac=True, options=options)
        if opt.success is True:
            self.W = np.reshape(opt.x, (K, D))
            self.trained = True
        else:
            sys.stderr.write('Optimisation failed')
            print(opt.items())
            self.trained = False
            
            
    def decision_function(self, X_test):
        """Make predictions (score is real number)"""
        
        assert self.trained is True, "Can't make prediction before training"
        D = X_test.shape[1]
        return np.dot(X_test, self.W.T)
        
    
    def predict(self, X_test):
        return self.decision_function(X_test)
    #    """Make predictions (score is boolean)"""   
    #    preds = sigmoid(self.decision_function(X_test))
    #    #return (preds >= 0)
    #    assert self.TH is not None
    #    return preds >= self.TH        
        
    # inherit from BaseEstimator instead of re-implement
    #
    #def get_params(self, deep = True):
    #def set_params(self, **params):
