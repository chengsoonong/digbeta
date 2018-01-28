import sys
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator


def avgF1(Y_true, Y_pred):
    # THs = [0, 0.05, 0.10, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75]  # SPEN THs
    THs = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    F1 = Parallel(n_jobs=-1)(delayed(f1_score_nowarn)(Y_true, Y_pred >= th, average='samples') for th in THs)
    bestix = np.argmax(F1)
    print('best threshold: %g, best F1: %g, #examples: %g' % (THs[bestix], F1[bestix], Y_true.shape[0]))
    return F1[bestix]

def obj_pclassification(w, X, Y, C, p, weighting=True):
    """
        Objective with L2 regularisation and p-classification loss
        
        Input:
            - w: current weight vector, flattened L x D + 1 (bias)
            - X: feature matrix, N x D
            - Y: label matrix,   N x L
            - C: regularisation constant, is consistent with scikit-learn C = 1 / (N * \lambda)
            - p: constant for p-classification push loss
    """
    N, D = X.shape
    K = Y.shape[1]
    assert(w.shape[0] == K * D + 1)
    assert(p >= 1)
    assert(C > 0)
    
    W = w[1:].reshape(K, D)  # reshape weight matrix
    b = w[0]           # bias
    OneN = np.ones(N)  # N by 1
    OneK = np.ones(K)  # K by 1
    
    if weighting is True:
        #KPosAll = np.sum(Y, axis=1)  # number of positive labels for each example, N by 1
        KPosAll = np.dot(Y, OneK)
        KNegAll = K - KPosAll        # number of negative labels for each example, N by 1
    else:
        KPosAll = np.ones(N)
        KNegAll = np.ones(N)
    A_diag = np.divide(1, KPosAll)  # N by 1
    P_diag = np.divide(1, KNegAll)  # N by 1
    
    #T1 = np.dot(X, W.T)  # N by K
    T1 = np.dot(X, W.T) + b # N by K
    
    T1p = np.multiply(Y, T1)
    T2 = np.multiply(Y, np.exp(-T1p))  # N by K
    T3 = T2 * A_diag[:, None]  # N by K
    
    #T1n = np.multiply(1-Y, T1)
    T1n = T1 - T1p
    T4 = np.multiply(1-Y, np.exp(p * T1n))  # N by K
    T5 = T4 * P_diag[:, None]  # N by K
    
    J = np.dot(W.ravel(), W.ravel()) * 0.5 / C 
    J += (np.dot(OneN, np.dot(T3, OneK)) + np.dot(OneN, np.dot(T5/p, OneK))) / N
    #J = np.dot(W.ravel(), W.ravel()) * 0.5 / C + (np.dot(OneN, np.dot(T3 + T5/p, OneK))) / N  # not as efficient
    
    #G = W / C + (np.dot(T3.T, -X) + np.dot(T5.T, X)) / N
    G = W / C + (np.dot((-T3 + T5).T, X)) / N   # more efficient
    
    db = np.dot(OneN, np.dot(-T3 + T5, OneK)) / N
    
    gradients = np.concatenate(([db], G.ravel()), axis=0) 
    
    return (J, gradients)


class MLC_pclassification(BaseEstimator):
    """All methods are necessary for a scikit-learn estimator"""
    
    def __init__(self, C=1, p=1, weighting=True):
        """Initialisation"""
        
        assert C >  0
        assert p >= 1
        self.C = C
        self.p = p
        self.weighting = weighting
        self.obj_func = obj_pclassification
        self.trained = False
        
    def fit(self, X_train, Y_train):
        """Model fitting by optimising the objective"""
        opt_method = 'L-BFGS-B' #'BFGS' #'Newton-CG'
        options = {'disp': 1, 'maxiter': 10**5, 'maxfun': 10**5} # , 'iprint': 99}
        sys.stdout.write('\nC: %g, p: %g, weighting: %s\n' % (self.C, self.p, self.weighting))
        sys.stdout.flush()
            
        N, D = X_train.shape
        K = Y_train.shape[1]
        #w0 = np.random.rand(K * D + 1) - 0.5  # initial guess in range [-1, 1]
        w0 = 0.001 * np.random.randn(K * D + 1)
        opt = minimize(self.obj_func, w0, args=(X_train, Y_train, self.C, self.p, self.weighting), \
                       method=opt_method, jac=True, options=options)
        if opt.success is True:
            self.b = opt.x[0]
            self.W = np.reshape(opt.x[1:], (K, D))
            self.trained = True
        else:
            sys.stderr.write('Optimisation failed')
            print(opt.items())
            self.trained = False
            
            
    def decision_function(self, X_test):
        """Make predictions (score is real number)"""
        
        assert self.trained is True, "Can't make prediction before training"
        D = X_test.shape[1]
        return np.dot(X_test, self.W.T) + self.b  # log of prediction score
        
    
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
