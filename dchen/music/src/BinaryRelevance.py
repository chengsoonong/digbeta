from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class BinaryRelevance(BaseEstimator):
    """
        Independent logistic regression based on OneVsRestClassifier wrapper.
    """

    def __init__(self, C=1, n_jobs=-1):
        assert C > 0
        self.C = C
        self.n_jobs = n_jobs
        self.trained = False

    def fit(self, X_train, Y_train):
        assert X_train.shape[0] == Y_train.shape[0]
        # don't make two changes at the same time
        # self.estimator = OneVsRestClassifier(LogisticRegression(class_weight='balanced', C=self.C))
        self.estimator = OneVsRestClassifier(LogisticRegression(C=self.C), n_jobs=self.n_jobs)
        self.estimator.fit(X_train, Y_train)
        self.trained = True

    def decision_function(self, X_test):
        assert self.trained is True
        return self.estimator.decision_function(X_test)

    def predict(self, X_test, binarise=False):
        preds = self.decision_function(X_test)
        return preds >= 0 if binarise is True else preds
