import math
import sys
import random
import numpy as np

class Estimation:
    """Estimate realMatition probabilities from observation,
       Generate new observation according to the estimator
    """

    def __init__(self):
        """Class Initialization"""
        self.nrows = 2
        self.ncols = 2
        self.realMat = np.zeros((self.nrows, self.ncols), dtype=np.float64)
        self.realMat[0, 0] = 0.2
        self.realMat[0, 1] = 1. - self.realMat[0, 0]
        self.realMat[1, 0] = 0.4
        self.realMat[1, 1] = 1. - self.realMat[1, 0]
        self.realMat.flags.writeable = False
        #print(self.realMat)


    def gen_observation(self, N, transMat=None):
        """Generate Observation according to a transition matrix"""
        if transMat == None: transMat = self.realMat
        obsMat = np.zeros(np.shape(transMat), dtype=np.int32)
        
        assert((transMat > 0.0).all())
        assert(N > 0)
        assert(np.shape(obsMat) == (2, 2))

        for i in range(N):
            if random.random() < transMat[0, 0]:
                obsMat[0, 0] += 1
            else: 
                obsMat[0, 1] += 1
            if random.random() < transMat[1, 0]:
                obsMat[1, 0] += 1
            else: 
                obsMat[1, 1] += 1
        return obsMat


    def estimate_MLE(self, obsMat):
        """Maximum Likelihood Esitmator"""
        assert(np.shape(obsMat) == np.shape(self.realMat))
        assert((obsMat >= 0).all())

        estMat = np.zeros(np.shape(obsMat), dtype=np.float64)
        for i in range(np.shape(obsMat)[0]):
            rowsum = np.sum(obsMat[i])
            for j in range(np.shape(obsMat)[1]):
                estMat[i, j] = obsMat[i, j] / rowsum
        return estMat

    
    def estimate_MAP(self, obsMat):
        """MAP estimation with Gaussian prior"""
        assert(np.shape(obsMat) == np.shape(self.realMat))
        assert((obsMat >= 0).all())

        estMat = np.zeros(np.shape(obsMat), dtype=np.float64)
        mu = 0.5
        sigma = 0.2

        pass;



    def main(self):
        """Main Procedure"""
        N = 1000000
        obsMat1 = self.gen_observation(N)
        estMat1 = self.estimate_MLE(obsMat1)
        obsMat2 = self.gen_observation(N, estMat1)

        print()
        print('Real Transition Matrix:')
        print(self.realMat)
        print()

        print('Observation 1:')
        print(obsMat1)
        print()

        print('Estimated Transition Matrix:')
        print(estMat1)
        print()

        print('Observation 2:')
        print(obsMat2)
        print()


if __name__ == '__main__':
    a = Estimation()
    a.main()
