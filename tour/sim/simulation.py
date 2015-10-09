import math
import sys
import random
import numpy as np


class Simulator:
    """Trajectory Simulation"""

    def __init__(self, poistr, matstr):
        """Class Initialization"""
        self.rng = random.SystemRandom()
        #self.poidata = self.load_poi(fpoi)
        #self.transMat, self.catdata = self.load_mat(ftransMat) # with extra state 'REST' at the last row/column
        self.poidata = self.parse_poi(poistr)
        self.transMat, self.catdata = self.parse_mat(matstr) # with extra state 'REST' at the last row/column
        assert(self.transMat.shape[0] == self.transMat.shape[1])
        assert(self.transMat.shape[0] == len(self.catdata))

        # replace POI category with its category ID
        catmap = {self.catdata[i]: i for i in range(len(self.catdata))}
        for i in range(len(self.poidata)):
            cat = self.poidata[i][3]
            assert(cat in catmap)
            catid = catmap[cat]
            self.poidata[i] = (self.poidata[i][0], self.poidata[i][1], self.poidata[i][2], catid)

        self.catpoidict = dict()
        for i in range(len(self.poidata)):
            catid = self.poidata[i][3]
            if catid not in self.catpoidict: self.catpoidict[catid] = []
            self.catpoidict[catid].append(i)


    def simulate_NN(self, N, randomized=False):
        """Trajectory simulation, generate N observations"""
        obsMat = np.zeros(np.shape(self.transMat), dtype=np.int32)

        # random.randint(a, b) returns a random integer N such that a <= N <= b.
        prevcat = self.rng.randint(0, len(self.catdata)-2) # do not start at 'REST'
        prevpoi = self.rng.choice(self.catpoidict[prevcat])
        nextcat = None
        nextpoi = None
      
        n = 0
        while n < N:
            nextcat = self.get_nextcat(prevcat)
            if self.catdata[nextcat] == 'REST': # if the next category is 'REST', then no POI is available to choose?
                nextpoi = -1 # -1 is the ID of a special POI for category 'REST'
            elif self.catdata[prevcat] != 'REST':
                assert(prevpoi != -1)
                if randomized: 
                    nextpoi = self.rule_RandNN(prevpoi, nextcat)
                else:
                    nextpoi = self.rule_NN(prevpoi, nextcat)
            else:
                nextpoi = self.rng.choice(self.catpoidict[nextcat])
            obsMat[prevcat, nextcat] += 1
            n += 1
            prevcat = nextcat
            prevpoi = nextpoi
        return obsMat

 
    def simulate_Pop(self, N, randomized=False):
        """Trajectory simulation, generate N observations"""
        obsMat = np.zeros(np.shape(self.transMat), dtype=np.int32)

        # random.randint(a, b) returns a random integer N such that a <= N <= b.
        prevcat = self.rng.randint(0, len(self.catdata)-2) # do not start at 'REST'
        prevpoi = self.rng.choice(self.catpoidict[prevcat])
        nextcat = None
        nextpoi = None
      
        n = 0
        while n < N:
            nextcat = self.get_nextcat(prevcat)
            if self.catdata[nextcat] == 'REST': # if the next category is 'REST', then no POI is available to choose?
                nextpoi = -1 # -1 is the ID of a special POI for category 'REST'
            elif self.catdata[prevcat] != 'REST':
                assert(prevpoi != -1)
                if randomized: 
                    nextpoi = self.rule_RandPop(nextcat)
                else:
                    nextpoi = self.rule_Pop(nextcat)
            else:
                nextpoi = self.rng.choice(self.catpoidict[nextcat])
            obsMat[prevcat, nextcat] += 1
            n += 1
            prevcat = nextcat
            prevpoi = nextpoi
        return obsMat


    def estimate_MLE(self, obsMat):
        """Maximum Likelihood Estimation"""
        assert(np.shape(obsMat) == self.transMat.shape)
        assert((obsMat >= 0).all())

        estMat = np.zeros(np.shape(obsMat), dtype=np.float64)
        for r in range(np.shape(obsMat)[0]):
            estMat[r] = obsMat[r] / np.sum(obsMat[r])
        return estMat

   
    def load_poi(self, fname):
        """Load POI data: geo-coordinates, category"""
        poidata = []

        with open(fname, 'r') as f:
            for line in f:
                t = line.strip().split(',')
                assert(len(t) == 4)
                lng = float(t[0])  # longitude 
                lat = float(t[1])  # latitude
                pop = int(t[2])    # popularity
                cat = t[3]         # category
                poidata.append((lng, lat, pop, cat))
        return poidata


    def load_mat(self, fname):
        """Load Transition Matrix of a Markov Chain"""
        ncols = 0
        firstRow = True
        catset = set()
        catdata = []
        data = []

        with open(fname, 'r') as f:
            for line in f:
                t = line.strip().split(',')
                if firstRow: # POI categories at the first row
                    ncols = len(t)
                    firstRow = False
                    catset = {x.strip() for x in t}
                    catdata = [x.strip() for x in t]
                else:
                    assert(ncols == len(t))
                    row = [float(x.strip()) for x in t]
                    data.append(row)
        assert(len(catset) == len(catdata))
        assert(catdata[-1] == 'REST') # state 'REST' at the last row/column
        return np.array(data), catdata


    def parse_poi(self, poistr):
        """Load POI data: geo-coordinates, category"""
        poidata = []
        lines = poistr.split('\n')
        for line in lines:
            t = line.strip().split(',')
            assert(len(t) == 4)
            lng = float(t[0])  # longitude 
            lat = float(t[1])  # latitude
            pop = int(t[2])    # popularity
            cat = t[3]         # category
            poidata.append((lng, lat, pop, cat))
        return poidata


    def parse_mat(self, matstr):
        """Load Transition Matrix of a Markov Chain"""
        ncols = 0
        firstRow = True
        catset = set()
        catdata = []
        data = []
        lines = matstr.split('\n')
        for line in lines:
            t = line.strip().split(',')
            if firstRow: # POI categories at the first row
                ncols = len(t)
                firstRow = False
                catset = {x.strip() for x in t}
                catdata = [x.strip() for x in t]
            else:
                assert(ncols == len(t))
                row = [float(x.strip()) for x in t]
                data.append(row)
        assert(len(catset) == len(catdata))
        assert(catdata[-1] == 'REST') # state 'REST' at the last row/column
        return np.array(data), catdata


    def calc_dist(self, longitude1, latitude1, longitude2, latitude2):
        """Calculate the distance (unit: km) between two places on earth"""
        # convert degrees to radians
        lng1 = math.radians(longitude1)
        lat1 = math.radians(latitude1)
        lng2 = math.radians(longitude2)
        lat2 = math.radians(latitude2)
        radius = 6371.009 # mean earth radius is 6371.009km, en.wikipedia.org/wiki/Earth_radius#Mean_radius

        # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
        dlng = math.fabs(lng1 - lng2)
        dlat = math.fabs(lat1 - lat2)
        dist =  2 * radius * math.asin( math.sqrt( 
                    (math.sin(0.5*dlat))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(0.5*dlng))**2 ))
        return dist


    def get_nextcat(self, curcatid):
        """Return the target category with probability proportional to the 
           transition probability from current category to target category
        """
        assert(curcatid in range(len(self.catdata)))
        pvals = self.transMat[curcatid]

        # catgorical/multinoulli distribution, special case of multinomial distribution (n=1)
        sample = np.random.multinomial(1, pvals)
        catid = -1
        for i in range(len(sample)):
            if sample[i] == 1:
                catid = i
                break
        assert(catid != -1)
        return catid


    def rule_NN(self, poiid, destcatid):
        """Nearest Neighbor Rule to select a POI within a certain category
           Return the index of target POI
        """
        assert(destcatid in self.catpoidict)
        assert(len(self.catpoidict[destcatid]) > 0)
        mindist = -1
        destpoi = -1
        lng1 = self.poidata[poiid][0]
        lat1 = self.poidata[poiid][1]

        for pid in self.catpoidict[destcatid]:
            lng2 = self.poidata[pid][0]
            lat2 = self.poidata[pid][1]
            dist = self.calc_dist(lng1, lat1, lng2, lat2)
            if mindist == -1 or dist < mindist:
                mindist = dist
                destpoi = pid
        return destpoi


    def rule_Pop(self, destcatid):
        """Popularity Rule to select a POI within a certain category
           Return the index of target POI
        """
        assert(destcatid in self.catpoidict)
        assert(len(self.catpoidict[destcatid]) > 0)
        maxpop = -1
        destpoi = -1

        for pid in self.catpoidict[destcatid]:
            pop = self.poidata[pid][2]
            if maxpop == -1 or pop > maxpop:
                maxpop = pop
                destpoi = pid
        return destpoi


    def rule_RandNN(self, poiid, destcatid):
        """Randomized Nearest Neighbor Rule to select a POI within a certain category,
           POI was choosen with probability proportional to the reciprocal of its distance to the given POI
           Return the index of target POI
        """
        assert(destcatid in self.catpoidict)
        assert(len(self.catpoidict[destcatid]) > 0)
        lng1 = self.poidata[poiid][0]
        lat1 = self.poidata[poiid][1]
        pvals = np.zeros(len(self.catpoidict[destcatid]), dtype=np.float64)

        for i in range(len(self.catpoidict[destcatid])):
            pid = self.catpoidict[destcatid][i]
            lng2 = self.poidata[pid][0]
            lat2 = self.poidata[pid][1]
            pvals[i] = 1 / (self.calc_dist(lng1, lat1, lng2, lat2) + 0.001) # taking care of 0
        pvals /= np.sum(pvals)
 
        # catgorical/multinoulli distribution, special case of multinomial distribution (n=1)
        sample = np.random.multinomial(1, pvals)
        
        destpoi = -1
        assert(len(sample) == len(self.catpoidict[destcatid]))
        for i in range(len(sample)):
            if sample[i] == 1: 
                destpoi = self.catpoidict[destcatid][i]
        assert(destpoi != -1)
        return destpoi

 
    def rule_RandPop(self, destcatid):
        """Randomized Popularity Rule to select a POI within a certain category,
           POI was choosen with probability proportional to its popularity
           Return the index of target POI
        """
        assert(destcatid in self.catpoidict)
        assert(len(self.catpoidict[destcatid]) > 0)
        pvals = np.zeros(len(self.catpoidict[destcatid]), dtype=np.float64)

        for i in range(len(self.catpoidict[destcatid])):
            pid = self.catpoidict[destcatid][i]
            pvals[i] = self.poidata[pid][2]
        pvals /= np.sum(pvals)
 
        # catgorical/multinoulli distribution, special case of multinomial distribution (n=1)
        sample = np.random.multinomial(1, pvals)
        
        destpoi = -1
        assert(len(sample) == len(self.catpoidict[destcatid]))
        for i in range(len(sample)):
            if sample[i] == 1: 
                destpoi = self.catpoidict[destcatid][i]
        assert(destpoi != -1)
        return destpoi


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage:', sys.argv[0], 'NUMBER_OF_STEP')
        sys.exit(0)

    N = int(sys.argv[1])
    #N = 500000
    poistr = """-79.3792433791,43.6431825014,3506,Sport
                -79.4186338571,43.6327716174,609,Sport
                -79.3800452078,43.6621752718,688,Sport
                -79.3892897441,43.6412973685,3056,Sport
                -79.3923959409,43.653662053,986,Cultural
                -79.3773273575,43.6471509752,2064,Cultural
                -79.3853489798,43.6423852627,1736,Cultural
                -79.3391695,43.7164471367,278,Cultural
                -79.3611074827,43.6670669191,346,Cultural
                -79.3944584714,43.6671825712,4142,Cultural
                -79.1822114886,43.8200801331,481,Cultural
                -79.4093638091,43.678156694,964,Cultural
                -79.4170006879,43.6339241915,141,Amusement
                -79.3735729115,43.6198358673,113,Amusement
                -79.387065479,43.6428491376,3553,Amusement
                -79.4160112537,43.6325628267,808,Amusement
                -79.4508076154,43.6371833846,26,Amusement
                -79.3782267477,43.6216974955,111,Beach
                -79.462381764,43.6465571685,89,Beach
                -79.3804532715,43.656274007,3594,Beach
                -79.3837019012,43.6524782093,3619,Beach
                -79.3798840402,43.6538681522,1874,Shopping
                -79.3823200204,43.6386213833,1028,Shopping
                -79.4011678236,43.654747843,1701,Shopping
                -79.4526832115,43.7257607115,104,Shopping
                -79.3909345778,43.6701035397,631,Shopping
                -79.3811841774,43.6521810267,936,Structure
                -79.3912647227,43.6621384805,744,Structure
                -79.3805837906,43.6456505403,1538,Structure"""

    matstr = """Amusement,Beach,Cultural,Shopping,Sport,Structure,REST
                3.043478260869565341e-02,3.913043478260869873e-02,1.152173913043478271e-01,3.695652173913043653e-02,\
                8.043478260869564578e-02,3.260869565217391214e-02,6.652173913043478715e-01
                1.473839351510685368e-02,3.168754605747973324e-02,4.568901989683124554e-02,6.705969049373618207e-02,\
                1.400147383935151056e-02,7.590272660280029948e-02,7.509211495946941373e-01
                3.054989816700610927e-02,4.752206381534283819e-02,2.647657841140529586e-02,4.684317718940936986e-02,\
                1.493550577053632047e-02,6.653088934147997902e-02,7.671418873048201359e-01
                1.310401310401310485e-02,8.190008190008189748e-02,4.995904995904995594e-02,1.310401310401310485e-02,\
                1.474201474201474252e-02,5.569205569205569473e-02,7.714987714987715517e-01
                5.148005148005147663e-02,2.960102960102960201e-02,2.702702702702702853e-02,1.673101673101673112e-02,\
                1.029601029601029567e-02,2.702702702702702853e-02,8.378378378378378288e-01
                2.813852813852813980e-02,1.017316017316017285e-01,8.766233766233766378e-02,6.709956709956710341e-02,\
                2.597402597402597574e-02,2.489177489177489197e-02,6.645021645021644829e-01
                1.051051051051051129e-02,2.402402402402402382e-02,3.453453453453453337e-02,1.909051909051908899e-02,\
                2.659802659802659730e-02,1.973401973401973236e-02,8.655083655083655181e-01"""

    sim = Simulator(poistr, matstr)
    #obs = sim.simulate_NN(N)
    obs = sim.simulate_NN(N, randomized=True)
    #obs = sim.simulate_Pop(N)
    #obs = sim.simulate_Pop(N, randomized=True)
    est = sim.estimate_MLE(obs)
    print('Observations')
    print(obs)
    print('-'*20)
    print('Transition Matrix Diff: Real - Estimated')
    print(sim.transMat - est)

