import math
import sys
import random
import numpy as np


class Simulator:
    """Trajectory Simulation"""

    def __init__(self, fpoi, ftransMat):
        """Class Initialization"""
        self.poidata = self.load_poi(fpoi)
        self.transMat, self.catdata = self.load_mat(ftransMat) # with extra state 'REST' at the last row/column
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
        prevcat = random.randint(0, len(self.catdata)-2) # do not start at 'REST'
        prevpoi = random.choice(self.catpoidict[prevcat])
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
                nextpoi = random.choice(self.catpoidict[nextcat])
            obsMat[prevcat, nextcat] += 1
            n += 1
            prevcat = nextcat
            prevpoi = nextpoi
        return obsMat

 
    def simulate_Pop(self, N, randomized=False):
        """Trajectory simulation, generate N observations"""
        obsMat = np.zeros(np.shape(self.transMat), dtype=np.int32)

        # random.randint(a, b) returns a random integer N such that a <= N <= b.
        prevcat = random.randint(0, len(self.catdata)-2) # do not start at 'REST'
        prevpoi = random.choice(self.catpoidict[prevcat])
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
                nextpoi = random.choice(self.catpoidict[nextcat])
            obsMat[prevcat, nextcat] += 1
            n += 1
            prevcat = nextcat
            prevpoi = nextpoi
        return obsMat

   
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
                cat = str(t[3])    # category
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
                    for x in t:
                        cat = str(x.strip())
                        catset.add(cat)
                        catdata.append(cat)
                else:
                    row = []
                    assert(ncols == len(t))
                    for x in t: row.append(float(x.strip()))
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
            pvals[i] = self.calc_dist(lng1, lat1, lng2, lat2)
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
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'POI_FILE  TRANSITION_MATRIX_FILE')
        sys.exit(0)

    fpoi = sys.argv[1]
    ftrans = sys.argv[2]
    N = 500000
    sim = Simulator(fpoi, ftrans)
    #obs = sim.simulate_NN(N)
    #obs = sim.simulate_NN(N, randomized=True)
    #obs = sim.simulate_Pop(N)
    obs = sim.simulate_Pop(N, randomized=True)
    trans = np.zeros(obs.shape, dtype=np.float64)
    for r in range(obs.shape[0]):
        total = np.sum(obs[r])
        trans[r] = obs[r] / total
    print(obs)
    #print(sim.transMat)
    #print('-'*20)
    #print(trans)
    print('-'*20)
    print(sim.transMat - trans)
