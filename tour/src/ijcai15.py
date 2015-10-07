import math
import re
import random
import numpy as np
import statistics as stat
#from datetime import datetime

class PersTour:
    """Reproduce the IJCAI'15 paper"""


    def __init__(self, dirname, fname, fpoi=None, writeFile=True):
        """Class Initialization"""
        # load records from file
        # each record contains tuples ("photoID";"userID";"dateTaken";"poiID";"poiTheme";"poiFreq";"seqID")
        self.dirname = dirname
        self.records = [] 
        self.load_records(self.dirname + '/' + fname)
        #idx = 9000
        #ousr = self.records[idx, 1]
        #opoi = self.records[idx, 3]
        #ocat = self.records[idx, 4]
        #oseq = self.records[idx, 6]
        #print()
        #print(self.records[idx], '\n')

        # build and save id maps
        self.poimap = dict()
        self.seqmap = dict()
        self.usrmap = dict()
        self.catmap = dict()
        self.build_maps()
        
        if writeFile:
            self.save_maps()

        # replace id in each record
        # (photoID, userID, dateTaken, poiID, poiTheme, poiFreq, seqID)
        for i, item in enumerate(self.records):
            self.records[i] = (item[0], self.usrmap[item[1]], item[2], self.poimap[item[3]], 
                    self.catmap[item[4]], item[5], self.seqmap[item[6]])
        #print(ousr, '->', self.usrmap[ousr])
        #print(opoi, '->', self.poimap[opoi])
        #print(ocat, '->', self.catmap[ocat])
        #print(oseq, '->', self.seqmap[oseq], '\n')
        #print(self.records[idx], '\n')

        # build (POI -> Category) and (Sequence -> User) mapping
        self.poicat = dict()
        self.sequsr = dict()
        for item in self.records:
            self.poicat[item[3]] = item[4]
            self.sequsr[item[6]] = item[1]

        self.adtime = dict() # arrival and departure time for each (usr, poi, seq)
        self.poi_pop           = np.zeros(len(self.poimap), dtype=np.int32)                     # POI popularity
        self.avg_poi_visit     = np.zeros(len(self.poimap), dtype=np.float64)                   # average POI visit duration
        #self.pers_poi_visit    = np.zeros((len(self.usrmap), len(self.poimap)), dtype=np.float64) # Personalized POI visit duration
        self.time_usr_interest = np.zeros((len(self.usrmap), len(self.catmap)), dtype=np.float64) # Time-based user interest
        self.freq_usr_interest = np.zeros((len(self.usrmap), len(self.catmap)), dtype=np.int32)   # Frequency-based user interest
        
        # expand sequences
        self.sequences = dict()
        seqset = {x for x in range(len(self.seqmap))}
        self.calc_adtime(seqset)
        self.expand_sequences()
        
        if writeFile:
            with open(self.dirname + '/' + 'seq.list', 'w') as f:
                for seq in range(len(self.seqmap)):
                    f.write(str(seq) + ' ' + str(self.sequences[seq]) + '\n')

        # calculate travel time
        self.traveltime = np.zeros((len(self.poimap), len(self.poimap)), dtype=np.float64)   # travel costs
        coordsfile = self.dirname + '/' + fname + '.coord'
        if fpoi:
            self.calc_traveltime(coordsfile, self.dirname + '/' + fpoi)
        else:
            self.calc_traveltime(coordsfile)

        # recommended sequences
        self.recommendSeqs = dict()


    def load_records(self, fname):
        """Load Dataset"""
        with open(fname, 'r') as f:
            for line in f:
                item = line.split(';')
                assert len(item) == 7
                photoID   = int(item[0])
                userID    = str(item[1][1:-1])  # get rid of double quotes
                dateTaken = int(item[2]) # Unix timestamp
                #dateTaken = datetime.fromtimestamp(int(item[2]))  # OK: convert it to date time format
                poiID     = int(item[3])
                poiTheme  = str(item[4][1:-1])
                poiFreq   = int(item[5])
                seqID     = int(item[6])
                self.records.append((photoID, userID, dateTaken, poiID, poiTheme, poiFreq, seqID))


    def build_maps(self):
        """Map POI ID, Sequence ID, User ID and Category ID to non-negative integers"""
        assert(len(self.records) > 0)

        usrset = set()
        poiset = set()
        catset = set()
        seqset = set()
        for item in self.records:
            usrset.add(item[1])
            poiset.add(item[3])
            catset.add(item[4])
            seqset.add(item[6])

        self.usrmap = {k: i for i, k in enumerate(sorted(usrset))}
        self.poimap = {k: i for i, k in enumerate(sorted(poiset))}
        self.catmap = {k: i for i, k in enumerate(sorted(catset))}
        self.seqmap = {k: i for i, k in enumerate(sorted(seqset))}


    def save_maps(self):
        """Save Mapped POI ID, Sequence ID, User ID and Category ID"""
        assert(len(self.usrmap)  > 0)
        assert(len(self.poimap)  > 0)
        assert(len(self.catmap)  > 0)
        assert(len(self.seqmap)  > 0)

        fusrmap = self.dirname + '/usr.map'
        fpoimap = self.dirname + '/poi.map'
        fcatmap = self.dirname + '/cat.map'
        fseqmap = self.dirname + '/seq.map'

        with open(fpoimap, 'w') as f:
            #for k, v in self.poimap.items():
            for k in sorted(self.poimap.keys()):
                f.write(str(k) + ':' + str(self.poimap[k]) + '\n')

        with open(fseqmap, 'w') as f:
            for k in sorted(self.seqmap.keys()):
                f.write(str(k) + ':' + str(self.seqmap[k]) + '\n')

        with open(fusrmap, 'w') as f:
            for k in sorted(self.usrmap.keys()):
                f.write(str(k) + ':' + str(self.usrmap[k]) + '\n')

        with open(fcatmap, 'w') as f:
            for k in sorted(self.catmap.keys()):
                f.write(str(k) + ':' + str(self.catmap[k]) + '\n')


    def calc_adtime(self, trainseqset):
        """Calculate arrival and departure time for each (poi, seq)"""
        assert(len(self.records) > 0)
        assert(len(self.poimap)  > 0)
        assert(len(self.seqmap)  > 0)
        assert(len(self.usrmap)  > 0)
        assert(len(self.catmap)  > 0)

        # relational mapping
        # * user <-(1:n)-> sequence
        # * sequence <-(m:n)-> POI
        # * user <-(m:n)-> POI
        for item in self.records:
            usr = item[1]
            poi = item[3]
            seq = item[6]
            if seq not in trainseqset: continue

            if (usr, seq, poi) not in self.adtime:
                self.adtime[usr, seq, poi] = np.zeros(2, dtype=np.int64)

            # arrival time, pick the earlist one
            if self.adtime[usr, seq, poi][0] == 0 or self.adtime[usr, seq, poi][0] > item[2]:
                self.adtime[usr, seq, poi][0] = item[2]

            # departure time, pick the latest one
            if self.adtime[usr, seq, poi][1] == 0 or self.adtime[usr, seq, poi][1] < item[2]:
                self.adtime[usr, seq, poi][1] = item[2]


    def expand_sequences(self):
        """Build the list of ordered POIs for each sequence"""
        for key in self.adtime.keys():
            seq = key[1]
            poi = key[2]
            if seq not in self.sequences:
                self.sequences[seq] = []
            self.sequences[seq].append(poi)

        for seq in range(len(self.seqmap)):
            usr = self.sequsr[seq]
            self.sequences[seq].sort(key=lambda poi:self.adtime[usr, seq, poi][0]) # sort POI by arrival time


    def calc_traveltime(self, coordsfile, fpoi=None):
        """Calculate travel time between each pair of POIs"""
        # load photo coordinates
        coord_records = []
        apoiset = set()
        with open(coordsfile, 'r') as f:
            for line in f:
                item = line.split(':')  # (poiID, photoID, longitude, latitude)
                assert len(item) == 4
                poiID   = int(item[0])
                longi   = float(item[2])
                lati    = float(item[3])
                coord_records.append((poiID, longi, lati));
                apoiset.add(poiID)

        assert(len(apoiset) == len(self.poimap))

        # load or calculate POI coordinate (as the average of coordinates of all photos assigned)
        longitudes = np.zeros(len(self.poimap), dtype=np.float64)
        latitudes  = np.zeros(len(self.poimap), dtype=np.float64)
        if fpoi:
            with open(fpoi, 'r') as f:
                for line in f:
                    item = line.split(',')
                    poiId = int(item[0])
                    lng = float(item[1])
                    lat = float(item[2])
                    assert(poiId in self.poimap)
                    poi = self.poimap[poiId]
                    longitudes[poi] = lng
                    latitudes [poi] = lat
        else:
            photo_cnt  = np.zeros(len(self.poimap), dtype=np.int32)
            for item in coord_records:
                poi = self.poimap[item[0]]
                photo_cnt [poi] += 1
                longitudes[poi] += item[1]
                latitudes [poi] += item[2]
            for poi in range(len(self.poimap)):
                assert(photo_cnt[poi] > 0)
                longitudes[poi] /= photo_cnt[poi]
                latitudes [poi] /= photo_cnt[poi]
            with open(self.dirname + '/poi.coord', 'w') as f:
                for poi in range(len(self.poimap)):
                    cat = self.poicat[poi]
                    catstr = 'NONE'
                    for k, v in self.catmap.items(): 
                        if v == cat: catstr = k
                    f.write(str(longitudes[poi]) + ',' + str(latitudes[poi]) + ',' + catstr + '\n')
                    #f.write(str(longitudes[poi]) + ',' + str(latitudes[poi]) + ',' + str(self.poi_pop[poi]) + ',' + catstr + '\n')


        # convert degrees to radians
        for poi in range(len(self.poimap)):
            longitudes[poi] = math.radians(longitudes[poi])
            latitudes [poi] = math.radians(latitudes [poi])

        # calculate travel time between two POIs
        speed = 4.        # 4km/h according to paper
        radius = 6371.009 # mean earth radius is 6371.009km, en.wikipedia.org/wiki/Earth_radius#Mean_radius
        for poi1 in range(len(self.poimap)):
            for poi2 in range(poi1+1, len(self.poimap)):
                # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
                dlat  = math.fabs(latitudes [poi1] - latitudes [poi2])
                dlong = math.fabs(longitudes[poi1] - longitudes[poi2])
                distance = 2 * radius * math.asin( 
                                            math.sqrt( 
                                                (math.sin(0.5*dlat))**2 + 
                                                math.cos(latitudes[poi1])*math.cos(latitudes[poi2]) *
                                                (math.sin(0.5*dlong))**2 
                                            ))
                self.traveltime[poi1, poi2] = 60*60*distance / speed      # seconds
                self.traveltime[poi2, poi1] = self.traveltime[poi1, poi2] # symmetrical
        #np.savetxt('dist.txt', self.traveltime, delimiter=',')


    def init_params(self):
        """Initialize parameters"""
        self.adtime.clear()
        for idx in range(len(self.poi_pop)): self.poi_pop[idx] = 0
        for idx in range(len(self.avg_poi_visit)): self.avg_poi_visit[idx] = 0
        #for r in range(np.shape(self.pers_poi_visit)[0]):
        #    for c in range(np.shape(self.pers_poi_visit)[1]): self.pers_poi_visit[r, c] = 0
        for r in range(np.shape(self.time_usr_interest)[0]):
            for c in range(np.shape(self.time_usr_interest)[1]): self.time_usr_interest[r, c] = 0
        for r in range(np.shape(self.freq_usr_interest)[0]):
            for c in range(np.shape(self.freq_usr_interest)[1]): self.freq_usr_interest[r, c] = 0


    def calc_metrics(self, trainseqset):
        """Calculate various metrics"""
        # calculate the popularity of each POI
        for item in self.records:
            poi  = item[3]
            freq = item[5]
            seq  = item[6]
            if seq not in trainseqset: continue
            self.poi_pop[poi] = freq

        # calculate average POI visit duration
        for item in self.records:
            usr = item[1]
            poi = item[3]
            seq = item[6]
            if seq not in trainseqset: continue
            self.avg_poi_visit[poi] += self.adtime[usr, seq, poi][1] - self.adtime[usr, seq, poi][0]
        for poi in range(len(self.poimap)):
            visit_cnt = self.poi_pop[poi]
            assert(visit_cnt > 0)
            #if self.avg_poi_visit[poi] == 0: print(poi) # just one photo taken at this POI for each visited user
            self.avg_poi_visit[poi] /= visit_cnt

        # calculate Time-based user interest
        for item in self.records:
            usr = item[1]
            poi = item[3]
            cat = item[4]
            seq = item[6]
            if seq not in trainseqset: continue
            term = 0
            if self.avg_poi_visit[poi] == 0: # if the average POI visit duration is 0 and a user visits this POI
                term = 2 # set his/her time-based interest as a constant
            else:
                term = (self.adtime[usr, seq, poi][1] - self.adtime[usr, seq, poi][0]) / self.avg_poi_visit[poi]
            self.time_usr_interest[usr, cat] += term

        # calculate Personalized POI visit duration
        #for usr in range(len(self.usrmap)):
        #    for poi in range(len(self.poimap)):
        #        cat = self.poicat[poi]
        #        self.pers_poi_visit[usr, poi] = self.time_usr_interest[usr, cat] * self.avg_poi_visit[poi]

        # calculate Frequency-based user interest
        for item in self.records:
            usr = item[1]
            cat = item[4]
            seq = item[6]
            if seq not in trainseqset: continue
            self.freq_usr_interest[usr, cat] += 1




    def calc_seqbudget(self, usr, seqpoilist, time_based=True):
        assert(len(seqpoilist) > 1)
        """Calculate the travel budget for the given travelling sequence"""

        budget = 0. # travel budget
        for i in range(len(seqpoilist)-1):
            px = seqpoilist[i]
            py = seqpoilist[i+1]
            assert(px in range(len(self.poimap)))
            assert(py in range(len(self.poimap)))
            budget += self.traveltime[px, py]
            cat = self.poicat[py]
            if time_based:
                budget += self.time_usr_interest[usr, cat] * self.avg_poi_visit[py]
            else:
                budget += self.freq_usr_interest[usr, cat] * self.avg_poi_visit[py]
        return budget


    def calc_seqscore(self, usr, seqpoilist, eta, time_based=True):
        """Calculate the score of a user's travelling sequence"""
        assert(len(seqpoilist) > 1)
        assert(0 <= eta <= 1)
        assert(usr in range(len(self.usrmap)))

        score = 0.0 # sequence score: the objective in ILP
        for i in range(1, len(seqpoilist)):
            px = seqpoilist[i]
            assert(px in range(len(self.poimap)))
            cat = self.poicat[px]
            interest = 0
            if time_based: interest = self.time_usr_interest[usr, cat]
            else:          interest = self.freq_usr_interest[usr, cat]
            score += eta * interest + (1. - eta) * self.poi_pop[px]
        return score


    def breakdown_time(self, usr, seqpoilist, time_based=True):
        """Breakdown time spent into visit duration and travelling"""
        assert(len(seqpoilist) > 1)
        assert(usr in range(len(self.usrmap)))

        usr_interest = None
        if time_based:
            usr_interest = self.time_usr_interest
        else:
            usr_interest = self.freq_usr_interest

        times = []
        for i in range(len(seqpoilist)-1):
            px = seqpoilist[i]
            py = seqpoilist[i+1]
            assert(px in range(len(self.poimap)))
            assert(py in range(len(self.poimap)))
            catx = self.poicat[px]
            caty = self.poicat[py]

            px0 = None # the actual POI ID in dataset
            py0 = None
            for k, v in self.poimap.items():
                if px0 and py0: break
                if v == px: px0 = k
                if v == py: py0 = k

            if i == 0:
                times.append((px0, round(usr_interest[usr, catx] * self.avg_poi_visit[px] / 60))) # time (minute) spent at POI px
            times.append((px0, py0, round(self.traveltime[px, py] / 60)))                         # travelling time from px to py
            times.append((py0, round(usr_interest[usr, caty] * self.avg_poi_visit[py] / 60)))     # time spent at POI py
        return times


    def get_realpoi(self, poilist):
        """Replace mapped POIs to real POI ID in dataset"""
        pois = []
        for pid in poilist:
            for k, v in self.poimap.items():
                if pid == v: 
                    pois.append(k)
                    break
        return pois


    def BruteForce_recommend(self, testseq, eta, seqfname, time_based=True):
        """Recommend by enumerating all possible short trajectories given an existing travel sequence S_N, 
           the first/last POI and travel budget calculated based on S_N
        """
        assert(testseq in range(len(self.seqmap)))
        assert(0 <= eta <= 1)

        if (len(self.sequences[testseq]) < 3):
            print('WARN: Given sequence is too short! NO recommendation!')
            return
        
        usr = self.sequsr[testseq]       # user of this sequence
        p0 = self.sequences[testseq][0]   # the first POI
        pN = self.sequences[testseq][-1]  # the last  POI

        budget = self.calc_seqbudget(usr, self.sequences[testseq], time_based)

        # brute force approach
        # enumerating all possible trajectories with length 3
        seqlist = []
        for pi in range(len(self.poimap)):
            if pi == p0 or pi == pN: continue
            newbudget = self.calc_seqbudget(usr, [p0, pi, pN], time_based)
            if newbudget > budget: continue
            score = self.calc_seqscore(usr, [p0, pi, pN], eta, time_based)
            times = self.breakdown_time(usr, [p0, pi, pN])
            seqlist.append(([p0, pi, pN], times, score))
        
        # enumerating all possible trajectories with length 4
        for pi in range(len(self.poimap)):
            if pi == p0 or pi == pN: continue
            for pj in range(len(self.poimap)):
                if pj in {p0, pN, pi}: continue
                newbudget = self.calc_seqbudget(usr, [p0, pi, pj, pN], time_based)
                if newbudget > budget: continue
                score = self.calc_seqscore(usr, [p0, pi, pj, pN], eta, time_based)
                times = self.breakdown_time(usr, [p0, pi, pj, pN])
                seqlist.append(([p0, pi, pj, pN], times, score))

        # enumerating all possible trajectories with length 5
        for pi in range(len(self.poimap)):
            if pi == p0 or pi == pN: continue
            for pj in range(len(self.poimap)):
                if pj in {p0, pN, pi}: continue
                for pk in range(len(self.poimap)):
                    if pk in {p0, pN, pi, pj}: continue
                    newbudget = self.calc_seqbudget(usr, [p0, pi, pj, pk, pN], time_based)
                    if newbudget > budget: continue
                    score = self.calc_seqscore(usr, [p0, pi, pj, pk, pN], eta, time_based)
                    times = self.breakdown_time(usr, [p0, pi, pj, pk, pN])
                    seqlist.append(([p0, pi, pj, pk, pN], times, score))


        # sort candidate sequences according to scores
        seqlist.sort(key=lambda item:item[2], reverse=True)

        # calculation for the actual sequence
        #seqset = {x for x in range(len(self.sequences))}
        #self.init_params()
        #self.calc_adtime(seqset)  # re-calculate the arrival and departure time, including sequence 'testseq'
        #self.calc_metrics(seqset) # re-calculate metrics, including information from sequence 'testseq'
        ascore = self.calc_seqscore(usr, self.sequences[testseq], eta, time_based)

        # rank of the actual sequence
        rank = 0
        for i, t in enumerate(seqlist):
            if abs(ascore - t[2]) < 1e-6:
                rank = i + 1
                break
        assert(rank != 0)
        atimes = []
        for i in range(len(self.sequences[testseq])-1):
            px = self.sequences[testseq][i]
            py = self.sequences[testseq][i+1]
            px0 = None # the actual POI ID in dataset
            py0 = None
            for k, v in self.poimap.items():
                if px0 and py0: break
                if v == px: px0 = k
                if v == py: py0 = k
            if i == 0:
                atimes.append((px0, round((self.adtime[usr, testseq, px][1] - self.adtime[usr, testseq, px][0]) / 60)))
            atimes.append((px0, py0, round(self.traveltime[px, py] / 60)))
            atimes.append((py0, round((self.adtime[usr, testseq, py][1] - self.adtime[usr, testseq, py][0]) / 60)))
        
        # write to file
        with open(seqfname, 'w') as f:
            f.write(str(self.get_realpoi(self.sequences[testseq])) + ', ' + str(atimes) + ', ' + str(ascore) + \
                    ' Actual, Rank ' + str(rank) +  '/' + str(len(seqlist)) + '\n')
            for item in seqlist:
                f.write(str(self.get_realpoi(item[0])) + ', ' + str(item[1]) + ', ' + str(item[2]) + '\n')


    def recommend(self, eta, time_based=True):
        """Trajectory Recommendation"""
        assert(0.0 <= eta <= 1.0)

        lpFileDir = self.dirname + '/eta'
        if   round(eta, 1) == 0.0: lpFileDir += '00'
        elif round(eta, 1) == 0.5: lpFileDir += '05'
        elif round(eta, 1) == 1.0: lpFileDir += '10'

        if time_based:
            if eta > 0.0: lpFileDir += '_time'
        else:
            if eta > 0.0: lpFileDir += '_freq'

        for seq in range(len(self.sequences)):
            if len(self.sequences[seq]) < 3: continue
            lpFileName = lpFileDir + '/' + str(seq) + '.lp'
            print('write', lpFileName)

            trainseqset = {x for x in range(len(self.sequences)) if x != seq}
            self.init_params()
            self.calc_adtime(trainseqset)
            self.calc_metrics(trainseqset)


    def recommend_bf(self, eta, time_based=True):
        """Trajectory Recommendation"""
        assert(0.0 <= eta <= 1.0)

        lpFileDir = self.dirname + '/eta'
        if   round(eta, 1) == 0.0: lpFileDir += '00.bf'
        elif round(eta, 1) == 0.5: lpFileDir += '05'
        elif round(eta, 1) == 1.0: lpFileDir += '10'

        if time_based:
            if eta > 0.0: lpFileDir += '_time.bf'
        else:
            if eta > 0.0: lpFileDir += '_freq.bf'

        # no leave-one-out
        seqset = {x for x in range(len(self.sequences))}
        self.init_params()
        self.calc_adtime(seqset)
        self.calc_metrics(seqset)
 
        for seq in range(len(self.sequences)):
            if len(self.sequences[seq]) < 3 or len(self.sequences[seq]) > 5: continue
            lpFileName = lpFileDir + '/' + str(seq) + '.seq.list'
            print('write', lpFileName)

            #trainseqset = {x for x in range(len(self.sequences)) if x != seq}
            #self.init_params()
            #self.calc_adtime(trainseqset)
            #self.calc_metrics(trainseqset)
            self.BruteForce_recommend(seq, eta, lpFileName, time_based)
 

    def evaluate(self, fseqlist, subdir):
        """Evaluate the recommended itinerary"""
        self.load_sequences(fseqlist, subdir)
        self.calc_evalmetrics()
        self.poicat_stat()

        with open(self.dirname + '/seq_a_r.list', 'w') as f:
            for seq in sorted(self.recommendSeqs.keys()):
                f.write(str(seq) + 'A:' + str(self.sequences[seq]) + '\n')
                f.write(str(seq) + 'R:' + str(self.recommendSeqs[seq]) + '\n')


    def load_sequences(self, fseqlist, subdir):
        """Load original and recommended sequences from file"""
        fname = self.dirname + '/' + fseqlist
        sequences = dict()
        with open(fname, 'r') as f:
            for line in f:
                item = line.split('[')
                assert(len(item) == 2)
                k = int(item[0])
                v = [int(x) for x in list(item[1].strip()[:-1].split(','))]  # e.g. 0 [1, 2, 3, 4, 5]\n
                sequences[k] = v

        # check consistency
        for k, v in sequences.items():
            assert(v == self.sequences[k])

        # load recommended itineraries from MIP solution files
        for k, v in self.sequences.items():
            if len(v) >= 3:
                fsol = self.dirname + '/' + subdir + '/' + str(k) + '.lp.sol'
                #self.recommendSeqs[k] = self.load_recommend_scip(v, fsol)
                self.recommendSeqs[k] = self.load_recommend_gurobi(v, fsol)
                if len(v) != len(self.recommendSeqs[k]):
                    print(k, ':', v)
                    print(k, ':', self.recommendSeqs[k])


    def load_recommend_scip(self, origseq, fsol):
        """Load recommended itinerary from MIP solution file by SCIP"""
        seqterm = []
        with open(fsol, 'r') as f:
            for line in f:
                if re.search('^visit_', line):      # e.g. visit_0_7              1 \t(obj:0)\n
                    item = line.split(' ')          #      visit_21_16            1.56406801399038e-09 \t(obj:125)\n
                    #item = re.sub('\t', ' ', line).split(' ')
                    words = []
                    for i in range(len(item)):
                        if len(item[i]) > 0: 
                            words.append(item[i])
                            if len(words) >= 2: break
                    if round(float(words[1])) == 1:
                        dummy = words[0].split('_')
                        seqterm.append((int(dummy[1]), int(dummy[2])))
        p0 = origseq[0]
        pN = origseq[-1]
        recseq = [p0]
    
        while True:
            px = recseq[-1]
            for term in seqterm:
                if term[0] == px:
                    recseq.append(term[1])
                    if term[1] == pN: return recseq
                    seqterm.remove(term)


    def load_recommend_gurobi(self, origseq, fsol):
        """Load recommended itinerary from MIP solution file by GUROBI"""
        seqterm = []
        with open(fsol, 'r') as f:
            for line in f:
                if re.search('^visit_', line):      # e.g. visit_0_7 1\n
                    item = line.strip().split(' ')  #      visit_21_16 1.56406801399038e-09\n, remove '\n' using strip()
                    if round(float(item[1])) == 1:
                        dummy = item[0].split('_')
                        seqterm.append((int(dummy[1]), int(dummy[2])))
        p0 = origseq[0]
        pN = origseq[-1]
        recseq = [p0]
    
        while True:
            px = recseq[-1]
            for term in seqterm:
                if term[0] == px:
                    recseq.append(term[1])
                    if term[1] == pN: return recseq
                    seqterm.remove(term)



    def calc_evalmetrics(self):
        """Calculate evaluation metrics: 
           Tour Recall, Tour Precision, Tour F1-score, RMSE of POI visit duration
           Tour Popularity, Tour Interest, Popularity and Interest Rank
        """
        assert(len(self.recommendSeqs) > 0)

        # calculate intersection size of recommended POI set and real POI set
        intersize = dict()
        for k, v in self.recommendSeqs.items():
            intersize[k] = len(set(v) & set(self.sequences[k]))

        # calculate tour recall
        recalls = []
        for k, v in intersize.items():
            recalls.append(v / len(self.sequences[k]))

        # calculate tour precision
        precisions = []
        for k, v in intersize.items():
            precisions.append(v / len(self.recommendSeqs[k]))

        # calculate F1-score
        f1scores = []
        assert(len(recalls) == len(precisions))
        for i in range(len(recalls)):
            f1scores.append(2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]))

        print(len(recalls), len(precisions), len(f1scores))
        print('Recall:   ', stat.mean(recalls),    stat.stdev(recalls))
        print('Precision:', stat.mean(precisions), stat.stdev(precisions))
        print('F1-score: ', stat.mean(f1scores),   stat.stdev(f1scores))

        #print(recalls)
        #print(precisions)
        #print(f1scores)
        np.savetxt(self.dirname + '/r.txt', recalls, delimiter=',')
        np.savetxt(self.dirname + '/p.txt', precisions, delimiter=',')
        np.savetxt(self.dirname + '/f1.txt', f1scores, delimiter=',')

        # calculate Root Mean Square Error of POI visit duration
        # calculate tour popularity
        # calculate tour interest
        # calculate popularity and interest rank



    def poicat_stat(self):
        """Calculate the transition matrix of POI category for both actual and recommended itineraries"""
        assert(len(self.recommendSeqs) > 0)

        catstat_visit = np.zeros((len(self.catmap), len(self.catmap)), dtype=np.float) # for actual itineraries
        catstat_rec   = np.zeros((len(self.catmap), len(self.catmap)), dtype=np.float) # for recommended itineraries

        print('#visit-actual:', len(self.sequences))
        print('#visit-recomm:', len(self.recommendSeqs))
        
        for k, v in self.recommendSeqs.items():
            for pi in range(len(v)-1):
                cati = self.poicat[v[pi]]
                catj = self.poicat[v[pi+1]]
                catstat_rec[cati, catj] += 1
            #for pj in range(len(self.sequences[k])-1):  # ignore actual sequences whose length is less than 3
            #     cati = self.poicat[self.sequences[k][pj]]
            #     catj = self.poicat[self.sequences[k][pj+1]]
        for k, v in self.sequences.items(): # do NOT ignore actual sequences whose length is less than 3
            for pj in range(len(v)-1):
                cati = self.poicat[v[pj]]
                catj = self.poicat[v[pj+1]]
                catstat_visit[cati, catj] += 1
        # normalize each row to get the transition probability from cati to catj
        for r in range(np.shape(catstat_rec)[0]):
            total = np.sum(catstat_rec[r])
            for c in range(np.shape(catstat_rec)[1]):
                catstat_rec[r, c] /= total
        for r in range(np.shape(catstat_visit)[0]):
            total = np.sum(catstat_visit[r])
            for c in range(np.shape(catstat_visit)[1]):
                catstat_visit[r, c] /= total

        print('transition matrix for recommended sequences:')
        print(catstat_rec)
        print('transition matrix for actual sequences:')
        print(catstat_visit)


    def ext_transmat(self):
        """Calculate the transition matrix of POI category for actual itineraries with a special category REST
           For a specific user, the different between the end time of the earlier sequence and the start time of
           latter sequence is less 24 hours, then their is a REST state between the two sequences
        """

        mat = np.zeros((len(self.catmap)+1, len(self.catmap)+1), dtype=np.float64)

        for k, v in self.sequences.items():
            for pj in range(len(v)-1):
                cati = self.poicat[v[pj]]
                catj = self.poicat[v[pj+1]]
                mat[cati, catj] += 1

        # for REST state
        usrseq = dict()
        restidx = len(self.catmap)
        for seq in self.sequences.keys():
            usr = self.sequsr[seq]
            if usr not in usrseq: usrseq[usr] = []
            usrseq[usr].append(seq)
        for usr, seqlist in usrseq.items():
            # sort according to the arrival time of the first POI of sequence
            seqlist.sort(key=lambda seq: self.adtime[usr, seq, self.sequences[seq][0]][0]) 
            for i in range(len(seqlist)-1):
                seq1 = seqlist[i]
                seq2 = seqlist[i+1]
                poi1 = self.sequences[seq1][-1]  # last POI of the earlier sequence
                poi2 = self.sequences[seq2][0] # first POI of the latter sequence
                dtime = self.adtime[usr, seq1, poi1][1]
                atime = self.adtime[usr, seq2, poi2][0]
                cat1 = self.poicat[poi1]
                cat2 = self.poicat[poi2]
                assert(atime > dtime)
                if atime - dtime < 24 * 60 * 60: # poi1-->REST-->poi2
                    mat[cat1, restidx] += 1
                    mat[restidx, cat2] += 1
                else: # poi1-->REST-->REST
                    mat[cat1, restidx] += 1
                    mat[restidx, restidx] += 1

        # normalize each row to get the transition probability from cati to catj
        for r in range(mat.shape[0]):
            total = np.sum(mat[r])
            for c in range(mat.shape[1]):
                mat[r, c] /= total

        np.savetxt(self.dirname + '/ext.transmat.txt', mat, delimiter=',')
        print('extended transition matrix for actual sequences:')
        cats = ''
        for k, v in self.catmap.items():
            cats += '(' + str(k) + ', ' + str(v) + ')  '
        print(cats)
        print(mat)


    def gen_transmat(self, seqset):
        """Calculate the transition matrix of POI category for given itineraries"""
        assert(len(seqset) > 0)
        transmat = np.zeros((len(self.catmap), len(self.catmap)), dtype=np.float)

        for k, v in self.sequences.items():
            if k not in seqset: continue
            for pj in range(len(v)-1):
                cati = self.poicat[v[pj]]
                catj = self.poicat[v[pj+1]]
                transmat[cati, catj] += 1

        # normalize each row to get the transition probability from cati to catj
        for r in range(transmat.shape[0]):
            total = np.sum(transmat[r])
            transmat[r] /= total

        return transmat

