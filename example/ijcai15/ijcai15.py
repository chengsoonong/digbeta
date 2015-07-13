import math
import pulp
import numpy as np
#from datetime import date

class PersTour:
    """Reproduce the IJCAI'15 paper"""


    def __init__(self, dirname, fname):
        """Class Initialization"""
        # load records from file
        # each record contains tuples ("photoID";"userID";"dateTaken";"poiID";"poiTheme";"poiFreq";"seqID")
        self.records = [] 
        self.load_records(dirname + '/' + fname)
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
        self.save_maps(dirname)

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

        # calculate arrival and departure time for each (poi, seq)
        self.atime = np.zeros((len(self.poimap), len(self.seqmap)), dtype=np.int64) # arrival time for each (poi, seq)
        self.dtime = np.zeros((len(self.poimap), len(self.seqmap)), dtype=np.int64) # departure time for each (poi, seq)
        self.calc_adtime()

        # calculate metrics
        self.poi_pop           = np.zeros((len(self.poimap)), dtype=np.int32)                     # POI popularity
        self.avg_poi_visit     = np.zeros((len(self.poimap)), dtype=np.float64)                   # average POI visit duration
        self.pers_poi_visit    = np.zeros((len(self.usrmap), len(self.poimap)), dtype=np.float64) # Personalized POI visit duration
        self.time_usr_interest = np.zeros((len(self.usrmap), len(self.catmap)), dtype=np.float64) # Time-based user interest
        self.freq_usr_interest = np.zeros((len(self.usrmap), len(self.catmap)), dtype=np.int32)   # Frequency-based user interest
        self.calc_metrics()

        # expand sequences
        self.sequences = dict()
        self.expand_sequences()

        # calculate travel time
        self.traveltime = np.zeros((len(self.poimap), len(self.poimap)), dtype=np.float64)   # travel costs
        poicoordsfile = dirname + '/' + fname + '.coord'
        self.calc_traveltime(poicoordsfile)

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
                #dateTaken = date.fromtimestamp(int(item[2]))  # OK: convert it to date time format
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


    def save_maps(self, dirname):
        """Save Mapped POI ID, Sequence ID, User ID and Category ID"""
        assert(len(self.usrmap)  > 0)
        assert(len(self.poimap)  > 0)
        assert(len(self.catmap)  > 0)
        assert(len(self.seqmap)  > 0)

        fusrmap = dirname + '/usr.map'
        fpoimap = dirname + '/poi.map'
        fcatmap = dirname + '/cat.map'
        fseqmap = dirname + '/seq.map'

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


    def calc_adtime(self):
        """Calculate arrival and departure time for each (poi, seq)"""
        assert(len(self.records) > 0)
        assert(len(self.poimap)  > 0)
        assert(len(self.seqmap)  > 0)
        assert(len(self.usrmap)  > 0)
        assert(len(self.catmap)  > 0)

        for item in self.records:
            poi = item[3]
            seq = item[6]

            # arrival time, pick the earlist one for each poi
            if self.atime[poi, seq] == 0 or self.atime[poi, seq] > item[2]:
                self.atime[poi, seq] = item[2]

            # departure time, pick the latest one for each poi
            if self.dtime[poi, seq] == 0 or self.dtime[poi, seq] < item[2]:
                self.dtime[poi, seq] = item[2]


    def calc_metrics(self):
        """Calculate various metrics"""
        # calculate the popularity of each POI
        for poi in range(len(self.poimap)):
            for seq in range(len(self.seqmap)):
                if self.atime[poi, seq] < self.dtime[poi, seq]:
                    self.poi_pop[poi] += 1

        # calculate average POI visit duration
        for poi in range(len(self.poimap)):
            visit_cnt = 0
            for seq in range(len(self.seqmap)):
                if self.atime[poi, seq] < self.dtime[poi, seq]:
                    visit_cnt += 1
                    self.avg_poi_visit[poi] += self.dtime[poi, seq] - self.atime[poi, seq]
            if visit_cnt > 0:
                self.avg_poi_visit[poi] /= visit_cnt

        # calculate Time-based user interest
        for poi in range(len(self.poimap)):
            cat = self.poicat[poi]
            for seq in range(len(self.seqmap)):
                usr = self.sequsr[seq]
                if self.atime[poi, seq] < self.dtime[poi, seq]:
                    self.time_usr_interest[usr, cat] += (self.dtime[poi, seq] - self.atime[poi, seq]) / self.avg_poi_visit[poi]

        # calculate Personalized POI visit duration
        for usr in range(len(self.usrmap)):
            for poi in range(len(self.poimap)):
                cat = self.poicat[poi]
                self.pers_poi_visit[usr, poi] = self.time_usr_interest[usr, cat] * self.avg_poi_visit[poi]

        # calculate Frequency-based user interest
        for poi in range(len(self.poimap)):
            cat = self.poicat[poi]
            for seq in range(len(self.seqmap)):
                usr = self.sequsr[seq]
                if self.atime[poi, seq] < self.dtime[poi, seq]:
                    self.freq_usr_interest[usr, cat] += 1


    def expand_sequences(self):
        """Build the list of ordered POIs for each sequence"""
        for seq in range(len(self.seqmap)):
            poilist = []
            for poi in range(len(self.poimap)):
                if self.atime[poi, seq] < self.dtime[poi, seq]:
                    poilist.append(poi)
            poilist.sort(key=lambda poi:self.atime[poi, seq])  # sort POI by arrival time
            self.sequences[seq] = poilist


    def calc_traveltime(self, poicoordsfile):
        """Calculate travel time between each pair of POIs"""
        # load POI coordinates
        coord_records = []
        apoiset = set()
        with open(poicoordsfile, 'r') as f:
            for line in f:
                item = line.split(':')  # (poiID, photoID, longitude, latitude)
                assert len(item) == 4
                poiID   = int(item[0])
                longi   = float(item[2])
                lati    = float(item[3])
                coord_records.append((poiID, longi, lati));
                apoiset.add(poiID)

        assert(len(apoiset) == len(self.poimap))

        # calculate POI coordinate as the average of coordinates of all photos taking
        photo_cnt  = np.zeros((len(self.poimap)), dtype=np.int32)
        longitudes = np.zeros((len(self.poimap)), dtype=np.float64)
        latitudes  = np.zeros((len(self.poimap)), dtype=np.float64)
        for item in coord_records:
            poi = self.poimap[item[0]]
            photo_cnt [poi] += 1
            longitudes[poi] += item[1]
            latitudes [poi] += item[2]
        for poi in range(len(self.poimap)):
            assert(photo_cnt[poi] > 0)
            longitudes[poi] /= photo_cnt[poi]
            latitudes [poi] /= photo_cnt[poi]

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
                self.traveltime[poi1, poi2] = distance / speed            # hour
                self.traveltime[poi2, poi1] = self.traveltime[poi1, poi2] # symmetrical


    def MIP_recommend(self, seq, eta, time_based=True):
        """Recommend a trajectory given an existing travel sequence S_N, 
           the first/last POI and travel budget are calculated based on S_N
        """
        assert(0 <= seq < len(self.seqmap))
        assert(0 <= eta <= 1)

        if (len(self.sequences[seq]) < 3):
            print('WARN: Given sequence is too short! NO recommendation!')
            return
        
        usr = self.sequsr[seq]       # user of this sequence
        p0 = str(self.sequences[seq][0])   # the first POI
        pN = str(self.sequences[seq][-1])  # the last  POI
        N  = len(self.sequences[seq]) # number of POIs in the given sequence
        
        # reference to either self.time_usr_interest or self.freq_usr_interest
        usr_interest = []            
        if time_based:
            usr_interest = self.time_usr_interest
        else:
            usr_interest = self.freq_usr_interest

        budget = 0. # travel budget
        for i in range(len(self.sequences[seq])-1):
            px = self.sequences[seq][i]
            py = self.sequences[seq][i+1]
            budget += self.traveltime[px, py]
            cat = self.poicat[py]
            budget += usr_interest[usr, cat] * self.avg_poi_visit[py]

        # The MIP problem
        # REF: pythonhosted.org/PuLP/index.html
        # create a string list for each POI
        pois = [str(p) for p in range(len(self.poimap))] 

        # create problem
        prob = pulp.LpProblem('TourRecommendation', pulp.LpMaximize)
        
        # visit_i_j = 1 means POI i and j are visited in sequence
        visit_vars = pulp.LpVariable.dicts('visit', (pois, pois), 0, 1, pulp.LpInteger) 
        
        # a dictionary contains all dummy variables
        dummy_vars = pulp.LpVariable.dicts('u', [x for x in pois if x != p0], 2, N, pulp.LpInteger)

        # add objective
        prob += pulp.lpSum([visit_vars[pi][pj] * \
                                (eta * usr_interest[usr, self.poicat[int(pi)]] + (1. - eta) * self.poi_pop[int(pi)]) \
                                for pi in pois if pi != p0 and pi != pN \
                                for pj in pois if pj != p0 \
                                ]), 'Objective'

        # add constraints
        # each constraint should be in ONE line
        prob += pulp.lpSum([visit_vars[p0][pj] for pj in pois if pj != p0]) == 1, 'StartAtp0' # starts at the first POI
        prob += pulp.lpSum([visit_vars[pi][pN] for pi in pois if pi != pN]) == 1, 'EndAtpN'   # ends at the last POI
        for pk in [x for x in pois if x != p0 and x != pN]:
            prob += pulp.lpSum([visit_vars[pi][pk] for pi in pois if pi != pN]) == \
                    pulp.lpSum([visit_vars[pk][pj] for pj in pois if pj != p0]), \
                    'Connected_' + pk # the itinerary is connected
            prob += pulp.lpSum([visit_vars[pi][pk] for pi in pois if pi != pN]) <= 1, 'LeaveAtMostOnce_' + pk # ENTER POIk at most once
            prob += pulp.lpSum([visit_vars[pk][pj] for pj in pois if pj != p0]) <= 1, 'EnterAtMostOnce_' + pk # LEAVE POIk at most once
        prob += pulp.lpSum([visit_vars[pi][pj] * ( \
                            self.traveltime[int(pi), int(pj)] + \
                            usr_interest[usr, self.poicat[int(pj)]] * self.avg_poi_visit[int(pj)]) \
                            for pi in pois if pi != pN \
                            for pj in pois if pj != p0 \
                            ]) <= budget, 'WithinBudget' # travel cost should be within budget
        for pi in pois:
            if pi != p0:
                for pj in pois:
                    if pj != p0:
                        prob += dummy_vars[pi] - dummy_vars[pj] + 1 <= \
                                (N - 1) * (1 - visit_vars[pi][pj]), \
                                'SubTourElimination_' + str(pi) + '_' + str(pj) # TSP sub-tour elimination

        # write problem data to an .lp file
        prob.writeLP('TourRecommend.lp')

        # solve problem using PuLP's default solver
        #prob.solve()
        prob.solve(pulp.PULP_CBC_CMD(options=['-threads', '4']))

        # print the status of the solution
        print('status:', pulp.LpStatus[prob.status])

        # print each variable with it's resolved optimum value
        #for v in prob.variables():
        #   print(v.name, '=', v.varValue)
        #   if v.varValue != 0:
        #      print(v.name, '=', v.varValue)

        visitMat = np.zeros((len(pois), len(pois)), dtype=np.bool)
        for pi in pois:
            for pj in pois:
                visitMat[int(pi), int(pj)] = visit_vars[pi][pj].varValue
        #        if visitMat[int(pi), int(pj)]:
        #            print(pi, pj)

        # print the optimised objective function value
        print('obj:', pulp.value(prob.objective))

        # build the recommended trajectory
        tour = [p0]
        while True:
            pi = tour[-1]
            for pj in pois:
                if visitMat[int(pi), int(pj)]:
                    pi = pj
                    tour.append(pi)
                    if pi == pN:
                        return tour
                    else:
                        break


    def recommend(self):
        """Trajectory Recommendation"""

        for seq in range(len(self.sequences)):
            if len(self.sequences[seq]) < 3:
                self.recommendSeqs[seq] = []
                continue

            tour = self.MIP_recommend(seq, 0.5);
            self.recommendSeqs[seq] = [int(x) for x in tour]

            print('REAL:', self.sequences[seq])
            print('RECO:', self.recommendSeqs[seq])
            print('-'*30)

