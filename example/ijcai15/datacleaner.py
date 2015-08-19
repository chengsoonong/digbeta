import sys
import math
import numpy as np

class DataCleaner:
    """Clean POI list and photo records"""

    def __init__(self, fpoi, fphoto):
        self.pois = []
        self.records = []
        self.load_pois(fpoi)
        self.load_records(fphoto)


    def load_pois(self, fname):
        """Load POI list"""
        with open(fname, 'r') as f:
            for line in f:
                x, y, t = line.split(',')
                self.pois.append((float(x.strip()), float(y.strip()), t.strip()))


    def load_records(self, fname):
        """Load selected (from DB) photo records"""
        with open(fname, 'r') as f:
            for line in f:
                #photoId, userId, date, longitude, latitude
                item = line.split(',')
                assert(len(item) == 5)
                photoId   = item[0].strip()
                userId    = item[1].strip()
                dateTaken = int(float(item[2].strip()))
                longitude = float(item[3].strip())
                latitude  = float(item[4].strip())
                self.records.append((photoId, userId, dateTaken, longitude, latitude))


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


    def assign_photo(self, poiIdx, recordIdx):
        """Assign photo the nearest POI"""
        assert(len(self.pois) > 0)
        assert(len(self.records) > 0)

        #poi_photo_maxdist = 0.2 # unit: km, 200m as in paper
        poi_photo_maxdist = 1
        assignments = []

        for idx in recordIdx:
            lng = self.records[idx][3]
            lat = self.records[idx][4]
            poi = -1
            mindist = poi_photo_maxdist + 1
            for p in poiIdx:
                dist = self.calc_dist(lng, lat, self.pois[p][0], self.pois[p][1])
                if poi == -1 or dist < mindist:
                    mindist = dist
                    poi = p
            if mindist < poi_photo_maxdist:
                assignments.append((idx, poi))
        return assignments


    def assign_and_filter(self):
        """Discard POI that is very close to its neighbor which has more photos assigned"""
        dmin = 2 # km, minimum distance between two POIs
        umin = 2 # minimum number of users associated with a POI
        poiIdx = list(range(len(self.pois)))
        recordIdx = list(range(len(self.records)))
        assigndict = dict()
        assignments = self.assign_photo(poiIdx, recordIdx)
        for term in assignments:
            idx = term[0]
            poi = term[1]
            if poi not in assigndict: assigndict[poi] = set()
            assigndict[poi].add(idx)

        #poiusers = dict()
        #for poi, rset in assigndict.items():
        #    if poi not in poiusers: poiusers[poi] = set()
        #    for idx in rset:
        #        userId = self.records[idx][1]
        #        poiusers[poi].add(userId)
   
        indicator = np.zeros(len(self.pois), dtype=np.bool)
        for poi in assigndict.keys(): indicator[poi] = True
        minidx = np.ones(len(self.pois), dtype=np.int32)*(-1)

        stack = [] # contain index of POI whose indicator is False but its nearest neighbors unnotified
                   # and the assigned photo should be re-assigned to other POIs
        for p1 in range(len(self.pois)):
            if indicator[p1] == False: continue
            mindist = dmin + 1
            for p2 in range(len(self.pois)):
                if indicator[p2] == False or p1 == p2: continue
                dist = self.calc_dist(self.pois[p1][0], self.pois[p1][1], self.pois[p2][0], self.pois[p2][1])
                if dist < mindist:
                    mindist = dist
                    minidx[p1] = p2
            if mindist < dmin:
                px = p1
                if len(assigndict[p1]) > len(assigndict[minidx[p1]]): px = minidx[p1]
                indicator[px] = False
                stack.append(px)

        #for p1, uset in poiusers.items():
        #    if len(uset) < umin:
        #        indicator[p1] = False
        #        stack.append(p1)
        
        while len(stack) > 0:
            print('stack size:', len(stack))
            p = stack.pop()

            # re-assign photos
            recordIdx = list(assigndict[p])
            del assigndict[p]
            #del poiusers[p]
            poiIdx = [x for x in range(len(self.pois)) if indicator[x]]
            assignments1 = self.assign_photo(poiIdx, recordIdx)
            for term in assignments1:
                idx = term[0]
                poi = term[1]
                #userId = self.records[idx][1]
                if poi not in assigndict: assigndict[poi] = set()
                #if poi not in poiusers: poiusers[poi] = set()
                assigndict[poi].add(idx)
                #poiusers[poi].add(userId)
            #for p1, uset in poiusers.items():
            #    if len(uset) < umin:
            #        indicator[p1] = False
            #        stack.append(p1)
           
            # re-calculate the nearest neighbor for affected POIs
            for p1 in range(len(self.pois)):
                if indicator[p1] and minidx[p1] == p: 
                    mindist = dmin + 1
                    for p2 in range(len(self.pois)):
                        if p1 == p2 or indicator[p2] == False: continue
                        dist = self.calc_dist(self.pois[p1][0], self.pois[p1][1], self.pois[p2][0], self.pois[p2][1])
                        if dist < mindist:
                            mindist = dist
                            minidx[p1] = p2
                    if mindist < dmin:
                        px = p1
                        if len(assigndict[p1]) > len(assigndict[minidx[p1]]): px = minidx[p1]
                        indicator[px] = False
 
        poiusers = dict()
        for poi, rset in assigndict.items():
            if poi not in poiusers: poiusers[poi] = set()
            for idx in rset:
                userId = self.records[idx][1]
                poiusers[poi].add(userId)
        plist = []
        for poi, uset in poiusers.items():
            if len(uset) < umin: plist.append(poi)
        for poi in plist:
            del assigndict[poi]

        return assigndict
        
        #poiIndicator = indicator
        #poiIdx = [x for x in range(len(self.pois)) if indicator[x]]
        #poicnt = 0
        #for poi in range(len(self.pois)):
        #    if indicator[poi]: poicnt += 1
        #print(len(poiIdx), '==', poicnt)

        #recordIndicator = np.zeros(len(self.records), dtype=np.bool)
        #for k in assigndict.keys():
        #    for ri in assigndict[k]:
        #        recordIndicator[ri] = True

        #recordcnt = 0
        #for i in range(len(self.records)):
        #    if recordIndicator[i]: recordcnt += 1
        #recordIdx1 = []
        #recordIdx2 = set()
        #for k, v in assigndict.items():
        #    recordIdx1.extend(list(v))
        #    recordIdx2 = recordIdx2 | v
        #print(len(recordIdx1), '==', len(recordIdx2), '==', recordcnt)

        #return poiIndicator, recordIndicator


    def build_sequence(self, poiIndicator, recordIndicator):
        """Build visiting sequences"""
        # assign photo to POI
        poiIdx = [x for x in range(len(self.pois)) if poiIndicator[x]]
        recordIdx = [x for x in range(len(self.records)) if recordIndicator[x]]
        assignments = self.assign_photo(poiIdx, recordIdx)

        # build visiting history for each user
        histories = dict()
        for i in range(len(self.records)):
            userId = self.records[i][1]
            if userId not in histories: 
                histories[userId] = []
            histories[userId].append(i)

        print('#users:', len(histories))

        # sort according to time
        for k in histories.keys():
            histories[k].sort(key=lambda idx:self.records[idx][2])

        # build visiting sequences
        timedelta = 8 * 60 * 60 # 8 hours according to paper
        sequences = dict()
        seq = 0
        for k, v in histories.items():
            if seq not in sequences: sequences[seq] = []
            assert(len(v) > 0)
            sequences[seq].append(v[0])
            for j in range(1, len(v)):
                delta = self.records[v[j]][2] - self.records[v[j-1]][2]
                if delta >= timedelta:
                    seq += 1
                    if seq not in sequences: sequences[seq] = []
                sequences[seq].append(v[j])
            seq += 1 # for sequences of next user

        # write sequences to file
        #with open('seq.list.x', 'w') as f:
        #    for seqId, v in sequences.items():
        #        assert(len(v) > 0)
        #        userId = self.records[v[0]][1]
        #        poilist = []
        #        for idx in v: 
        #            assert(self.records[idx][1] == userId)
        #            assert(assignments[idx][0] == idx)
        #            poiId = assignments[idx][1]
        #            if len(poilist) == 0 or poilist[-1] != poiId: 
        #                poilist.append(poiId)
        #        if len(poilist) > 1: print(poilist)
        #        #poilist = [records[idx][5] for idx in v]
        #        #userId, seqId, poi1, poi2, ... poiN
        #        line = str(seqId) + ',' + str(userId)
        #        for poi in poilist:
        #            line += ','
        #            line += str(poi)
        #        f.write(line + '\n')
        return sequences


    def main(self):
        """Main procedure"""
        assigndict = self.assign_and_filter()

        # build visiting sequences
        poiInd = np.zeros(len(self.pois), dtype=np.bool)
        for poi in assigndict.keys(): poiInd[poi] = True
        recordInd = np.zeros(len(self.records), dtype=np.bool)
        for k in assigndict.keys():
            for ri in assigndict[k]: recordInd[ri] = True
        sequences = self.build_sequence(poiInd, recordInd)

        # save POI info
        # poiId,longitude,latitude,category
        with open('poi.coord', 'w') as f:
            for poiId in assigndict.keys():
                lng = self.pois[poiId][0]
                lat = self.pois[poiId][1]
                cat = self.pois[poiId][2]
                line = str(poiId) + ',' + str(lng) + ',' + str(lat) + ',' + str(cat)
                f.write(line + '\n')

        # save user visiting records
        poifreq = dict()
        poiIds = np.ones(len(self.records), dtype=np.int32) * (-1)
        seqIds = np.ones(len(self.records), dtype=np.int32) * (-1)
        for poi in assigndict.keys():
            if poi not in poifreq: poifreq[poi] = len(assigndict[poi])
            for i in assigndict[poi]: poiIds[i] = poi
        for seq in sequences.keys():
            for i in sequences[seq]: seqIds[i] = seq
        #(photoID, userID, dateTaken, poiID, poiTheme, poiFreq, seqID)
        with open('userVisits-Melb.csv', 'w') as f:
            for i in range(len(self.records)):
                if recordInd[i]:
                    item = self.records[i]
                    photoId   = str(item[0])
                    userId    = "'" + str(item[1]) + "'"
                    dateTaken = str(item[2])
                    poiId     = poiIds[i]
                    poiTheme  = "'" + str(self.pois[poiId][2]) + "'"
                    poiFreq   = str(poifreq[poiId])
                    seqId     = str(seqIds[i])
                    line = photoId + ';' + userId + ';' + dateTaken + ';' + str(poiId) + ';' + poiTheme + ';' + poiFreq + ';' + seqId
                    f.write(line + '\n')

        # save photo coordinates
        with open('userVisits-Melb.csv.coord', 'w') as f:
            for i in range(len(self.records)):
                 # (poiID, photoID, longitude, latitude)
                if recordInd[i]:
                    poiId = str(poiIds[i])
                    photoId = str(self.records[i][0])
                    lng = str(self.records[i][3])
                    lat = str(self.records[i][4])
                    line = poiId + ':' + photoId + ':' + lng + ':' + lat
                    f.write(line + '\n')


if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print('Usage: python', sys.argv[0], 'POI_FILE  PHOTO_FILE')
        sys.exit(0)

    fpoi = sys.argv[1]
    fphoto  = sys.argv[2]

    cleaner = DataCleaner(fpoi, fphoto)
    cleaner.main()

