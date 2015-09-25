#!/usr/bin/env python3
import sys
import math
from datetime import datetime


def load_data(fname):
    """Load data records"""
    # * Photo/video identifier 
    # * User NSID
    # * Date taken
    # * Longitude
    # * Latitude
    # * Accuracy
    # * Photo/video page URL
    # * Photos/video marker (0 = photo, 1 = video)

    data = []
    with open(fname, 'r') as f:
        for line in f:
            t = line.strip().split(',')
            pid = t[0].strip()
            uid = t[1].strip()
            time = datetime.strptime(t[2].strip(), '%Y-%m-%d %H:%M:%S.%f')  # 2011-05-09 19:19:58.0
            lng = float(t[3].strip())
            lat = float(t[4].strip())
            acc = int(t[5].strip())
            url = t[6].strip()
            marker = int(t[7].strip())
            data.append((pid, uid, time, lng, lat, acc, url, marker))
    return data


def gen_trajectories(data):
    """Generate trajectories"""
    udict = dict()
    
    # group photos by user ID
    for i in range(len(data)):
        uid = data[i][1]
        if uid not in udict: udict[uid] = []
        udict[uid].append(i)

    # construct travel history (i.e. sort photos by time) for each user
    for uid, dlist in udict.items():
        dlist.sort(key=lambda idx: data[idx][2])

    # construct trajectories by splitting user's travel history
    TGAP = 8 * 60 * 60  # 8 hours
    trajs = []
    for uid in sorted(udict.keys()): # sort by user ID
        dlist = udict[uid]
        if len(dlist) < 1: continue
        if len(dlist) == 1: 
            trajs.append(dlist)
            continue
        trajs.append([dlist[0]])
        for j in range(1, len(dlist)):
            p1 = dlist[j-1]
            p2 = dlist[j]
            t1 = data[p1][2]
            t2 = data[p2][2]
            assert(t1 <= t2)
            if (t2 - t1).seconds < TGAP:
                trajs[-1].append(p2)
            else:
                trajs.append([p2])
    return trajs


def filter_trajectories(lng_min, lat_min, lng_max, lat_max, trajlist, data):
    """Drop Trajectories which are completely out of the bounding box:
       [(lng_min, lat_min), (lng_max, lat_max)]
    """
    assert(lng_min < lng_max)
    assert(lat_min < lat_max)

    indexes = []
    for i in range(len(trajlist)):
        traj = trajlist[i]
        anyin = False
        for p in traj:
            assert(p in range(len(data)))
            lng = data[p][3]
            lat = data[p][4]
            if lng_min < lng < lng_max and lat_min < lat < lat_max:
                anyin = True
                break
        if anyin: 
            indexes.append(i)
    return [trajlist[x] for x in indexes]


def calc_dist(longitude1, latitude1, longitude2, latitude2):
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


def dump_trajectories(fout1, fout2, trajlist, data):
    """Save Trajectories"""
    # data table 1
    with open(fout1, 'w') as f1:
        f1.write('Trajectory_ID, Photo_ID, User_ID, Timestamp, Longitude, Latitude, Accuracy, Marker(photo=0, video=1), URL\n')
        for i, dlist in enumerate(trajlist):
            assert(len(dlist) > 0)
            tid = str(i)
            for p in dlist:
                assert(p in range(len(data)))
                pid = str(data[p][0])
                uid = str(data[p][1])
                time = str(data[p][2])
                lng = str(data[p][3])
                lat = str(data[p][4])
                acc = str(data[p][5])
                url = str(data[p][6])
                marker = str(data[p][7])
                f1.write(tid + ',' + pid + ',' + uid + ',' + \
                         time + ',' + lng + ',' + lat + ',' + \
                         acc + ',' + marker + ',' + url + '\n')

    # data table 2
    with open(fout2, 'w') as f2:
        f2.write('Trajectory_ID, User_ID, #Photo, Start_Time, Travel_Distance(km), Total_Time(HH:MM:SS), Average_Speed(km/h)\n')
        for i, dlist in enumerate(trajlist):
            assert(len(dlist) > 0)
            tid = str(i)
            p1 = dlist[0]
            p2 = dlist[-1]
            uid = str(data[p1][1])
            num = str(len(dlist))
            dist = calc_dist(data[p1][3], data[p1][4], data[p2][3], data[p2][4])  # km
            t1 = data[p1][2]
            t2 = data[p2][2]
            assert(t1 <= t2)
            tdelta = (t2 - t1).seconds
            hh = int(tdelta / 3600)
            mm = int(tdelta / 60) - hh * 60
            ss = tdelta % 60
            if hh < 10: hh = '0' + str(hh)
            if mm < 10: mm = '0' + str(mm)
            if ss < 10: ss = '0' + str(ss)
            ttime = str(hh) + ':' + str(mm) + ':' + str(ss)
            speed = None
            if tdelta == 0: speed = 0
            else:           speed = dist * 60 * 60 / tdelta  # km/h
            f2.write(tid + ',' + uid + ',' + num + ',' + str(t1) + ',' + \
                    str(dist) + ',' + ttime + ',' + str(speed) + '\n')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:', sys.argv[0], 'BIGBOX_DATA_FILE')
        sys.exit(0)
    
    fin = sys.argv[1]
    fout1 = './Melb-table1.csv'
    fout2 = './Melb-table2.csv'
    lng_min = 144.597363
    lat_min = -38.072257
    lng_max = 145.360413
    lat_max = -37.591764

    data = load_data(fin)
    trajs = gen_trajectories(data)
    trajs = filter_trajectories(lng_min, lat_min, lng_max, lat_max, trajs, data)
    dump_trajectories(fout1, fout2, trajs, data)

