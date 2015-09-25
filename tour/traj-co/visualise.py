#!/usr/bin/env python3
import sys
from fastkml import kml, styles
from shapely.geometry import Point, LineString


def load_data(ftable1, ftable2):
    """Load data"""
    trajdict1 = dict()
    trajdict2 = dict()

    # Trajectory_ID,User_ID,#Photo,Start_Time,Travel_Distance,Total_Time,Average_Speed
    firstline = True
    with open(ftable2, 'r') as f2:
        for line in f2:
            if firstline: 
                firstline = False
                continue
            t = line.strip().split(',')
            assert(len(t) == 7)
            speed = float(t[6])
            if speed < 1e-6: continue # do not visualise trajectories at a point
            tid = int(t[0])
            uid = t[1]
            stime = t[3]
            dist = t[4]
            ttime = t[5]
            trajdict2[tid] = (uid, stime, dist, ttime, speed)

    #Trajectory_ID,Photo_ID,User_ID,Timestamp,Longitude,Latitude,Accuracy,Marker,URL
    firstline = True
    with open(ftable1, 'r') as f1:
        for line in f1:
            if firstline: 
                firstline = False
                continue
            t = line.strip().split(',')
            assert(len(t) == 9)
            tid = int(t[0])
            if tid not in trajdict2: continue
            if tid not in trajdict1: trajdict1[tid] = []
            pid = t[1]
            uid = t[2]
            time = t[3]
            lng = float(t[4])
            lat = float(t[5])
            acc = int(t[6])
            marker = int(t[7])
            url = t[8]
            trajdict1[tid].append((tid, pid, uid, time, lng, lat, acc, marker, url))
    return trajdict1, trajdict2


def gen_kml(trajdict1, trajdict2, keylist, fname):
    """Generate KML file"""
    k = kml.KML()
    ns = '{http://www.opengis.net/kml/2.2}'
    stid = 'style1'
    # colors in KML: aabbggrr, aa=00 is fully transparent
    # developers.google.com/kml/documentation/kmlreference?hl=en#colorstyle
    st = styles.Style(id=stid, styles=[styles.LineStyle(color='2f0000ff', width=3)]) # transparent red
    d = kml.Document(ns, '001', 'Trajectories', 'Trajectory visualization', styles=[st])
    k.append(d)

    pm_photo = []
    pm_traj = []
    for tid in keylist:
        assert(tid in trajdict1)
        assert(tid in trajdict2)
        pdata = trajdict1[tid]
        trajcoords = [(p[4], p[5]) for p in pdata]
        name = 'Trajectory_' + str(tid)
        desc = 'User_ID: '              + str(trajdict2[tid][0]) + \
               '<br/>Start_Time: '      + str(trajdict2[tid][1]) + \
               '<br/>Travel_Distance: ' + str(trajdict2[tid][2]) + ' km' + \
               '<br/>Total_Time: '      + str(trajdict2[tid][3]) + \
               '<br/>Average_Speed: '   + str(trajdict2[tid][4]) + ' km/h' + \
               '<br/>#Photos: ' + str(len(pdata)) + '<br/>Photos: ' + str([p[1] for p in pdata])
        pm = kml.Placemark(ns, str(tid), name, desc, styleUrl='#' + stid)
        pm.geometry = LineString(trajcoords)
        pm_traj.append(pm)

        for p in pdata:
            name = 'Photo_' + p[1]
            desc = 'Trajectory_ID: ' + str(p[0]) + \
                   '<br/>Photo_ID: ' + p[1] + \
                   '<br/>User_ID: ' + p[2] + \
                   '<br/>Timestamp: ' + p[3] + \
                   '<br/>Coordinates: (' + str(p[4]) + ', ' + str(p[5]) + ')' + \
                   '<br/>Accuracy: ' + str(p[6]) + \
                   '<br/>URL: ' + str(p[8])
            pm = kml.Placemark(ns, p[1], name, desc)
            pm.geometry = Point(p[4], p[5])
            pm_photo.append(pm)

    for pm in pm_traj:  d.append(pm)
    for pm in pm_photo: d.append(pm)

    kmlstr = k.to_string(prettyprint=True)
    with open(fname, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(kmlstr)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'TABLE1_FILE  TABLE2_FILE')
        print('For example: ', sys.argv[0], 'Melb-table1.csv  Melb-table2.csv')
        sys.exit(0)

    ftable1 = sys.argv[1]
    ftable2 = sys.argv[2]
    N = 50

    # for test
    #trajdict = dict() 
    #for i in range(35): trajdict[i+1] = i

    trajdict1, trajdict2 = load_data(ftable1, ftable2)
    inc = int(len(trajdict1.keys()) / N)
    keys = []
    keys.append([])
    idx = 0
    end = idx + inc
    for k in sorted(trajdict1.keys()):
        idx += 1
        if idx > end: 
            end += inc
            if end > len(trajdict1.keys()): end = len(trajdict1.keys())
            keys.append([k])
        else: keys[-1].append(k)

    #print(trajdict)
    #print(keys)

    kmlfiles = ['Melb-traj-part' + str(i+1) + '.kml' for i in range(N)]
    for i in range(N):
        gen_kml(trajdict1, trajdict2, keys[i], kmlfiles[i])

