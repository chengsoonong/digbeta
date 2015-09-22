import sys
import numpy as np
from fastkml import kml, styles
from shapely.geometry import Point, LineString


class KMLGenerator:

    def __init__(self, poidata, seqlist):
        self.poidata = poidata
        self.seqlist = seqlist


    def gen_placemark(self, poilist, eid, styleurl):
        assert(len(poilist) > 1)
        ns = '{http://www.opengis.net/kml/2.2}'
        poi0 = poilist[0]
        poiN = poilist[-1]
        assert(poi0 in range(len(self.poidata)))
        assert(poiN in range(len(self.poidata)))
        cat0 = self.poidata[poi0][2]
        catN = self.poidata[poiN][2]
        name = cat0 + '->' + catN
        desc = 'Tajectory: from ' + cat0 + ' to ' + catN
        p = kml.Placemark(ns, str(eid), name, desc, styleUrl=styleurl)
        coords = []
        for pid in poilist:
            assert(pid in range(len(self.poidata)))
            coords.append((self.poidata[pid][0], self.poidata[pid][1]))
        p.geometry = LineString(coords)
        return p


    def gen_kml(self, fname):
        seqlist = self.seqlist
        k = kml.KML()
        ns = '{http://www.opengis.net/kml/2.2}'
        stid = 'style1'
        # colors in KML: aabbggrr, aa=00 is fully transparent
        # developers.google.com/kml/documentation/kmlreference?hl=en#colorstyle
        st = styles.Style(id=stid, styles=[styles.LineStyle(color='0f0000ff', width=3)]) # transparent red
        d = kml.Document(ns, '001', 'Trajectory', 'Trajectory visualization', styles=[st])
        k.append(d)

        poiset = set()
        for i in range(len(seqlist)):
            for poi in seqlist[i]:
                poiset.add(poi)
        for j, poi in enumerate(poiset):
            cat = self.poidata[poi][2]
            p = kml.Placemark(ns, str(j), cat, 'POI: ' + cat)
            p.geometry = Point(self.poidata[poi][0], self.poidata[poi][1])
            d.append(p)
        for i in range(len(seqlist)):
            if len(seqlist[i]) < 2: continue
            p = self.gen_placemark(seqlist[i], i, '#' + stid)
            d.append(p)

        kmlstr = k.to_string(prettyprint=True)
        with open(fname, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(kmlstr)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'POI_FILE  SEQUENCE_FILE')
        sys.exit(0)
    fpoi = sys.argv[1]
    fseq = sys.argv[2]

    # load POI data
    poidata = []
    with open(fpoi, 'r') as f:
        for line in f:
            t = line.strip().split(',')
            lng = float(t[0])
            lat = float(t[1])
            cat = t[2]
            poidata.append((lng, lat, cat))

    # load travelling sequences
    seqlist = []
    with open(fseq, 'r') as f:
        for line in f:
            item = line.split(']')[0].split('[')[1] # e.g. [1, 2, 3, 4, 5], 1281.56 Actual\n
            poilist = [int(x.strip()) for x in list(item.split(','))]
            seqlist.append(poilist)

    kmlfile = 'exmaple.kml'
    genkml = KMLGenerator(poidata, seqlist)
    genkml.gen_kml(kmlfile)

