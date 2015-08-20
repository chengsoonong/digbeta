#!/usr/bin/env python

import sys
from imposm.parser import OSMParser

class NodeFilter:
    def __init__(self, fname):
        self.num = 0
        self.tagdict = dict()
        self.load_taglist(fname)

    def load_taglist(self, fname):
        """Load interesting tags"""
        with open(fname, 'r') as f:
            for line in f:
                k, v = line.split(':')
                k = k.strip()
                v = v.strip()
                self.tagdict[k] = set()
                terms = v.split(',')
                for t in terms:
                    self.tagdict[k].add(t.strip())

    def nodes_callback(self, nodes):
        with open('nodes.txt', 'a') as f:
            for osmid, tags, coord in nodes:
                flag = False
                desc = ''
                for k, v in tags.items():
                    if k in self.tagdict and (self.tagdict[k] == '_ALL_' or v in self.tagdict[k]):
                        flag = True
                        desc = str(k) + '-' + str(v)
                        break
                if flag:
                    self.num += 1
                    lng = coord[0]
                    lat = coord[1]
                    f.write(str(lng) + ',' + str(lat) + ',' + desc + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage:', sys.argv[0], 'LIST_FILE')
        sys.exit(0)

    fname = sys.argv[1]
    nodefilter = NodeFilter(fname)
    parser = OSMParser(nodes_callback=nodefilter.nodes_callback)
    parser.parse('melbourne.osm')
    print(nodefilter.num)

