import math
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm


def main():
    """Main Procedure"""
    pois = [(10, 0), (20, 0), (30, 0)]
    photos1 = [(10, 1), (20, 1), (30, 1)]
    photos2 = [(10, 1), (11, 1), (12, 1)]
    photos3 = [(20, 1), (21, 1), (22, 1)]
    photos4 = [(30, 1), (31, 1), (32, 1)]
    photos5 = [(5, 6), (18, 8), (28, 3)]

    assignment = assign(photos1, pois)
    draw(photos1, pois, assignment)

    assignment = assign(photos2, pois)
    draw(photos2, pois, assignment)
    
    assignment = assign(photos3, pois)
    draw(photos3, pois, assignment)

    assignment = assign(photos4, pois)
    draw(photos4, pois, assignment)
    
    assignment = assign(photos5, pois)
    draw(photos5, pois, assignment)

    a = input('Press any key to continue ...')


def assign(photos, pois):
    """Assign photos to POIs with minimum cost"""
    #REF: en.wikipedia.org/wiki/Minimum-cost_flow_problem
    #assert(len(photos) == len(pois))

    dists = np.zeros((len(photos), len(pois)), dtype=np.float64)
    for i, d in enumerate(photos):
        for j, p in enumerate(pois):
            dists[i, j] = round(math.sqrt( (d[0] - p[0])**2 + (d[1] - p[1])**2 ))
    #print(dists)

    G = nx.DiGraph()
 
    # complete bipartite graph: photo -> POI
    # infinity capacity
    for i in range(len(photos)):
        for j in range(len(pois)):
            u = 'd' + str(i)
            v = 'p' + str(j)
            G.add_edge(u, v, weight=dists[i, j])

    # source -> photo
    # capacity = 1
    for i in range(len(photos)):
        u = 's'
        v = 'd' + str(i)
        G.add_edge(u, v, capacity=1, weight=0)

    # POI -> sink
    # infinity capacity
    for j in range(len(pois)):
        u = 'p' + str(j)
        v = 't'
        G.add_edge(u, v, weight=0)

    # demand for source and sink
    G.add_node('s', demand=-len(photos))
    G.add_node('t', demand=len(photos))

    #print(G.nodes())
    #print(G.edges())

    flowDict = nx.min_cost_flow(G)

    assignment = dict()
    for e in G.edges():
        u = e[0]
        v = e[1]
        if u != 's' and v != 't' and flowDict[u][v] > 0:
            #print(e, flowDict[u][v])
            assignment[u] = v
    return assignment


def draw(photos, pois, assignment):
    """visualize the photo-POI assignment"""
    #assert(len(photos) == len(pois) == len(assignment.keys()))
    assert(len(photos) == len(assignment.keys()))
    
    fig = plt.figure()
    plt.axis('equal')
    colors_poi = np.linspace(0, 1, len(pois))
    X_poi = [t[0] for t in pois]
    Y_poi = [t[1] for t in pois]
    #plt.scatter(X_poi, Y_poi, s=50, marker='o', c=colors_poi, cmap=cm.Greens)
    plt.scatter(X_poi, Y_poi, s=50, marker='s', c=cm.Greens(colors_poi), label='POI')

    colors_photo = np.zeros(len(photos), dtype=np.float64)
    for i in range(len(photos)):
        k = 'd' + str(i)
        v = assignment[k] # e.g. 'p1'
        idx = int(v[1])
        colors_photo[i] = colors_poi[idx]
    X_photo = [t[0] for t in photos]
    Y_photo = [t[1] for t in photos]
    plt.scatter(X_photo, Y_photo, s=30, marker='o', c=cm.Greens(colors_photo), label='Photo')
    plt.legend()
    plt.show()

    
if __name__ == '__main__':
    main()

