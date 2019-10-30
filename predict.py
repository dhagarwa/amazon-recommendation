import networkx as nx
import sys
sys.path.append('./lib')

from hrg import Dendrogram

D=nx.read_gml('./temp-hrg.gml')

removed_edges=[(0,10),(2,6),(3,4),(5,49),(7,9),(8,60),(9,48),(11,13),(13,31)]

# iterate through all removed edges
for (node1,node2) in removed_edges:
    # find ancestors of node1 and node2
    ancestors1=[node1]
    ancestors2=[node2]
    for i in range(100):
        for n,d in D.nodes_iter(data=True):
            # check if n is an internal node
            if type(n)==unicode :
                # check if n has already been added to ancestors1
                if n not in ancestors1:
                    # check if n is a parent of any node in ancestors1
                    if d['left'] in ancestors1 or  d['right'] in ancestors1:
                        ancestors1.append(n)
                # check if n has already been added to ancestors2
                if n not in ancestors2:
                    # check is n is a parent of any node in ancestors2
                    if d['left'] in ancestors2 or  d['right'] in ancestors2:
                        ancestors2.append(n)
    # find common ancestor
    common_ancestor=None

    for ancestor in ancestors1:
        if ancestor in ancestors2:
            common_ancestor=ancestor
            break

    print('edge=(%d, %d), probability=%f'% (node1,node2,D.node[common_ancestor]['p']))
