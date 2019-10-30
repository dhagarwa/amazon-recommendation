#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:17:46 2017

@author: dhwanit
"""
import networkx as nx
import inspect
import numpy as np
from scipy.sparse.linalg import norm
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import gmres
from scipy.sparse import identity
from scipy.sparse import diags
import time
import matplotlib.pyplot as plt

Gref = nx.read_edgelist('amazon0601.txt', comments='#', nodetype=int) #Reading the graph, note that G.nodes is not sorted
G = nx.read_edgelist('amazontest2.txt', comments='#', nodetype=int) #Reading the graph, note that G.nodes is not sorted

                     
                     
                     
nodes_list = list(G.nodes)
#print(sorted(list(G.nodes)) == list(G.nodes)) 
#print(G.edges)
#print('Connected:', nx.is_connected(G))
edge_list = list(G.edges)
print('no of edges:', len(edge_list))
M = nx.adjacency_matrix(G)
L = nx.laplacian_matrix(G)
diagonal = [[1.0/val for (node, val) in G.degree()]]
#diagonal = [[1.0/x for x in degree_list ]]
D = diags(diagonal, [0] )
P = D.dot(M)
#print(M)
n = M.shape[0]   #no. of nodes
Mref = nx.adjacency_matrix(Gref)
nref = Mref.shape[0]
print('Norm Adjacency= ', norm(M))
print('Norm Laplacian= ', norm(L))




#def difference(S, R):
#    DIF = nx.create_empty_copy(R)
#    DIF.name = "Difference of (%s and %s)" % (S.name, R.name)
#
#    r_edges = set(R.edges())
#    s_edges = set(S.edges())
#
#    # I'm not sure what the goal is: the difference, or the edges that are in R but not in S
#    # In case it is the difference:
#    #diff_edges = r_edges.symmetric_difference(s_edges)
#
#    # In case its the edges that are in R but not in S:
#    diff_edges = r_edges - s_edges
#
#    DIF.add_edges_from(diff_edges)
#
#    return DIF
#
#diff = difference(Gtest, G)
#diff_edges = list(diff.edges())
#print('Diff:', diff_edges[:100])



def Katz(M, beta, x): #beta*norm(M) < 1, M is adjacency matrix, x is the node index in nodes list
    start = time.time()
    e = np.zeros(n) 
    e[x] = 1.0   #e is canonical basis vector at node position
    score = np.zeros(n)
    prev_corr = e
    for i in range(10):
        curr_corr = beta*(M@prev_corr)
        score += curr_corr
        prev_corr = curr_corr
    end = time.time()  
    print('Time for Katz: ', end - start)
    return score


#Jaccard Similarity score, G is the nx graph object , x is the 'node' not its index in node_list
def Jaccard(G, x):
    start = time.time()
    score_jaccard = np.zeros(nref)-10000000.0 #Jaccard score
    score_cn = np.zeros(nref) -100000000.0    #Common neighbours score
    score_pa = np.zeros(nref)-100000000.0    #Preferential attachment score
    score_adam = np.zeros(nref) -10000000 #Adamic/Adar score
    nbors_x = list(G.neighbors(x))
    for y in G.nodes:
        common_nbors = list(nx.common_neighbors(G, x, y ))
        score_cn[y] = 1.0*len(common_nbors)
        nbors_y = list(G.neighbors(y))
        score_jaccard[y] = (1.0*len(common_nbors))/ len(set(nbors_x) | set(nbors_y))
        score_pa[y] = len(nbors_x)*len(nbors_y)
        for z in common_nbors:
            score_adam[y] += 1.0/ np.log(len(list(G.neighbors(z))))

        
    end = time.time()
    print('Time for local indices:', end - start)
    return score_jaccard, score_cn, score_pa, score_adam
 





def report(xk):
    frame = inspect.currentframe().f_back
    print(frame.f_locals['resid'])


def MatrixForestIndex(L, alpha, x): #alpha*norm(L) < 1, x  is the node index in nodes_list
    start = time.time()
    I = identity(n)
    e = np.zeros(n)
    K = I + alpha*L
    e[x] =1.0
    score = np.zeros(n)
    score = cg(K, e, tol=1e-2)
    end = time.time()
    print('Time for MFI:', end - start)
    return score[0]
    
    

def LocalPathIndex(M, eps, x):
    start = time.time()
    score = np.zeros(n)
    e = np.zeros(n)
    e[x] =1.0
    score = (M@e)
    score = M@score
    score = score + eps*M@score
    end = time.time()
    print('Time for LPI:', end - start)
    return score


def RWR(P, x, c, similar_indices):
    start = time.time()
    score = np.zeros(n)
    score = score - 10000.0
    e = np.zeros(n)
    e[x] = 1.0
    I = identity(n)
    K = (I - c*P.T)
    qx = cg(K, e, tol=1e-2)
    qx = qx[0]
    for y in similar_indices:
        #print('y=', y)
        e = np.zeros(n)
        e[y] = 1.0
        #print(K.shape)
        #print(e.shape)
        qy = gmres(K, e, tol=1e-2)
        qy = (1.0-c)*qy[0]
        score[y] = qx[y] + qy[x]
        
    
    end = time.time()
    print('Time for RWR:', end-start)
    return score



def correct_pred(method, topsimilar_nodes):
    ref_set = set(ref_neighbors)
    test_set = set(neighbors)
    pred_set = set(topsimilar_nodes)
    correct_pred_set = (ref_set.difference(test_set)).intersection(pred_set)
    print('Correct predictions precision', method, ': ', len(correct_pred_set)/ (n_to_pred))
    return (len(correct_pred_set)*1.0)/ (n_to_pred )



pred_node_list = np.random.choice(nodes_list, 10)
prec, prec2, prec3, prec4 = 0.0, 0.0, 0.0, 0.0
count = 0
for pred_node in pred_node_list:
    pred_node_index = nodes_list.index(pred_node)
    print('node :', nodes_list[pred_node_index])
    
    ref_neighbors = list(Gref.neighbors(pred_node))
    neighbors = list(G.neighbors(pred_node))
    n_to_pred = len(ref_neighbors) - len(neighbors)
    n_pred = len(neighbors) + n_to_pred 
    print('Nbors to be predicted:', n_to_pred)
    if n_to_pred==0:
        continue
    
#    
#    #print('Node chosen:', nodes_list[6])
#    score = Katz(M, 0.0001, pred_node_index)  #score has the katz score(x, y) for all y in the order in which they appear in G.nodes 
#    #print(score)
#    topsimilar_indices = np.argsort(score)[-n_pred:] # in increasing order
#    topsimilar_nodes = [nodes_list[i] for i in topsimilar_indices]
#    #print('\nKatz: ', topsimilar_nodes)
#    prec += correct_pred('Katz',topsimilar_nodes)
#    count += 1
#    
#    
#    #print(nodes_list[0])
#    score = RWR(P, pred_node_index, 0.8, topsimilar_indices)  #score has the RWR score(x, y) for all y in the order in which they appear in G.nodes 
#    #print(score)
#    topsimilar_indices = np.argsort(score)[-n_pred:] # in increasing order
#    topsimilar_nodes = [nodes_list[i] for i in topsimilar_indices]
#    #print('\nRWR: ', topsimilar_nodes)
#    correct_pred('RWR',topsimilar_nodes)
#    
#    
    score_jac, score_cn, score_pa, score_adam = Jaccard(G, pred_node)
    topsimilar_list_jac = np.argsort(score_jac)[-n_pred:] # in increasing order
    topsimilar_list_cn = np.argsort(score_cn)[-n_pred:]
    topsimilar_list_pa = np.argsort(score_pa)[-n_pred:]
    topsimilar_list_adam = np.argsort(score_adam)[-n_pred:]
    #print('\nJaccard:', topsimilar_list_jac)
    #print('\nCommon neighbours:', topsimilar_list_cn)
    #print('\nPreferential attachment:', topsimilar_list_pa)
    #print('\nAdam/Adar:', topsimilar_list_adam)
    prec +=correct_pred('Jaccard',topsimilar_list_jac)
    prec2 += correct_pred('Common neighbors',topsimilar_list_cn)
    prec3 += correct_pred('Preferential Attachment',topsimilar_list_pa)
    prec4 += correct_pred('Adam/Adar',topsimilar_list_adam)
    count += 1
    
#    
    
#    #print(nodes_list[2])
#    score = MatrixForestIndex(L, 0.2, pred_node_index)  #score has the MFI score(x, y) for all y in the order in which they appear in G.nodes 
#    #print(score)
#    topsimilar_indices = np.argsort(score)[-n_pred:] # in increasing order
#    topsimilar_nodes = [nodes_list[i] for i in topsimilar_indices]
#    #print('\n Matrix Forest Index: ', topsimilar_nodes)
#    prec += correct_pred('MFI',topsimilar_nodes)
#    count += 1
    
#    
#    #print(nodes_list[2])
#    score = LocalPathIndex(M, 0.2, pred_node_index)  #score has the MFI score(x, y) for all y in the order in which they appear in G.nodes 
#    #print(score)
#    topsimilar_indices = np.argsort(score)[-n_pred:] # in increasing order
#    topsimilar_nodes = [nodes_list[i] for i in topsimilar_indices]
#    #print('\n Local Path Index: ', topsimilar_nodes)
#    correct_pred('LPI',topsimilar_nodes)

       


    
print('precision1:', prec/count)
print('precision1:', prec2/count)
print('precision1:', prec3/count)
print('precision1:', prec4/count)
            




 