import numpy
import json, torch, os
import networkx as nx

from draft import authors_to_pred, coauthor_covid, idx2ct, ct2idx, max_field, ct, ca


# print("authors_to_pred:", authors_to_pred[0])
# print("coauthor_covid:", coauthor_covid.shape) # (875873, 3)
# print("ct:", ct.shape) # (1275456)
# print("ca:", ca.shape) # (1275456, 22)

def normal(dic):
    max_value = max(dic.values())
    min_value = min(dic.values())
    normalized_dict = {k: (v - min_value) / (max_value - min_value) for k, v in dic.items()}
    return normalized_dict

def get_features():
    G = nx.Graph()
    G0 = nx.Graph()
    G.add_nodes_from([i for i in range(1275456)])
    # print(G.nodes())
    # print(coauthor_covid[:, :2].tolist()[:10])
    G0.add_edges_from(coauthor_covid[:, :2].tolist())
    G.add_weighted_edges_from(coauthor_covid.tolist())
    print('construct graph ok.')

    # print(nx.is_connected(G))
    # G.remove_nodes_from(nx.isolates(G.copy()))
    # print(nx.is_connected(G))

    k = 10
    DEG = nx.degree_centrality(G0)
    DEG = normal(DEG)
    BC = nx.betweenness_centrality(G0, k) 
    BC = normal(BC)
    EBC = nx.edge_betweenness_centrality(G, k, weight='weight')  # 边的特征
    EBC = normal(EBC)
    EVC = nx.eigenvector_centrality(G, weight='weight')
    EVC = normal(EVC)

    # KATZ = nx.katz_centrality(G, weight='weight') # 这两个很慢，没跑出来
    # KATZ = normal(KATZ)
    # CC = nx.closeness_centrality(G)
    # CC = normal(CC)

    return DEG, BC, EBC, EVC