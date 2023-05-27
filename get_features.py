import numpy
import json, torch, os
import networkx as nx

os.environ["DGLBACKEND"] = "pytorch"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ca2vec(author_info_covid):
    ret = torch.zeros((len(author_info_covid), max_field + 1), dtype=torch.int64)
    for i, elem in sorted(author_info_covid.items()):
        for j, k in elem['ca']:
            ret[i, j] = k
    return ret

if not os.path.exists('./tmp.bin'):
    with open('CS3319-project/archive/author_info_covid.json', 'r') as f:
        author_info_covid = {int(k): {k1: (v1 if k1 == "ct" else [[int(i0) - 1, i1] for i0, i1 in v1]) for k1, v1 in v.items()} for k, v in json.load(f).items()}
    with open('CS3319-project/archive/authors_to_pred.csv', 'r') as f:
        authors_to_pred = torch.tensor([int(i) for i in f.read().strip().split('\n')[1:]], dtype=torch.int64)
    with open('CS3319-project/archive/coauthor_covid.txt', 'r') as f:
        coauthor_covid = torch.tensor([[int(j) for j in i.split()] for i in f.read().strip().split('\n')], dtype=torch.int64)

    idx2ct = sorted({v['ct'] for k, v in author_info_covid.items()})
    ct2idx = {elem: i for i, elem in enumerate(idx2ct)}
    max_field = max(max(i[0] for i in v['ca']) for k, v in author_info_covid.items())
    ct = torch.tensor([ct2idx[v['ct']] for k, v in sorted(author_info_covid.items())], dtype=torch.int64)
    ca = ca2vec(author_info_covid)

    torch.save((authors_to_pred, coauthor_covid, idx2ct, ct2idx, max_field, ct, ca), "tmp.bin")
else:
    authors_to_pred, coauthor_covid, idx2ct, ct2idx, max_field, ct, ca = torch.load("tmp.bin")

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
    G.add_nodes_from([i for i in range(1275456)])
    # print(G.nodes())
    # print(coauthor_covid[:, :2].tolist()[:10])
    G.add_edges_from(coauthor_covid[:, :2].tolist())
    print('construct graph ok.')

    # print(nx.is_connected(G))
    G.remove_nodes_from(nx.isolates(G.copy()))
    # print(nx.is_connected(G))

    k = 10
    BC = nx.betweenness_centrality(G, k) 
    BC = normal(BC)
    EBC = nx.edge_betweenness_centrality(G, k, weight='weight') 
    EBC = normal(EBC)
    EVC = nx.eigenvector_centrality(G, weight='weight')
    EVC = normal(EVC)

    return BC, EBC, EVC