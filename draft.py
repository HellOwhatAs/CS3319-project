import json, dgl, torch, os

os.environ["DGLBACKEND"] = "pytorch"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ca2vec(author_info_covid):
    ret = torch.zeros((len(author_info_covid), max_field + 1), dtype=torch.int64)
    for i, elem in sorted(author_info_covid.items()):
        for j, k in elem['ca']:
            ret[i, j] = k
    return ret

if not os.path.exists('./tmp.bin'):
    with open('./archive/author_info_covid.json', 'r') as f:
        author_info_covid = {int(k): {k1: (v1 if k1 == "ct" else [[int(i0) - 1, i1] for i0, i1 in v1]) for k1, v1 in v.items()} for k, v in json.load(f).items()}
    with open('./archive/authors_to_pred.csv', 'r') as f:
        authors_to_pred = torch.tensor([int(i) for i in f.read().strip().split('\n')[1:]], dtype=torch.int64)
    with open('./archive/coauthor_covid.txt', 'r') as f:
        coauthor_covid = torch.tensor([[int(j) for j in i.split()] for i in f.read().strip().split('\n')], dtype=torch.int64)

    idx2ct = sorted({v['ct'] for k, v in author_info_covid.items()})
    ct2idx = {elem: i for i, elem in enumerate(idx2ct)}
    max_field = max(max(i[0] for i in v['ca']) for k, v in author_info_covid.items())
    ct = torch.tensor([ct2idx[v['ct']] for k, v in sorted(author_info_covid.items())], dtype=torch.int64)
    ca = ca2vec(author_info_covid)

    torch.save((authors_to_pred, coauthor_covid, idx2ct, ct2idx, max_field, ct, ca), "tmp.bin")
else:
    authors_to_pred, coauthor_covid, idx2ct, ct2idx, max_field, ct, ca = torch.load("tmp.bin")


coauthor_covid, ct, ca = coauthor_covid.to(device), ct.to(device), ca.to(device)
g = dgl.graph((torch.cat((coauthor_covid[:, 0], coauthor_covid[:, 1])), torch.cat((coauthor_covid[:, 1], coauthor_covid[:, 0]))), num_nodes = len(ct), device=device)
g.edata['t'] = torch.cat((coauthor_covid[:, 2], coauthor_covid[:, 2])).float()
g.ndata['feat'] = torch.cat((torch.unsqueeze(ct, dim=1), ca), dim=1).float()
g.ndata['old_id'] = g.nodes()
g = dgl.node_subgraph(g, authors_to_pred.to(device))
new2old = authors_to_pred.numpy()
old2new = {elem: i for i, elem in enumerate(authors_to_pred.numpy())}