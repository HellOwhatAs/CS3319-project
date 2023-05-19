import json, dgl, torch, os

os.environ["DGLBACKEND"] = "pytorch"

if not os.path.exists('./tmp.bin'):
    with open('./archive/author_info_covid.json', 'r') as f:
        author_info_covid = {int(k): {k1: (v1 if k1 == "ct" else [[int(i0) - 1, i1] for i0, i1 in v1]) for k1, v1 in v.items()} for k, v in json.load(f).items()}
    with open('./archive/authors_to_pred.csv', 'r') as f:
        authors_to_pred = torch.tensor([int(i) for i in f.read().strip().split('\n')[1:]], dtype=torch.int64)
    with open('./archive/coauthor_covid.txt', 'r') as f:
        coauthor_covid = torch.tensor([[int(j) for j in i.split()] for i in f.read().strip().split('\n')], dtype=torch.int64)
    torch.save((author_info_covid, authors_to_pred, coauthor_covid), "tmp.bin")
else:
    author_info_covid, authors_to_pred, coauthor_covid = torch.load("tmp.bin")

g = dgl.graph((coauthor_covid[:, 0], coauthor_covid[:, 1]), num_nodes = len(author_info_covid))
print(g)