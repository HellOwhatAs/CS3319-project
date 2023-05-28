from tqdm import tqdm
from get_features import classes as cluster_label
from draft import g, old2new
import torch

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# g.ndata['label'] = cluster_label.to(device).long()

# with open('node.csv', 'w') as f:
#     f.write('id,mylabel,num_fields\n')
#     f.writelines(tqdm((','.join(str(int(j)) for j in i) + '\n' for i in zip(
#         g.nodes().cpu().numpy(),
#         g.ndata['label'].cpu().numpy(),
#         (g.ndata['feat'][:, 1:] > 0.5).sum(1).cpu().numpy()
#     )), total=g.num_nodes()))

new_edge = []
with open('./edge.csv', 'r') as f:
    new_edge.append(f.readline().strip().split())
    for l in f.readlines():
        s, t, w = l.strip().split()
        new_edge.append([str(old2new[int(s)]), str(old2new[int(t)]), w])

with open('./new_edge.csv', 'w') as f:
    for l in new_edge:
        f.write(','.join(l))
        f.write('\n')