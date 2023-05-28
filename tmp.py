from tqdm import tqdm
from get_features import classes as cluster_label
from draft import g, authors_to_pred
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

label = g.ndata['label'] = cluster_label.to(device).long()
features = g.ndata['feat']
train_mask = g.ndata['train_mask']

edged_node = set(g.edges()[0].cpu().numpy()).union(g.edges()[1].cpu().numpy())

# with open('node.csv', 'w') as f:
#     f.write('id,mylabel,train_mask\n')
#     f.writelines(tqdm(','.join(str(int(j)) for j in i) + '\n' for i in zip(g.nodes().cpu().numpy(), g.ndata['label'].cpu().numpy(), train_mask.cpu().numpy()) if i[0] in edged_node))

test = set(authors_to_pred.cpu().numpy())

train = {i for i in range(g.num_nodes()) if not i in test}

print(len(test), len(edged_node))