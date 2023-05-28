import dgl.function as fn
import torch, dgl
import torch.nn as nn
from dgl.data import CoraGraphDataset
from dgl.nn.pytorch import GNNExplainer
from model_def import GNN as Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from get_features import classes as cluster_label

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from draft import g, old2new

    g.ndata['label'] = cluster_label.to(device).long()

    features = g.ndata['feat']
    labels = g.ndata['label']
    test_mask = g.ndata['test_mask']
    train_mask = ~test_mask

    model = Model(features.shape[1], 128, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    with tqdm(range(2000)) as tbar:
        for epoch in tbar:
            logits = model(g, features)
            loss = criterion(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = (labels[test_mask] == torch.argmax(logits[test_mask], 1)).sum() / test_mask.sum()
            tbar.set_postfix(loss = loss.item(), acc = acc.item())
            tbar.update()

    explainer = GNNExplainer(model, num_hops=1)
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(old2new[36291], g, features)

    print(new_center, sg, feat_mask, edge_mask, sep='\n')
    print()
    print(sg.ndata['old_id'])