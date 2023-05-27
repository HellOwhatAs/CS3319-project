import dgl.function as fn
import torch, dgl
import torch.nn as nn
from dgl.data import CoraGraphDataset
from dgl.nn.pytorch import GNNExplainer
from model_def import GNN as Model

if __name__ == '__main__':
    data = CoraGraphDataset()
    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']

    model = Model(features.shape[1], 256, data.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(10):
        logits = model(g, features)
        loss = criterion(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    explainer = GNNExplainer(model, num_hops=1)
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(10, g, features)

    print(sg.device)