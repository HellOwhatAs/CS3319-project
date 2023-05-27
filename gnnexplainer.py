import dgl.function as fn
import torch, dgl
import torch.nn as nn
from dgl.data import CoraGraphDataset
from dgl.nn.pytorch import GNNExplainer
from model_def import GNN as Model
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = CoraGraphDataset()
    g = data[0].to(device)
    # from draft import g
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']

    model = Model(features.shape[1], 128, data.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    with tqdm(range(100)) as tbar:
        for epoch in tbar:
            logits = model(g, features)
            loss = criterion(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tbar.set_postfix(loss = loss.item())
            tbar.update()

    with torch.no_grad():
        output = model(g, features)
    
    acc = (labels[~train_mask] == torch.argmax(output[~train_mask], 1)).sum() / labels.shape[0]
    print(acc.item())
    
    explainer = GNNExplainer(model, num_hops=1)
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(10, g, features)