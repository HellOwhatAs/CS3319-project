import torch
import torch.nn as nn
from dgl.nn.pytorch import GNNExplainer
from model_def import GNN as Model
from tqdm import tqdm
from get_features import classes as cluster_label
from mycsv import csv

def calc_sgmask(g, sg, k2idx):
    node_sg = ['0'] * g.num_nodes()
    for i in sg.ndata['ori_id'].cpu().numpy():
        node_sg[i] = '1'

    edge_sg = ['0'] * (g.num_edges() // 2)
    for s, t in zip(*sg.edges()):
        s, t = sg.ndata['ori_id'][s].item(), sg.ndata['ori_id'][t].item()
        k = f'{min(s, t)},{max(s, t)}'
        edge_sg[k2idx[k]] = '1'

    return node_sg, edge_sg

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

    node_csv = csv.read('./node.csv')
    edge_csv = csv.read('./edge.csv')
    st = [f'{min(int(s), int(t))},{max(int(s), int(t))}' for s, t in zip(edge_csv['source'], edge_csv['target'])]
    k2idx = {elem: idx for idx, elem in enumerate(st)}

    explainer = GNNExplainer(model, num_hops=1)
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(old2new[36291], g, features) # 36291: 桥梁节点; 528886: 中心节点; 7976: 边缘节点
    node_csv['node_sg_bridge'], edge_csv['edge_sg_bridge'] = calc_sgmask(g, sg, k2idx)

    explainer = GNNExplainer(model, num_hops=1)
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(old2new[528886], g, features)
    node_csv['node_sg_center'], edge_csv['edge_sg_center'] = calc_sgmask(g, sg, k2idx)

    explainer = GNNExplainer(model, num_hops=1)
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(old2new[7976], g, features)
    node_csv['node_sg_margin'], edge_csv['edge_sg_margin'] = calc_sgmask(g, sg, k2idx)

    node_csv.write('node_sg.csv')
    edge_csv.write('edge_sg.csv')