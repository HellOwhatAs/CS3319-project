import torch
import torch.nn as nn
from dgl.nn.pytorch import GNNExplainer
from model_def import GNN as Model
from tqdm import tqdm
from get_features import classes as cluster_label
from mycsv import csv

def calc_sgmask(g, sg, edge_mask, k2idx):
    node_sg = ['0'] * g.num_nodes()
    for i in sg.ndata['ori_id'].cpu().numpy():
        node_sg[i] = '1'

    edge_sg = ['0'] * (g.num_edges() // 2)
    edge_weight = [0.] * (g.num_edges() // 2)
    for s, t, w in tqdm(zip(*sg.edges(), edge_mask), desc='Edge mask calculate', total=sg.num_edges()):
        s, t = sg.ndata['ori_id'][s].item(), sg.ndata['ori_id'][t].item()
        idx = k2idx[f'{min(s, t)},{max(s, t)}']
        edge_sg[idx] = '1'
        edge_weight[idx] += w.item()

    return node_sg, edge_sg, [str(i) for i in edge_weight]

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

    loss_list, acc_list = [], []

    with tqdm(range(2000), desc='GNN training') as tbar:
        for epoch in tbar:
            logits = model(g, features)
            loss = criterion(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = (labels[test_mask] == torch.argmax(logits[test_mask], 1)).sum() / test_mask.sum()
            loss_list.append(loss.item()), acc_list.append(acc.item())
            tbar.set_postfix(loss = loss.item(), acc = acc.item())
            tbar.update()
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    plt.plot(loss_list)
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('log(loss)')
    plt.tight_layout()
    plt.savefig("./assets/loss_curve.svg", transparent=True)
    plt.cla()

    plt.plot(acc_list)
    plt.yscale('logit')
    plt.xlabel('epoch')
    plt.ylabel('logit(acc)')
    plt.tight_layout()
    plt.savefig("./assets/acc_curve.svg", transparent=True)
    plt.cla()

    node_csv = csv.read('./node.csv')
    edge_csv = csv.read('./edge.csv')
    st = [f'{min(int(s), int(t))},{max(int(s), int(t))}' for s, t in zip(edge_csv['source'], edge_csv['target'])]
    k2idx = {elem: idx for idx, elem in enumerate(st)}

    explainer = GNNExplainer(model, num_hops=2)
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(old2new[36291], g, features) # 36291: 桥梁节点; 528886: 中心节点; 7976: 边缘节点
    node_csv['node_sg_bridge'], edge_csv['edge_sg_bridge'], edge_csv['edge_sg_bridge_mask'] = calc_sgmask(g, sg, edge_mask, k2idx)

    explainer = GNNExplainer(model, num_hops=2)
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(old2new[528886], g, features)
    node_csv['node_sg_center'], edge_csv['edge_sg_center'], edge_csv['edge_sg_center_mask'] = calc_sgmask(g, sg, edge_mask, k2idx)

    explainer = GNNExplainer(model, num_hops=2)
    new_center, sg, feat_mask, edge_mask = explainer.explain_node(old2new[7976], g, features)
    node_csv['node_sg_margin'], edge_csv['edge_sg_margin'], edge_csv['edge_sg_margin_mask'] = calc_sgmask(g, sg, edge_mask, k2idx)

    node_csv.write('node_sg.csv')
    edge_csv.write('edge_sg.csv')