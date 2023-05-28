import torch, os
import networkx as nx
import dgl
from draft import coauthor_covid, g, old2new
from sklearn.cluster import KMeans

def get_features(g: dgl.DGLGraph):
    G = nx.Graph()
    G.add_nodes_from(range(g.num_nodes()))
    G.add_weighted_edges_from([[old2new[s], old2new[e], w] for s, e, w in coauthor_covid.cpu().numpy()])
    G0 = nx.Graph()
    G0.add_edges_from([[old2new[s], old2new[e]] for s, e, w in coauthor_covid.cpu().numpy()])

    DEG = nx.degree_centrality(G)
    BC = nx.betweenness_centrality(G0, 1000)
    EVC = nx.eigenvector_centrality(G, max_iter = 1000, weight='weight')

    feature_dicts = [DEG, BC, EVC]

    res = torch.zeros((g.num_nodes(), len(feature_dicts)))
    for feat_id, feat_dict in enumerate(feature_dicts):
        for k, v in feat_dict.items():
            res[k, feat_id] = v

        res[:, feat_id] -= res[:, feat_id].min()
        res[:, feat_id] /= res[:, feat_id].max()
    
    return res

if not os.path.exists('./cluster_label.bin'):
    pos = get_features(g)
    bc_idx = torch.argmax(pos[:, 1]).item()
    evc_idx = torch.argmax(pos[:, 2]).item()
    classes = KMeans(n_clusters=3, init=[pos[bc_idx].numpy(), pos[evc_idx].numpy(), [0, 0, 0]], n_init=1).fit_predict(
        pos,
        sample_weight = g.ndata['feat'][:, 1:].sum(1).cpu().numpy()
    )
    bc_class, evc_class = classes[bc_idx], classes[evc_idx]
    classes = torch.from_numpy(classes)

    torch.save((classes, pos, {'bc_class': bc_class, 'evc_class': evc_class}), './cluster_label.bin')
else:
    classes, pos, class_name = torch.load('./cluster_label.bin')
    bc_class, evc_class = class_name['bc_class'], class_name['evc_class']

if __name__ == '__main__':
    import plotly.express as px
    import pandas as pd

    non_zero = g.ndata['feat'][:, 1:].sum(1) > 3
    df = pd.DataFrame(pos[non_zero], columns=['DEG', 'BC', 'EVC'])
    df['classes'] = classes[non_zero]

    from collections import Counter
    print(Counter(classes.numpy()))

    fig = px.scatter_3d(df, x='DEG', y='BC', z='EVC', color='classes')
    with open('cluster_result.html', 'w', encoding='utf-8') as f:
        f.write(fig.to_html())