import torch
import networkx as nx
import dgl
from draft import coauthor_covid, g

def get_features(g: dgl.DGLGraph):
    G = nx.Graph()
    G.add_nodes_from(range(g.num_nodes()))
    G.add_weighted_edges_from(coauthor_covid.tolist())
    G0 = nx.Graph()
    G0.add_edges_from(coauthor_covid[:, :2].tolist())

    DEG = nx.degree_centrality(G)
    BC = nx.betweenness_centrality(G0, 10)
    EVC = nx.eigenvector_centrality(G, weight='weight')

    feature_dicts = [DEG, BC, EVC]

    res = torch.zeros((g.num_nodes(), len(feature_dicts)))
    for feat_id, feat_dict in enumerate(feature_dicts):
        for k, v in feat_dict.items():
            res[k, feat_id] = v

        res[:, feat_id] -= res[:, feat_id].min()
        res[:, feat_id] /= res[:, feat_id].max()
    
    return res

if __name__ == '__main__':
    pos = get_features(g)

    from sklearn.cluster import KMeans
    bc_idx = torch.argmax(pos[:, 1]).item()
    evc_idx = torch.argmax(pos[:, 2]).item()
    classes = KMeans(n_clusters=3, init=[pos[bc_idx].numpy(), pos[evc_idx].numpy(), [0, 0, 0]], n_init=1).fit_predict(pos)

    import plotly.express as px
    import pandas as pd

    non_zero = pos.square().sum(1) > 0
    df = pd.DataFrame(pos[non_zero], columns=['DEG', 'BC', 'EVC'])
    df['classes'] = classes[non_zero]

    fig = px.scatter_3d(df, x='DEG', y='BC', z='EVC', color='classes')
    with open('cluster_result.html', 'w', encoding='utf-8') as f:
        f.write(fig.to_html())