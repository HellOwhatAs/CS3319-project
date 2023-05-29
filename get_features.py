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

def make_classifier(pos, classes):
    classifier_data = torch.stack([pos[classes == 0, :2].mean(0), pos[classes == 1, :2].mean(0), pos[classes == 2, :2].mean(0)])
    center_class = classifier_data[:, 0].argmax(0).item()
    classifier_data[center_class] *= 0
    bridge_class = classifier_data[:, 1].argmax(0).item()
    margin_class, = set(range(3)) - {center_class, bridge_class}
    return {
        center_class: 'center',
        bridge_class: 'bridge',
        margin_class: 'margin'
    }

if not os.path.exists('./cluster_pos.bin'):
    pos = get_features(g)
    torch.save(pos, './cluster_pos.bin')
else:
    pos = torch.load('./cluster_pos.bin')

pos[:, 0] /= 2
pos[:, 1] *= 2
classes = KMeans(n_clusters=3, n_init='auto').fit_predict(
    pos[:, :2],
    sample_weight = g.ndata['feat'][:, 1:].sum(1).cpu().numpy()
)
pos[:, 0] *= 2
pos[:, 1] /= 2
classes = torch.from_numpy(classes)
bc_idx = torch.argmax(pos[:, 1]).item()
evc_idx = torch.argmax(pos[:, 2]).item()
classifier = make_classifier(pos, classes)

if __name__ == '__main__':
    import plotly.express as px
    import pandas as pd

    non_zero = g.ndata['feat'][:, 1:].sum(1) > 3
    df = pd.DataFrame(pos[non_zero][:, :2], columns=['degree', 'betweenness centrality'])
    df['classes'] = [classifier[i.item()] for i in classes[non_zero]]
    df['weight'] = g.ndata['feat'][non_zero][:, 1:].sum(1).cpu().numpy()
    df['node_id'] = g.ndata['old_id'][non_zero].cpu().numpy()

    from collections import Counter
    print({classifier[k]: v for k, v in Counter(classes.numpy()).items()})

    fig = px.scatter(df, x='degree', y='betweenness centrality', color='classes', size='weight', hover_data=['node_id'], size_max=40)
    fig.write_html('cluster_result.html')