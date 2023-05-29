from mycsv import csv
from draft import g, new2old, old2new, coauthor_covid
from get_features import classes

edge_data = coauthor_covid.cpu().numpy()
cluster_labels = classes.cpu().numpy()

node = csv(g.num_nodes())
edge = csv(g.num_edges() // 2)

edge['source'] = [str(old2new[i]) for i in edge_data[:, 0]]
edge['target'] = [str(old2new[i]) for i in edge_data[:, 1]]
edge['weight'] = [str(i) for i in edge_data[:, 2]]

node['id'] = [str(i) for i in range(g.num_nodes())]
node['label'] = [str(new2old[i]) for i in range(g.num_nodes())]
node['mylabel'] = [str(i) for i in cluster_labels]
node['max_field'] = [str(i.item()) for i in g.ndata['feat'][:, 1:].argmax(1)]

edge.write('edge.csv')
node.write('node.csv')