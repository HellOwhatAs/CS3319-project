import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 2,
                 dropout = 0.5, return_embeds = False):
        super(GNN, self).__init__()
        
        aggregator_type = 'mean'
        self.convs = nn.ModuleList([
            dglnn.SAGEConv(input_dim, hidden_dim, aggregator_type),
            *(dglnn.SAGEConv(hidden_dim, hidden_dim, aggregator_type) for _ in range(num_layers - 2)),
            dglnn.SAGEConv(hidden_dim, output_dim, aggregator_type)
        ])
        
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)])

        self.softmax = nn.LogSoftmax(dim = 1)

        self.dropout = dropout

        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, graph, feat, eweight = None):

        for conv, bn in zip(self.convs, self.bns):
            feat = F.dropout(F.relu(bn(conv(graph, feat, edge_weight = eweight))), p = self.dropout, training = self.training)
        feat = self.convs[-1](graph, feat, edge_weight = eweight)

        return feat if self.return_embeds else self.softmax(feat)