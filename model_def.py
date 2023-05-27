import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch, copy

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
    


def train(model, g, optimizer, loss_fn, max_epoch):
    loss = 0
    best_val_acc = 0
    
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    
    for e in range(max_epoch):
        
        output = model(g, features, edge_weight = g.edata['t'])
        loss = loss_fn(output[train_mask], labels[train_mask])
        val_acc = torch.sum(output[val_mask].max(1)[1] == labels[val_mask]) / torch.sum(val_mask)
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(
                "In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})".format(
                    e, loss, val_acc, best_val_acc
                )
            )
    return best_model, best_val_acc

@torch.no_grad()
def test(model, g):
    model.eval()
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    out = model(g, features)
    
    test_acc = torch.sum(out[test_mask].max(1)[1] == labels[test_mask]) / torch.sum(test_mask)

    return test_acc

if __name__ == '__main__':
    from draft import g, device
    model = GNN(
        input_dim = g.ndata["feat"].shape[1],
        hidden_dim = 128,
        output_dim = 10,                            # TODO: output 什么？
        num_layers = 2,
        dropout = 0.5
    ).to(device)
    model.reset_parameters()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = F.nll_loss

    best_model, best_valid_acc = train(model, g, optimizer, loss_fn, 100)
    
    test_acc = test(best_model, g)
    print(f'Test: {100 * test_acc:.2f}%')