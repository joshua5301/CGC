from src.module import *

class GCN(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayers, dropout=0.5, normalize=True):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GCNConv(nin, nout, normalize=normalize))
        else:
            self.layers.append(GCNConv(nin, nhid, normalize=normalize))
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv(nhid, nhid, normalize=normalize))
            self.layers.append(GCNConv(nhid, nout, normalize=normalize))
        self.dropout = dropout
        self.initialize()

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)
           