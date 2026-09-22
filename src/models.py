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
           

class SAGE(torch.nn.Module):
    """Mean-aggregation student that uses the propagation matrix exactly as given (no self-loops, no renormalisation):
    x <- W_root x + W_nbr (P x), with P[target, source] = edge_attr for edge_index = [source; target]."""
    def __init__(self, nin, nhid, nout, nlayers, dropout=0.5):
        super().__init__()
        dims = [nin] + [nhid] * (nlayers - 1) + [nout]
        self.root = torch.nn.ModuleList([torch.nn.Linear(dims[i], dims[i + 1]) for i in range(nlayers)])
        self.nbr = torch.nn.ModuleList([torch.nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(nlayers)])
        self.dropout = dropout

    def forward(self, data):
        x, n = data.x, data.x.shape[0]
        P = torch.sparse_coo_tensor(data.edge_index.flip(0), data.edge_attr.to(x.dtype), (n, n)).coalesce()
        for i in range(len(self.root)):
            x = self.root[i](x) + self.nbr[i](torch.sparse.mm(P, x))
            if i < len(self.root) - 1:
                x = F.dropout(F.relu(x), self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


class PropGNN(torch.nn.Module):
    """Faithful student of the structure bound: bias-free layers x <- sigma(P x W) with the propagation matrix
    P[target, source] = edge_attr taken exactly as given (no self-loops, no renormalisation, no root term);
    the last layer is linear (logits)."""
    def __init__(self, nin, nhid, nout, nlayers, dropout=0.5):
        super().__init__()
        dims = [nin] + [nhid] * (nlayers - 1) + [nout]
        self.lins = torch.nn.ModuleList([torch.nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(nlayers)])
        self.dropout = dropout

    def forward(self, data):
        x, n = data.x, data.x.shape[0]
        P = torch.sparse_coo_tensor(data.edge_index.flip(0), data.edge_attr.to(x.dtype), (n, n)).coalesce()
        for i, lin in enumerate(self.lins):
            x = torch.sparse.mm(P, lin(x))
            if i < len(self.lins) - 1:
                x = F.dropout(F.relu(x), self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)
