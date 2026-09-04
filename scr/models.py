from scr.module import *
from torch_geometric.nn import ChebConv, SAGEConv, APPNP, GATConv
from torch.nn import Linear

class GCN(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayers, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GCNConv(nin, nout))
        else:
            self.layers.append(GCNConv(nin, nhid)) 
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv(nhid, nhid)) 
            self.layers.append(GCNConv(nhid, nout))  
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

class SGC(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayers, cached=False, dropout=0):
        super().__init__()
        self.layers = SGConv(nin, nout, nlayers, cached=cached) 
        self.H_val =None
        self.H_test=None
        self.dropout = dropout
        self.initialize()

    def initialize(self):
        self.layers.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1) 
    
    def MLP(self, H):
        x = self.layers.lin(H)
        return F.log_softmax(x, dim=1)


class GNN(torch.nn.Module):
    USES_WEIGHT = {'gcn', 'cheby', 'appnp'}

    def __init__(self, kind, nin, nhid, nout, nlayers=2, dropout=0.5, heads=8):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        self.layers = torch.nn.ModuleList([])
        self.prop = None
        if kind == 'appnp':
            self.layers.append(Linear(nin, nhid))
            self.layers.append(Linear(nhid, nout))
            self.prop = APPNP(K=10, alpha=0.1)
        elif kind == 'gat':
            self.layers.append(GATConv(nin, max(nhid // heads, 1), heads=heads))
            self.layers.append(GATConv(max(nhid // heads, 1) * heads, nout, heads=1))
        else:
            conv = {'gcn': GCNConv, 'sage': SAGEConv,
                    'cheby': lambda a, b: ChebConv(a, b, K=2)}[kind]
            for i in range(nlayers):
                self.layers.append(conv(nin if i == 0 else nhid,
                                        nout if i == nlayers - 1 else nhid))
        self.initialize()

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.prop is not None:
            self.prop.reset_parameters()

    def _call(self, layer, x, ei, ew):
        if self.kind in self.USES_WEIGHT and not isinstance(layer, Linear):
            return layer(x, ei, ew)
        if self.kind == 'gat':
            return layer(x, ei)
        return layer(x, ei) if not isinstance(layer, Linear) else layer(x)

    def forward(self, data):
        x, ei, ew = data.x, data.edge_index, data.edge_attr
        for layer in self.layers[:-1]:
            x = F.relu(self._call(layer, x, ei, ew))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self._call(self.layers[-1], x, ei, ew)
        if self.prop is not None:
            x = self.prop(x, ei, ew)
        return F.log_softmax(x, dim=1)
