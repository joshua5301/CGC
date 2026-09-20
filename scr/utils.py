from scr.hyperparams import *
from scr.module import *
from scr.models import *


# Following GCond and other condensation methods...
BUDGET = {('cora', 0.013): 35,  ('cora', 0.026): 70,  ('cora', 0.052): 140,
          ('citeseer', 0.009): 30,  ('citeseer', 0.018): 60,  ('citeseer', 0.036): 120,
          ('arxiv', 0.0005): 90,  ('arxiv', 0.0025): 454,  ('arxiv', 0.005): 909,
          ('flickr', 0.001): 44,  ('flickr', 0.005): 223,  ('flickr', 0.01): 446,
          ('reddit', 0.0005): 77,  ('reddit', 0.001): 153,  ('reddit', 0.002): 307}

def budget(args):
    key = (args.dataset_name, args.ratio)
    return BUDGET[key]


def device_setting(args):
    if args.gpu != -1:
        args.device = 'cuda'
        torch.cuda.set_device(args.gpu)
    else:
        args.device = 'cpu'
    return args


def conv_graph_multi(args, data):
    adj_norm = normalize_adj_sparse(data).to(data.x.device)
    H0 = data.x
    H1 = torch.spmm(adj_norm, H0)
    H2 = torch.spmm(adj_norm, H1)
    return H0, H1, H2


def normalize_adj_sparse(data):
    mx = sp.csr_matrix((np.ones(data.edge_index.shape[1]), data.edge_index.cpu().numpy()), shape=(data.x.shape[0], data.x.shape[0]))
    mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)

    sparse_mx = mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    adj = torch.sparse_coo_tensor(sparseconcat.t(), sparsedata, sparse_mx.shape)
    return adj


def model_training(model, args, data, graph, data_val=None, data_test=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc = test_acc = 0
    for epoch in range(1, args.epoch+1):
        if epoch == args.epoch // 2:
            lr = args.lr*0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

        model.train()
        output = model(graph)
        loss = -(graph.y[graph.train_mask] * output[graph.train_mask]).sum(1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every != 0 and epoch != args.epoch:
            continue
        if args.dataset_name in ['flickr', 'reddit']:
            train_acc, val_acc, tmp_test_acc = test_inductive(args, model, data_val, data_test)
        else:
            train_acc, val_acc, tmp_test_acc = test(model, data.to(args.device))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if epoch%100 == 0 :
            print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Train: {train_acc:.4f}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
    print()
    return best_val_acc, test_acc

def test_inductive(args, model, data_val, data_test, k=2):
    with torch.no_grad():
        model.eval()
        accs = []
        accs.append(0)
        for data in [data_val, data_test]:
            out = model(data)
            pred = out.argmax(1)
            acc = pred.eq(data.y).sum().item() / len(data.y)
            accs.append(acc)
    return accs

def test(model, data):
    with torch.no_grad():
        model.eval()
        out, accs = model(data), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = out[mask].argmax(1)
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return accs
