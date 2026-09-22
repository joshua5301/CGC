"""Student train/serve mismatch (epsilon_student).
The student is trained on the condensed graph with A' = I, i.e. it learns an MLP on the propagated
features H = A^2 X.  At serving time it is run as a GCN on (X, A).  This script trains the student
exactly as main.py does and evaluates the same checkpoint in both ways:
  gcn : GCN on the original graph (the protocol; val selection uses this)
  iso : the same weights applied to H = A^2 X with an identity graph (what the student was trained for)
The gap iso - gcn is the accuracy the isolated-node condensation gives away to the architecture
mismatch; it is the upper bound on what condensed edges can recover."""
from src.hyperparams import *
from src.models import *
from src.utils import *
from src.module import *
from src.dataloader import *
from src.teacher import get_teacher_labels
from src.partition import partition

args = get_hyperparams()
args = device_setting(args)
seed_everything(args.seed)
args = override_to_best_hyperparams(args)

datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)


def isolated(d):
    H = conv_graph_multi(args, d)[-1]
    n = len(H)
    return Data(x=H, y=d.y, edge_index=torch.eye(n).nonzero().t().to(H.device), edge_attr=torch.ones(n, device=H.device))


views = {'gcn': (data, data_val, data_test), 'iso': (isolated(data), None if data_val is None else isolated(data_val), None if data_test is None else isolated(data_test))}


def evaluate(model, view):
    d, dv, dt = views[view]
    model.eval()
    with torch.no_grad():
        if dv is None:
            out = model(d)
            val = out[data.val_mask].argmax(1).eq(data.y[data.val_mask]).double().mean().item()
            pred = out[data.test_mask].argmax(1)
            test = pred.eq(data.y[data.test_mask]).double().mean().item()
        else:
            val = model(dv).argmax(1).eq(dv.y).double().mean().item()
            pred = model(dt).argmax(1)
            test = pred.eq(dt.y).double().mean().item()
    return val, test, pred


def train(model, graph):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {'gcn': (0, 0, None), 'iso': (0, 0, None)}
    at_gcn = None  # iso accuracy at the gcn-selected checkpoint
    for epoch in range(1, args.epoch + 1):
        if epoch == args.epoch // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
        model.train()
        output = model(graph)
        loss = -(graph.y * output).sum(1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % args.eval_every != 0 and epoch != args.epoch:
            continue
        cur = {v: evaluate(model, v) for v in views}
        for v in views:
            if cur[v][0] > best[v][0]:
                best[v] = cur[v]
                if v == 'gcn':
                    at_gcn = cur['iso']
    return best, at_gcn


## condensation (as main.py)
budget_node_num = budget(args)
X = conv_graph_multi(args, data)[-1]
y_pred = get_teacher_labels(X, data.train_mask, data.y, args.teacher_kernel, args.gamma, args.T, args.basis)
X_cond, y_cond = partition(X, y_pred, budget_node_num, args.kl_weight)
X_cond, y_cond = X_cond.float(), y_cond.float()
m = len(X_cond)
graph = Data(x=X_cond, y=y_cond, edge_index=torch.eye(m).nonzero().t().to(X_cond.device), edge_attr=torch.ones(m, device=X_cond.device)).to(args.device)

print(f'{args.dataset_name} r={args.ratio:g}: {m} nodes, teacher {args.teacher_kernel} gamma={args.gamma:g} T={args.T:g} kl={args.kl_weight:g}')
print('  per dropout, mean over repeats (%):  gcn = serve as GCN on (X, A) [protocol];  iso@gcn = same checkpoint as MLP on A^2 X;'
      '  iso = MLP on A^2 X with its own val selection;  dis = test disagreement of the two views at the gcn checkpoint')
for dropout in [float(v) for v in str(args.dropouts).split(',')]:
    rows = []
    for repeat in range(args.repeat):
        model = GCN(data.num_features, args.n_dim, args.num_class, 2, dropout).to(args.device)
        best, at_gcn = train(model, graph)
        dis = best['gcn'][2].ne(at_gcn[2]).double().mean().item()
        rows.append((best['gcn'][1], at_gcn[1], best['iso'][1], dis, best['gcn'][0], best['iso'][0]))
    g, ig, i, dis, vg, vi = (100 * np.mean([r[k] for r in rows]) for k in range(6))
    print(f'  dropout {dropout:g}: gcn {g:.2f}  iso@gcn {ig:.2f} ({ig - g:+.2f})  iso {i:.2f} ({i - g:+.2f})  dis {dis:.2f}%   [val gcn {vg:.2f} iso {vi:.2f}]')
