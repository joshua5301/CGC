from src.hyperparams import *
from src.models import *
from src.utils import *
from src.module import *
from src.dataloader import *
from src.teacher import get_teacher_labels
from src.partition import partition, geometric_medians
from src.edges import coarsen, to_graph
from src.partition_ot import build_transition, transition_to_edges, partition_ot_1hop
from src.partition_ot_entropic import partition_ot_1hop_entropic
from src.partition_struct import structure_bound, bound_coefficients, column_sum_max, kmeans_nonempty
from src.partition_distance import DistanceIdentity
from src.partition_mpnn import MPNNIdentity

args = get_hyperparams()
args = device_setting(args)
seed_everything(args.seed)

args = override_to_best_hyperparams(args)
print(f'{args.dataset_name} r={args.ratio:g}: teacher {args.teacher_kernel} gamma={args.gamma:g} T={args.T:g} '
      f'kl_weight={args.kl_weight:g} edges={args.edges}' + (f' ({args.ot_solver}, beta={args.neighbor_weight:g} mu={args.label_weight:g})' if args.edges == 'ot_1hop' else '') + f' | student lr={args.lr:g} wd={args.weight_decay:g} dropouts={args.dropouts}')

## data
datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)

## condensation
begin = time.time()
budget_node_num = budget(args)
H0, H1, H2 = conv_graph_multi(args, data)
X = H2

y_pred = get_teacher_labels(X, data.train_mask, data.y, args.teacher_kernel, args.gamma, args.T, args.basis)
if data_val is None:
    print(f'teacher test acc: {100 * (y_pred.argmax(1)[data.test_mask] == data.y[data.test_mask]).double().mean():.2f}%')
if args.edges in ('distance_identity', 'mpnn_identity'):
    all_node_num = budget_node_num
elif args.edges.startswith('structure') and args.struct_init == 'kmeans':
    assign, all_node_num = torch.from_numpy(kmeans_nonempty(H0.cpu().numpy(), budget_node_num, args.seed)).to(X.device), budget_node_num
else:
    X_cond, y_cond, assign = partition(X, y_pred, budget_node_num, args.kl_weight)
    all_node_num = len(X_cond)
if args.edges == 'mpnn_identity':
    out = MPNNIdentity(H0, data.edge_index, y_pred, all_node_num,
                       mu=args.label_weight, seed=args.seed, depth=args.mpnn_depth, rho=args.mpnn_rho,
                       batch_size=args.distance_batch_size,
                       median_iters=args.distance_median_iters).run(args.outer_iters)
    X_cond, y_cond, assign = out['H_cond'], out['Y_cond'], out['assign']
    ids = torch.arange(all_node_num, device=X_cond.device)
    edge_index, edge_attr = torch.stack([ids, ids]), torch.ones(all_node_num, device=X_cond.device)
elif args.edges == 'distance_identity':
    out = DistanceIdentity(H0, normalize_adj_sparse(data), y_pred, all_node_num,
                           mu=args.label_weight, seed=args.seed,
                           batch_size=args.distance_batch_size,
                           median_iters=args.distance_median_iters).run(args.outer_iters)
    X_cond, y_cond, assign = out['H_cond'], out['Y_cond'], out['assign']
    ids = torch.arange(all_node_num, device=X_cond.device)
    edge_index, edge_attr = torch.stack([ids, ids]), torch.ones(all_node_num, device=X_cond.device)
    data = attach_propagation(data)
    if data_val is not None:
        data_val, data_test = attach_propagation(data_val), attach_propagation(data_test)
elif args.edges == 'none':
    edge_index, edge_attr = torch.eye(all_node_num).nonzero().t().to(X_cond.device), torch.ones(all_node_num, device=X_cond.device)
elif args.edges == 'ot_1hop':
    # 1-hop neighbourhood-OT objective on the raw features; the GRIP partition (on A^2 X) is the initial assignment
    P, meta = build_transition(data.edge_index.cpu().numpy(), len(H0))
    print(f'transition: {meta}  level {args.ot_level}')
    H_struct = H0
    if args.ot_level == 'prop1':   # structural features P X: the condensed node carries one hop, its edges the second
        Psp = torch.sparse_coo_tensor(torch.from_numpy(np.vstack(P.nonzero())).long().to(H0.device),
                                      torch.from_numpy(P.data).float().to(H0.device), (len(H0), len(H0)))
        H_struct = torch.sparse.mm(Psp, H0)
    if args.ot_solver == 'exact':
        out = partition_ot_1hop(H_struct.cpu().numpy(), P, y_pred.cpu().numpy(), all_node_num, assign.cpu().numpy(),
                                args.root_weight, args.neighbor_weight, args.label_weight, args.outer_iters, args.max_lp_variables)
    else:
        out = partition_ot_1hop_entropic(H_struct, P, y_pred, all_node_num, assign.cpu().numpy(),
                                         args.root_weight, args.neighbor_weight, args.label_weight, args.ot_eps,
                                         args.sink_iters, args.sink_iters, args.candidates, args.device, args.outer_iters)
    X_cond, y_cond = torch.from_numpy(out['H_cond']).to(args.device), torch.from_numpy(out['Y_cond']).to(args.device)
    ei, ea = transition_to_edges(out['P_cond'])
    edge_index, edge_attr = torch.from_numpy(ei).long().to(args.device), torch.from_numpy(ea).float().to(args.device)
    prop = args.ot_level == 'prop1'
    data = attach_transition(data, prop)
    if data_val is not None:
        data_val, data_test = attach_transition(data_val, prop), attach_transition(data_test, prop)
elif args.edges.startswith('structure'):
    # structure-aware partition from the message-passing risk bound: J = (alpha D_H + beta D_S + mu D_KL) / N on the
    # raw features H0 and the row-stochastic P; initial assignment = k-means on raw X (or the GRIP partition), teacher unchanged
    P, meta = build_transition(data.edge_index.cpu().numpy(), len(H0))
    if args.struct_coef == 'bound':
        R = float(H0.norm(dim=1).max())
        alpha, beta = bound_coefficients(column_sum_max(P), 2, args.struct_abar, R, all_node_num)
        mu = 1.0
        print(f'transition: {meta}  bound coefficients: K 2, abar {args.struct_abar:g}, R {R:.3f} -> alpha {alpha:.4g} beta {beta:.4g} mu 1')
    else:
        alpha, beta, mu = args.root_weight, args.neighbor_weight, args.label_weight
        print(f'transition: {meta}  init {args.struct_init}  surrogate coefficients alpha {alpha:g} beta {beta:g} mu {mu:g}')
    out = structure_bound(H0.cpu().numpy(), P, y_pred.cpu().numpy(), all_node_num, assign.cpu().numpy(), alpha, beta, mu,
                          'identity' if args.edges == 'structure_identity' else 'learned_median', args.outer_iters, args.seed)
    assign = torch.from_numpy(out['assign']).to(X.device)
    y_cond = torch.from_numpy(out['Y_cond']).to(args.device)
    # the condensed graph of the objective: raw-X medians and Q, used as given (no renormalisation)
    X_cond = torch.from_numpy(out['H_cond']).to(args.device)
    ei, ea = transition_to_edges(out['Q'])
    edge_index, edge_attr = torch.from_numpy(ei).long().to(args.device), torch.from_numpy(ea).float().to(args.device)
    attach = attach_transition if args.struct_student == 'faithful' else attach_propagation   # served on P, or on A_hat as the main table
    data = attach(data)
    if data_val is not None:
        data_val, data_test = attach(data_val), attach(data_test)
else:
    # A' on the one-hop features: representatives of A X, edges = cell-averaged propagation matrix, so that A' X' ~ A^2 X cell-wise
    adj = normalize_adj_sparse(data).to(data.x.device)
    X_cond = geometric_medians(H1.double(), assign, all_node_num)
    edge_index, edge_attr = to_graph(coarsen(adj, assign, all_node_num).float())
    data = attach_propagation(data)
    if data_val is not None:
        data_val, data_test = attach_propagation(data_val), attach_propagation(data_test)
X_cond, y_cond = X_cond.float(), y_cond.float()
graph = Data(x=X_cond, y=y_cond, edge_index=edge_index, edge_attr=edge_attr, train_mask=torch.ones(all_node_num, dtype=torch.bool, device=X_cond.device))
args.cond_time = time.time() - begin
print(f'condensed: {all_node_num} nodes (budget {budget_node_num}), {len(edge_attr)} edges  time {args.cond_time:.2f} s')

## student
graph = graph.to(args.device)
faithful = args.edges.startswith('structure') and args.struct_student == 'faithful'
results = {}
for dropout in [float(v) for v in str(args.dropouts).split(',')]:
    runs = []
    for repeat in range(args.repeat):
        model = (SAGE(data.num_features, args.n_dim, args.num_class, 2, dropout) if args.edges == 'ot_1hop' else
                 PropGNN(data.num_features, args.n_dim, args.num_class, 2, dropout) if faithful else
                 GCN(data.num_features, args.n_dim, args.num_class, 2, dropout, normalize=args.edges in ('none', 'mpnn_identity'))).to(args.device)
        runs.append(model_training(model, args, data, graph, data_val, data_test))
    val, test = np.mean([v for v, _ in runs]), [t for _, t in runs]
    results[dropout] = (val, np.mean(test), np.std(test, ddof=1) if len(test) > 1 else 0.0)
    print(f'-- dropout {dropout:g}: val {100 * val:.2f}  test {100 * np.mean(test):.2f} +- {100 * results[dropout][2]:.2f}')
best = max(results, key=lambda d: results[d][0])
print(f'== {args.dataset_name} r={args.ratio:g}: {100 * results[best][1]:.2f} +- {100 * results[best][2]:.2f}  (dropout {best:g}, val {100 * results[best][0]:.2f})')
