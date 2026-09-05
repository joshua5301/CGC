from scr.para import *
from scr.models import *
from scr.utils import *
from scr.module import *
from scr.dataloader import *
from scr.label_solve import *

args = para()
args.result_path =  f'./results/'
args = create_folder(args)
args = device_setting(args)
seed_everything(args.seed)

## data
datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)

##hyper para
args = hyperpara(args) if args.generate_adj == 1 else hyperpara_noadj(args)

## cond data
begin = time.time()
args, label_cond = generate_labels_syn(args, data)
H = conv_graph_multi(args, data)

if args.landmark == 'cgc':
    model = linear_model(args, H, data, data_test)
    H_aug, y_aug, conf = data_assessment(args, data, model, H)
    M_norm = mask_generation_conf(H_aug, y_aug, args, 'spectral', conf)
    h = torch.spmm(M_norm.to(args.device), H_aug.to(args.device))
    n_pool = len(H_aug)
else:
    depths = [H[i] for i in range(args.conv_depth + 1)]
    H_pool, y_pool, tr_pool, pool_d = select_pool(args, data, H[args.conv_depth], depths)
    h, assign, h_d = generate_landmarks(args, H_pool, y_pool, pool_d)

    y_L = data.y[data.train_mask]
    H_all = label_feats(args.label_feat, depths)
    H_L = H_all[data.train_mask]
    hl = label_feats(args.label_feat, h_d)
    Y_L = F.one_hot(y_L, args.num_class).to(hl.dtype)
    H_fit, T_fit = H_L, Y_L
    if args.teacher == 'probe':
        fm = data.train_mask if args.distill_pool == 'train' else             torch.ones(len(H_all), dtype=torch.bool, device=H_all.device)
        H_fit = H_all[fm]
        T_fit = teacher_targets(H_fit, H_L, Y_L, args.teacher_gamma, args.teacher_temp,
                                200, args.teacher_folds, data.train_mask[fm],
                                args.seed).to(hl.dtype)
        print(f'teacher: fit {len(H_fit)} nodes  T={args.teacher_temp}  '
              f'maxp_teacher {T_fit.max(1)[0].mean():.4f}')

    if args.label_mode == 'closed':
        Y, ctx = solve_labels(H_fit, hl, T_fit, args.beta, args.gamma, args.label_kernel)
        Y, rho = constrain(ctx, Y, Y_L.mean(0), args.constraint)
        label_cond = Y.float()
        print(f'dim: {hl.shape[1]}  rank: {ctx["rank"]}  rho: {rho:.4f}')
    elif args.label_mode in ('logistic', 'probe', 'probe_mean'):
        prior = None
        if args.label_prior == 'cluster':
            prior = cluster_prior(assign, tr_pool, y_pool, len(hl), args.num_class,
                                  torch.float64, hl.device)
        if args.label_mode == 'logistic':
            Y, ctx = solve_labels_logistic(H_fit, hl, T_fit, args.beta, args.gamma,
                                           args.ce_steps, prior, args.label_kernel,
                                           args.target_maxp)
        else:
            pf = None
            if args.label_mode == 'probe_mean':
                if assign is None:
                    raise SystemExit('probe_mean needs a cluster assignment')
                pf = label_feats(args.label_feat, pool_d)
            Y, ctx = solve_labels_probe(H_fit, hl, T_fit, args.gamma, args.ce_steps,
                                        args.target_maxp, pool=pf, assign=assign)
        label_cond = Y.float()
        print(f'cfg: beta={args.beta:g} whiten={args.whiten:g} kernel={args.label_kernel} '
              f'lm={args.landmark} pool={args.h_pool} feat={args.label_feat}')
        print(f'dim: {hl.shape[1]}  rank: {ctx["rank"]}  loss: {ctx["loss"]:.4f}  '
              f'gnorm: {ctx["gnorm"]:.2e}  maxp: {Y.max(1)[0].mean():.4f}'
              + (f'  gamma*: {ctx["gamma_rel"]:.3g}' if 'gamma_rel' in ctx else ''))
    else:
        label_cond = onehot_labels(args, hl, H_L, y_L, assign, y_pool, tr_pool)
    n_pool = len(H_pool)

label_cond = label_cond.to(args.device)
if args.generate_adj == 1:
    if args.adj_mode in ('coarsen', 'commute'):
        if args.landmark == 'cgc' or assign is None:
            raise SystemExit('adj_mode needs a cluster assignment '
                             '(--landmark kmeans/class_kmeans/random_split)')
        pm = pool_mask(args, data, len(data.y), args.device)
        a = coarsen_adj(data.edge_index, getattr(data, 'edge_attr', None), pm, assign, len(h))
        if args.adj_mode == 'commute':
            a = commute_adj(h_d, args.conv_depth, args.adj_steps, args.adj_lr, args.adj_l1,
                            a if args.adj_init == 'coarsen' else None).float()
    else:
        a = get_adj(h, args.adj_T)
    if args.cond_feat == 'raw':
        x = h_d[0]
    elif args.cond_feat == 'prop':
        x = h
    else:
        x = get_feature(a, h, args.alpha)
    if args.landmark != 'cgc':
        res = commutation_residual(normalize_adj_tensor(a), [x] + h_d[1:])
        print('commute resid: ' + '  '.join(f'k={i+1}:{r:.4f}' for i, r in enumerate(res))
              + f'   #edges: {int((a > 0).sum())}')
    graph = Data(x=x, y=label_cond, edge_index=a.nonzero().t(), edge_attr=a[a.nonzero()[:,0], a.nonzero()[:,1]], train_mask=torch.ones(len(x), dtype=torch.bool))
else:
    graph = Data(x=h, y=label_cond, edge_index=torch.eye(len(h)).nonzero().t(), edge_attr=torch.ones(len(h)), train_mask=torch.ones(len(h), dtype=torch.bool))

args.cond_time = time.time()-begin
print('Condensation time:',  f'{args.cond_time:.3f}', 's')
print('#edges:', int(torch.sum(a).item())) if args.generate_adj == 1 else print('No adj')
print('#nodes:', len(h))
print('#training labels:', data.train_mask.sum().item())
print('#pool:', n_pool)
args.changed_label = n_pool-data.train_mask.sum().item()

# model training
graph=graph.to(args.device)
ARCHS = ['gcn', 'sage', 'gat', 'cheby', 'appnp'] if args.test_gnn == 'all'     else [k.strip().lower() for k in args.test_gnn.split(',') if k.strip()]
for arch in ARCHS:
    acc = []
    for repeat in range(args.repeat):
        model = GNN(arch, data.num_features, args.n_dim, args.num_class, 2, args.dropout).to(args.device)
        acc.append(model_training(model, args, data, graph, data_val, data_test))
    args.test_gnn = arch
    print(f'== {arch}: {100*np.mean(acc):.2f} +- {100*np.std(acc, ddof=1) if len(acc) > 1 else 0.0:.2f}')
    result_record(args, acc)
