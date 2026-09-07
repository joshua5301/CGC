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
args = apply_hyperpara(args)
print(f'downstream: lr={args.lr:g} wd={args.weight_decay:g} dropout={args.dropout:g} epoch={args.epoch} hidden={args.n_dim}')

## cond data
begin = time.time()
args, label_cond = generate_labels_syn(args, data)
H = conv_graph_multi(args, data)

tan = None
if args.landmark == 'cgc':
    model = linear_model(args, H, data, data_test)
    H_aug, y_aug, conf = data_assessment(args, data, model, H)
    M_norm = mask_generation_conf(H_aug, y_aug, args, 'spectral', conf)
    h = torch.spmm(M_norm.to(args.device), H_aug.to(args.device))
    n_pool = len(H_aug)
else:
    depths = [H[i] for i in range(args.conv_depth + 1)]
    H_pool, y_pool, tr_pool, pool_d = select_pool(args, data, H[args.conv_depth], depths)
    n_keep = int(args.budget)
    if args.landmark in ('easy', 'hard'):
        args.budget = min(int(n_keep * args.cand_mult), len(H_pool))
    Hc = None
    if args.lam_p > 0:
        pf0 = label_feats(args.label_feat, pool_d)
        HL0 = label_feats(args.label_feat, depths)[data.train_mask]
        YL0 = F.one_hot(data.y[data.train_mask], args.num_class).to(pf0.dtype)
        W0c = fit_probe_W(HL0, YL0, args.gamma, args.ce_steps)[0]
        P0 = F.softmax(pf0.double() @ W0c, dim=1)
        Hc = posterior_feats(pf0, P0, args.lam_p)
        print(f'cluster space: {pf0.shape[1]}d feature + {P0.shape[1]}d posterior '
              f'(lam={args.lam_p:g})')
    h, assign, h_d = generate_landmarks(args, H_pool, y_pool, pool_d, Hc)
    n_cand, args.budget = len(h), n_keep

    y_L = data.y[data.train_mask]
    H_all = label_feats(args.label_feat, depths)
    H_L = H_all[data.train_mask]
    hl = label_feats(args.label_feat, h_d)
    Y_L = F.one_hot(y_L, args.num_class).to(hl.dtype)
    H_fit, T_fit = H_L, Y_L
    if args.teacher in ('probe', 'ridge'):
        fm = data.train_mask if args.distill_pool == 'train' else             torch.ones(len(H_all), dtype=torch.bool, device=H_all.device)
        H_fit = H_all[fm]
        T_fit = (teacher_targets_ridge(H_fit, H_L, Y_L, args.teacher_gamma)
                 if args.teacher == 'ridge' else
                 teacher_targets(H_fit, H_L, Y_L, args.teacher_gamma, args.teacher_temp,
                                 200, args.teacher_folds, data.train_mask[fm],
                                 args.seed)).to(hl.dtype)
        print(f'teacher[{args.teacher}]: fit {len(H_fit)} nodes  T={args.teacher_temp}  '
              f'maxp {T_fit.max(1)[0].mean():.4f}  rowsum {T_fit.sum(1).mean():.4f}  '
              f'min {T_fit.min():.3f}')

    if args.landmark in ('easy', 'hard'):
        W0 = fit_probe_W(H_fit, T_fit, args.gamma, args.ce_steps)[0]
        h, assign, h_d, e_sel, e_all = refine_landmarks(
            h, h_d, H_pool, W0, args.label_feat, n_keep, args.landmark)
        hl = label_feats(args.label_feat, h_d)
        print(f'landmark[{args.landmark}]: kept {len(h)}/{n_cand}  '
              f'entropy sel {e_sel:.4f}  all {e_all:.4f}')

    if args.label_mode == 'closed':
        Y, ctx = solve_labels(H_fit, hl, T_fit, args.beta, args.gamma, args.label_kernel)
        Y, rho = constrain(ctx, Y, Y_L.mean(0), args.constraint)
        label_cond = Y.float()
        print(f'dim: {hl.shape[1]}  rank: {ctx["rank"]}  rho: {rho:.4f}')
    elif args.label_mode == 'cs_loss':
        S = normalize_adj_sparse(data).to(args.device)
        Y_all = F.one_hot(data.y, args.num_class).to(hl.dtype)
        Y, ctx = solve_labels_cs_loss(H_all, hl, Y_all, data.train_mask, S, args.beta,
                                      args.gamma, args.cs_folds, args.cs_a1, args.cs_a2,
                                      args.cs_iters, args.cs_scale, args.ce_steps,
                                      args.seed, args.label_kernel, bool(args.cs_reset))
        label_cond = Y.float()
        print(f'dim: {hl.shape[1]}  rank: {ctx["rank"]}  loss: {ctx["loss"]:.4f}  '
              f'gnorm: {ctx["gnorm"]:.2e}  maxp: {Y.max(1)[0].mean():.4f}')
    elif args.label_mode == 'cs':
        if assign is None:
            raise SystemExit('cs needs a cluster assignment')
        S = normalize_adj_sparse(data).to(args.device)
        pm = pool_mask(args, data, len(data.y), args.device)
        Y, ctx = solve_labels_cs(H_all, H_L, Y_L, args.gamma, S, data.train_mask,
                                 pm, assign, len(hl), args.cs_a1, args.cs_a2,
                                 args.cs_iters, args.cs_scale, args.ce_steps,
                                 bool(args.cs_reset))
        label_cond = Y.float()
        na = lambda msk: (100 * (ctx['node_pred'][msk].argmax(1) == data.y[msk]).double().mean()
                          ).item() if hasattr(data, 'val_mask') else float('nan')
        print(f'C&S node-pred  val {na(data.val_mask):.2f}%  test {na(data.test_mask):.2f}%')
        print(f'dim: {hl.shape[1]}  maxp: {Y.max(1)[0].mean():.4f}  '
              f'rowsum: {Y.sum(1).mean():.3f}')
    elif args.label_mode in ('logistic', 'logistic_mean', 'probe', 'probe_mean',
                             'ridge', 'ridge_mean', 'restricted', 'weighted'):
        prior, sel = None, None
        if args.label_prior == 'cluster':
            prior = cluster_prior(assign, tr_pool, y_pool, len(hl), args.num_class,
                                  torch.float64, hl.device)
        pf = None
        if args.label_mode.endswith('_mean') or args.label_mode == 'weighted':
            if assign is None:
                raise SystemExit('*_mean needs a cluster assignment')
            pf = label_feats(args.label_feat, pool_d)
            if args.avg_pool == 'unlabeled':
                sel = ~tr_pool.to(pf.device)
                keep = int(sel.sum())
                print(f'avg pool: {keep}/{len(sel)} unlabeled  '
                      f'({keep / max(len(hl), 1):.0f} per landmark)')
                if keep == 0:
                    raise SystemExit('avg_pool unlabeled: no unlabeled node in the pool')
        if args.label_mode == 'weighted':
            W0 = fit_probe_W(H_fit, T_fit, args.gamma, args.ce_steps)[0]
            P = F.softmax(pf.double() @ W0, dim=1)
            Y, ctx, wm = solve_labels_weighted(
                H_fit, T_fit, pf, P, assign, len(hl), args.beta, args.label_kernel,
                args.w_steps, args.w_lr, args.w_mu, args.w_batch, args.seed,
                [H_pool] + list(pool_d), args.w_target)
            h, h_d = wm[0], wm[1:]
            hl, ctx['W'] = label_feats(args.label_feat, h_d), W0
        elif args.label_mode.startswith('logistic'):
            basis, n_cl = hl, 0
            if args.expert_basis > 0:
                if pf is None:
                    raise SystemExit('--expert_basis needs --label_mode logistic_mean')
                gz = torch.Generator(); gz.manual_seed(args.seed)
                pick = torch.randperm(len(pf), generator=gz)[:args.expert_basis]
                basis, n_cl = pf[pick.to(pf.device)], len(hl)
                print(f'expert basis: {len(basis)} inducing points (landmarks {len(hl)})')
            Y, ctx = solve_labels_logistic(H_fit, basis, T_fit, args.beta, args.gamma,
                                           args.ce_steps, prior, args.label_kernel,
                                           args.target_maxp, args.maxp_iters, pf, assign,
                                           n_cl, sel)
            ctx['basis'] = basis
        else:
            if args.label_mode == 'restricted':
                Y, ctx = solve_labels_restricted(H_fit, hl, T_fit, args.gamma, args.ce_steps,
                                                 args.target_maxp, args.maxp_iters, pf, assign)
            elif args.label_mode.startswith('ridge'):
                Y, ctx = solve_labels_ridge(H_fit, hl, T_fit, args.gamma, pf, assign)
            else:
                Y, ctx = solve_labels_probe(H_fit, hl, T_fit, args.gamma, args.ce_steps,
                                            args.target_maxp, args.maxp_iters, pf, assign,
                                            sel)
        label_cond = Y.float()
        if 'W' in ctx or 'dual' in ctx:
            pred = ((H_all.double() @ ctx['W']) if 'W' in ctx
                    else dual_logits(H_all, ctx.get('basis', hl), ctx['dual'])).argmax(1)
            ea = lambda msk: (100 * (pred[msk] == data.y[msk]).double().mean()).item()
            print(f'expert: train {ea(data.train_mask):.2f}%  '
                  + (f'val {ea(data.val_mask):.2f}%  ' if hasattr(data, 'val_mask') else '')
                  + (f'test {ea(data.test_mask):.2f}%' if hasattr(data, 'test_mask') else ''))
        tan = None
        if args.tangent > 0 and pf is not None and assign is not None:
            Pp = F.softmax((pf.double() @ ctx['W']) if 'W' in ctx
                           else dual_logits(pf, ctx.get('basis', hl), ctx['dual']), dim=1)
            *tan, energy = tangent_stats(H_pool, Pp, assign, len(hl), args.tan_rank)
            d0, c0, r0 = h.shape[1], Y.shape[1], args.tan_rank
            per = d0 + c0 + (r0 if r0 > 0 else d0) + c0 + 1
            shared = d0 * r0 if r0 > 0 else 0
            print(f'tangent: sigma mean {tan[2].mean():.4f}  |g| mean {tan[1].norm(dim=1).mean():.4f}  '
                  f'bytes/node {d0 + c0} -> {per} (+{shared} shared)')
            print('tangent basis energy: ' + '  '.join(
                f'r={r}:{energy[min(r, len(energy)) - 1]:.3f}' for r in (4, 8, 16, 32, 64)))
        print(f'cfg: beta={args.beta:g} whiten={args.whiten:g} kernel={args.label_kernel} '
              f'lm={args.landmark} pool={args.h_pool} feat={args.label_feat}')
        print(f'dim: {hl.shape[1]}  rank: {ctx["rank"]}  loss: {ctx["loss"]:.4f}  '
              f'gnorm: {ctx["gnorm"]:.2e}  maxp: {Y.max(1)[0].mean():.4f}  '
              f'rowsum: {Y.sum(1).mean():.3f}  min: {Y.min():.3f}'
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
    if tan is not None:
        graph.u, graph.g, graph.sig = [t.to(h.device) for t in tan]

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
        model = GNN(arch, data.num_features, args.n_dim, args.num_class, 2, args.dropout,
                    logits=(args.head == 'mse_logit')).to(args.device)
        acc.append(model_training(model, args, data, graph, data_val, data_test))
    args.test_gnn = arch
    print(f'== {arch}: {100*np.mean(acc):.2f} +- {100*np.std(acc, ddof=1) if len(acc) > 1 else 0.0:.2f}')
    result_record(args, acc)
