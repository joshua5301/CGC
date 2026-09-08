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

SOFT_MODES = ('closed', 'logistic', 'logistic_mean', 'probe', 'probe_mean', 'ridge',
              'ridge_mean', 'restricted', 'weighted', 'gcn_mean', 'mlp_mean', 'cs', 'cs_loss')
if (args.label_mode in SOFT_MODES and args.head == 'mse'
        and 'head' not in getattr(args, 'explicit', set())):
    args.head = 'ce'
    print('note: soft labels -> downstream head set to ce (pass --head to override)')
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
                             'ridge', 'ridge_mean', 'restricted', 'weighted', 'gcn_mean', 'mlp_mean'):
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
        if args.label_mode == 'mlp_mean':
            if assign is None:
                raise SystemExit('mlp_mean needs a cluster assignment')
            tf = fit_mlp_teacher(H_fit, T_fit, args.n_dim, 500, 1e-2, args.weight_decay,
                                 args.dropout, args.seed)
            with torch.no_grad():
                P = F.softmax(tf(pf), dim=1).double()
                tp = tf(H_all).argmax(1)
            yd = data.y.to(tp.device)
            ta = lambda m: (100 * (tp[m.to(tp.device)] == yd[m.to(tp.device)]).double().mean()).item()
            print(f'teacher[mlp]: train {ta(data.train_mask):.2f}%'
                  + (f'  val {ta(data.val_mask):.2f}%  test {ta(data.test_mask):.2f}%'
                     if hasattr(data, 'val_mask') else ''))
            Y = teacher_mean_labels(P, assign, len(hl), sel)
            ctx = {'loss': float('nan'), 'gnorm': 0.0,
                   'rank': int(torch.linalg.matrix_rank(hl)), 'P': P}
        elif args.label_mode == 'gcn_mean':
            if assign is None:
                raise SystemExit('gcn_mean needs a cluster assignment')
            print('teacher[gcn]: training on the full graph')
            tm = GNN('gcn', data.num_features, args.n_dim, args.num_class, 2,
                     args.dropout).to(args.device)
            model_training(tm, args, data, data.to(args.device), data_val, data_test)
            tm.eval()
            with torch.no_grad():
                Pfull = tm(data.to(args.device)).exp()
            pm = pool_mask(args, data, len(data.y), args.device).to(Pfull.device)
            yd, tp = data.y.to(Pfull.device), Pfull.argmax(1)
            ta = lambda m: (100 * (tp[m.to(Pfull.device)] == yd[m.to(Pfull.device)]).double().mean()).item()
            print(f'teacher[gcn]: train {ta(data.train_mask):.2f}%'
                  + (f'  val {ta(data.val_mask):.2f}%  test {ta(data.test_mask):.2f}%'
                     if hasattr(data, 'val_mask') else ''))
            P = Pfull[pm].double()
            Y = teacher_mean_labels(P, assign, len(hl), sel)
            ctx = {'loss': float('nan'), 'gnorm': 0.0,
                   'rank': int(torch.linalg.matrix_rank(hl)), 'P': P}
        elif args.label_mode == 'weighted':
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
        if args.feat_sub:
            Wl = ctx['W'] if 'W' in ctx else fit_probe_W(H_fit, T_fit, args.gamma, args.ce_steps)[0]
            Bq = torch.linalg.qr(Wl.double().to(h.device))[0].float()
            res = (h - h @ Bq @ Bq.T).norm() / h.norm()
            h = h @ Bq @ Bq.T
            print(f'feat_sub: rank {Bq.shape[1]}  dropped energy {res ** 2:.3f}  '
                  f'bytes/node {h.shape[1] + Y.shape[1]} -> {2 * Bq.shape[1]} (+{Bq.numel()} shared)')
        if args.cell_k > 0 and pf is not None:
            Pk = (ctx['P'] if 'P' in ctx else
                  F.softmax((pf.double() @ ctx['W']) if 'W' in ctx
                            else dual_logits(pf, ctx.get('basis', hl), ctx['dual']), dim=1))
            knn = knn_cells(h, H_pool, min(args.cell_k, len(H_pool)))
            cover = torch.unique(knn).numel()
            h = knn_means(H_pool, knn)
            h_d = [knn_means(E, knn) for E in pool_d]
            hl = label_feats(args.label_feat, h_d)
            Y = knn_means(Pk, knn)
            print(f'cell_k: K={knn.shape[1]}  voronoi mean {len(H_pool) / len(h):.0f}  '
                  f'pool covered {cover / len(H_pool):.2%}  multiplicity {knn.numel() / cover:.2f}')
        tan = None
        if ((args.tangent > 0 or args.adj_mode == 'tangent' or args.tan_static)
                and pf is not None and assign is not None):
            Pp = ctx['P'] if 'P' in ctx else F.softmax((pf.double() @ ctx['W']) if 'W' in ctx
                           else dual_logits(pf, ctx.get('basis', hl), ctx['dual']), dim=1)
            *tan, energy = tangent_stats(H_pool, Pp, assign, len(hl), args.tan_rank,
                                         args.tan_code)
            d0, c0, r0, k0 = h.shape[1], Y.shape[1], args.tan_rank, args.tan_code
            n_tan = len(hl)
            if args.tan_frac < 1:
                score = tan[2] * tan[1].norm(dim=1)
                n_tan = max(int(round(args.tan_frac * len(hl))), 1)
                drop = score.argsort(descending=True)[n_tan:]
                tan[2][drop] = 0.0
                print(f'tangent frac: kept {n_tan}/{len(hl)} cells  '
                      f'score cut {score.sort(descending=True)[0][n_tan - 1]:.4f}')
            per = d0 + c0 + (n_tan / len(hl)) * ((1 if k0 > 0 else r0 if r0 > 0 else d0) + c0 + 1)
            shared = d0 * (k0 if k0 > 0 else r0)
            print(f'tangent: sigma mean {tan[2].mean():.4f}  |g| mean {tan[1].norm(dim=1).mean():.4f}  '
                  f'bytes/node {d0 + c0} -> {per:.0f} (+{shared} shared)')
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
    elif args.adj_mode == 'tangent':
        if tan is None:
            raise SystemExit('adj_mode tangent needs a *_mean label mode with a cluster assignment')
        a = tangent_adj(h, tan[0].to(h.device), tan[2].to(h.device), args.tan_edge_k, args.tan_edge_T)
        deg = (a > 0).sum(1)
        print(f'tangent adj: {int((a > 0).sum()) // 2} undirected edges  '
              f'{int((deg > 0).sum())}/{len(h)} nodes connected  mean deg {deg.float().mean():.2f}')
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
    if args.relabel and pf is not None and assign is not None:
        Ah, Xp = normalize_adj_tensor(a).to(x.device), x
        for _ in range(args.conv_depth):
            Xp = Ah @ Xp
        delta = Xp - h.to(x.device)
        label_cond = relabel_shifted(pf, assign, len(h), delta, ctx, ctx.get('basis', hl),
                                     sel).float().to(args.device)
        print(f'relabel: mean shift {delta.norm(dim=1).mean():.4f}  '
              f'maxp {label_cond.max(1)[0].mean():.4f}')
    graph = Data(x=x, y=label_cond, edge_index=a.nonzero().t(), edge_attr=a[a.nonzero()[:,0], a.nonzero()[:,1]], train_mask=torch.ones(len(x), dtype=torch.bool))
else:
    xs, ys, n0 = h, label_cond.to(h.device), len(h)
    ei = torch.eye(n0).nonzero().t()
    if args.tan_static and tan is not None:
        U, G, S = [t.to(h.device) for t in tan]
        dx, dy = S.unsqueeze(1) * U, S.unsqueeze(1) * G
        nrm = lambda Y: (Y.clamp_min(0) / Y.clamp_min(0).sum(1, keepdim=True).clamp_min(1e-12))
        keep_c = args.tan_static == 1
        xs = torch.cat(([h] if keep_c else []) + [h + dx, h - dx])
        ys = torch.cat(([ys] if keep_c else []) + [nrm(ys + dy), nrm(ys - dy)])
        ei = torch.eye(len(xs)).nonzero().t()
        if args.tan_static_edge and keep_c:
            j = torch.arange(n0)
            star = torch.stack([torch.cat([j, j, j + n0, j + 2 * n0]),
                                torch.cat([j + n0, j + 2 * n0, j, j])])
            ei = torch.cat([ei, star], dim=1)
        print(f'static endpoints: {n0 if keep_c else 0} centres + {2 * n0} endpoints'
              + ('  (star edges)' if args.tan_static_edge else '  (no edges)')
              + f'  maxp {ys.max(1)[0].mean():.4f}')
    graph = Data(x=xs, y=ys, edge_index=ei.to(h.device), edge_attr=torch.ones(ei.shape[1], device=h.device),
                 train_mask=torch.ones(len(xs), dtype=torch.bool, device=h.device))
    if tan is not None and args.tangent > 0 and not args.tan_static:
        graph.u, graph.g, graph.sig = [t.to(h.device) for t in tan]
    if args.gen > 0:
        if assign is None or not ('W' in ctx or 'dual' in ctx):
            raise SystemExit('--gen needs a *_mean label mode with an explicit teacher (probe/logistic)')
        graph.sd = cell_std(H_pool, assign, len(h)).to(h.device)
        if 'W' in ctx:
            Wt = ctx['W'].float().to(h.device)
            graph.teacher = lambda x: x @ Wt
        else:
            basis = ctx.get('basis', hl).to(h.device)
            dual = ctx['dual']
            graph.teacher = lambda x: dual_logits(x, basis, dual).float()
        print(f'gen: per-cell std mean {graph.sd.mean():.4f}  bytes/node '
              f'{h.shape[1] + label_cond.shape[1]} -> {2 * h.shape[1] + label_cond.shape[1]}')

args.cond_time = time.time()-begin
print('Condensation time:',  f'{args.cond_time:.3f}', 's')
print('#edges:', int(torch.sum(a).item())) if args.generate_adj == 1 else print('No adj')
print('#nodes:', graph.num_nodes)
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
