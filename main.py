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
    H_L = label_feats(args.label_feat, [D[data.train_mask] for D in depths])
    hl = label_feats(args.label_feat, h_d)
    Y_L = F.one_hot(y_L, args.num_class).to(hl.dtype)

    if args.label_mode == 'closed':
        Y, ctx = solve_labels(H_L, hl, Y_L, args.beta, args.gamma, args.label_kernel)
        Y, rho = constrain(ctx, Y, Y_L.mean(0), args.constraint)
        label_cond = Y.float()
        print(f'dim: {hl.shape[1]}  rank: {ctx["rank"]}  rho: {rho:.4f}')
    elif args.label_mode in ('logistic', 'probe'):
        prior = None
        if args.label_prior == 'cluster':
            prior = cluster_prior(assign, tr_pool, y_pool, len(hl), args.num_class,
                                  torch.float64, hl.device)
        if args.label_mode == 'logistic':
            Y, ctx = solve_labels_logistic(H_L, hl, Y_L, args.beta, args.gamma, args.ce_steps,
                                          prior, args.label_kernel)
        else:
            Y, ctx = solve_labels_probe(H_L, hl, Y_L, args.gamma, args.ce_steps)
        label_cond = Y.float()
        print(f'dim: {hl.shape[1]}  rank: {ctx["rank"]}  loss: {ctx["loss"]:.4f}  '
              f'gnorm: {ctx["gnorm"]:.2e}  maxp: {Y.max(1)[0].mean():.4f}')
    else:
        label_cond = onehot_labels(args, hl, H_L, y_L, assign, y_pool, tr_pool)
    n_pool = len(H_pool)

label_cond = label_cond.to(args.device)
if args.generate_adj == 1:
    a = get_adj(h, args.adj_T)
    x = get_feature(a, h, args.alpha)
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
acc= []
for repeat in range(args.repeat): 
    model = GCN(data.num_features, args.n_dim, args.num_class, 2, args.dropout).to(args.device)
    args.test_gnn = model.__class__.__name__
    acc.append(model_training(model, args, data, graph, data_val, data_test))
result_record(args, acc)
