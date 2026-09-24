from src.hyperparams import *
from src.models import *
from src.utils import *
from src.module import *
from src.dataloader import *
from src.teacher import get_teacher_labels
from src.partition import partition
from src.metric import label_scale
from src.refine import refine_sgc

args = get_hyperparams()
args = device_setting(args)
seed_everything(args.seed)

args = override_to_best_hyperparams(args)
print(f'{args.dataset_name} r={args.ratio:g}: teacher {args.teacher_kernel} gamma={args.gamma:g} T={args.T:g} '
      f'kl_weight={args.kl_weight:g} | student lr={args.lr:g} wd={args.weight_decay:g} dropouts={args.dropouts}')

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
if args.sgc_refine:
    result = refine_sgc(X, y_pred, budget_node_num, args.kl_weight, args.refine_beta,
                        args.refine_rounds, args.sgc_ridge, args.sgc_steps,
                        args.grip_steps, args.refine_tolerance)
    if data_val is None:
        prediction = (X.double() @ result['W'] + result['b']).argmax(1)
        val = (prediction[data.val_mask] == data.y[data.val_mask]).double().mean()
        test = (prediction[data.test_mask] == data.y[data.test_mask]).double().mean()
        print(f'SGC val {100 * val:.2f} test {100 * test:.2f} | teacher risk {result["risk"]:.8f}')
    else:
        print(f'SGC teacher risk {result["risk"]:.8f}; evaluation supported for transductive datasets only')
    raise SystemExit(0)
elif args.metric_alpha > 0:
    scale = label_scale(X, y_pred, args.metric_alpha, args.metric_pairs,
                        args.metric_ridge, args.metric_steps, args.seed)
    X_cond, y_cond = partition(X * scale, y_pred, budget_node_num, args.kl_weight)
    X_cond = X_cond / scale
else:
    X_cond, y_cond = partition(X, y_pred, budget_node_num, args.kl_weight)
X_cond, y_cond = X_cond.float(), y_cond.float()
all_node_num = len(X_cond)
graph = Data(x=X_cond, y=y_cond, edge_index=torch.eye(all_node_num).nonzero().t().to(X_cond.device),
             edge_attr=torch.ones(all_node_num, device=X_cond.device), train_mask=torch.ones(all_node_num, dtype=torch.bool, device=X_cond.device))
args.cond_time = time.time() - begin
print(f'condensed: {all_node_num} nodes (budget {budget_node_num})  time {args.cond_time:.2f} s')

## student
graph = graph.to(args.device)
results = {}
for dropout in [float(v) for v in str(args.dropouts).split(',')]:
    runs = []
    for repeat in range(args.repeat):
        model = GCN(data.num_features, args.n_dim, args.num_class, 2, dropout).to(args.device)
        runs.append(model_training(model, args, data, graph, data_val, data_test))
    val, test = np.mean([v for v, _ in runs]), [t for _, t in runs]
    results[dropout] = (val, np.mean(test), np.std(test, ddof=1) if len(test) > 1 else 0.0)
    print(f'-- dropout {dropout:g}: val {100 * val:.2f}  test {100 * np.mean(test):.2f} +- {100 * results[dropout][2]:.2f}')
best = max(results, key=lambda d: results[d][0])
print(f'== {args.dataset_name} r={args.ratio:g}: {100 * results[best][1]:.2f} +- {100 * results[best][2]:.2f}  (dropout {best:g}, val {100 * results[best][0]:.2f})')
