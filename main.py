from scr.hyperparams import *
from scr.models import *
from scr.utils import *
from scr.module import *
from scr.dataloader import *
from scr.teacher import get_teacher_labels
from scr.partition import partition

args = get_hyperparams()
args = device_setting(args)
seed_everything(args.seed)

args = override_to_best_hyperparams(args)
print(f'{args.dataset_name} r={args.ratio:g}: teacher {args.teacher_kernel} gamma={args.gamma:g} T={args.T:g} '
      f'kl_weight={args.kl_weight:g} | student lr={args.lr:g} wd={args.weight_decay:g} dropout={args.dropout}')

## data
datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)

## condensation
begin = time.time()
budget_node_num = budget(args)
H0, H1, H2 = conv_graph_multi(args, data)

P = get_teacher_labels(H2, data.train_mask, data.y, args.teacher_kernel, args.gamma, args.T, args.basis)
if data_val is None:
    print(f'teacher test acc: {100 * (P.argmax(1)[data.test_mask] == data.y[data.test_mask]).double().mean():.2f}%')
x_cond, y_cond = partition(H2, P, budget_node_num, args.kl_weight)
x_cond, y_cond = x_cond.float(), y_cond.float()
all_node_num = len(x_cond)
graph = Data(x=x_cond, y=y_cond, edge_index=torch.eye(all_node_num).nonzero().t().to(x_cond.device),
             edge_attr=torch.ones(all_node_num, device=x_cond.device), train_mask=torch.ones(all_node_num, dtype=torch.bool, device=x_cond.device))
args.cond_time = time.time() - begin
print(f'condensed: {all_node_num} nodes (budget {budget_node_num})  time {args.cond_time:.2f} s')

## student (dropout list: every value is trained, the one with the best mean validation accuracy is reported)
graph = graph.to(args.device)
results = {}
for dropout in [float(v) for v in str(args.dropout).split(',')]:
    args.dropout = dropout
    runs = []
    for repeat in range(args.repeat):
        model = GCN(data.num_features, args.n_dim, args.num_class, 2, dropout).to(args.device)
        runs.append(model_training(model, args, data, graph, data_val, data_test))
    val, test = np.mean([v for v, _ in runs]), [t for _, t in runs]
    results[dropout] = (val, np.mean(test), np.std(test, ddof=1) if len(test) > 1 else 0.0)
    print(f'-- dropout {dropout:g}: val {100 * val:.2f}  test {100 * np.mean(test):.2f} +- {100 * results[dropout][2]:.2f}')
best = max(results, key=lambda d: results[d][0])
print(f'== {args.dataset_name} r={args.ratio:g}: {100 * results[best][1]:.2f} +- {100 * results[best][2]:.2f}  (dropout {best:g}, val {100 * results[best][0]:.2f})')
