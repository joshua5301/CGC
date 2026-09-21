from src.hyperparams import *
from src.utils import *
from src.module import *
from src.dataloader import *
from src.teacher import get_kernel_values, fit_logistic

args = get_hyperparams()
args = device_setting(args)
seed_everything(args.seed)
args = override_to_best_hyperparams(args)

datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)
X = conv_graph_multi(args, data)[-1].double()
B = X if args.basis >= len(X) else X[torch.randperm(len(X))[:args.basis]]
K_BB = get_kernel_values(B, B, args.teacher_kernel)
K_BB = (K_BB + K_BB.T) / 2
eye = torch.eye(len(B), dtype=B.dtype, device=B.device)
L = torch.linalg.cholesky(K_BB + 1e-8 * K_BB.diagonal().mean() * eye)
T = torch.linalg.solve_triangular(L, eye, upper=False).T
feat = lambda Z: torch.cat([get_kernel_values(z, B, args.teacher_kernel) @ T for z in Z.double().split(8192)])

Phi = feat(X)
if data_val is None:
    evals = {'val': (Phi[data.val_mask], data.y[data.val_mask]), 'test': (Phi[data.test_mask], data.y[data.test_mask])}
else:
    evals = {'val': (feat(conv_graph_multi(args, data_val)[-1]), data_val.y), 'test': (feat(conv_graph_multi(args, data_test)[-1]), data_test.y)}
y_train = F.one_hot(data.y[data.train_mask], args.num_class).to(Phi.dtype)

gammas = [float(g) for g in args.gammas.split(',')]
print(f'{args.dataset_name}: kernel {args.teacher_kernel}  basis {len(B)}')
for g in gammas:
    W = fit_logistic(Phi[data.train_mask], y_train, g)
    accs = {k: 100 * (P @ W).argmax(1).eq(y).double().mean().item() for k, (P, y) in evals.items()}
    ce = F.cross_entropy(Phi[data.train_mask] @ W, y_train).item()
    print(f"  gamma {g:<8g}  train CE {ce:.4f}  val {accs['val']:.2f}  test {accs['test']:.2f}")
