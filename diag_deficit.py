from src.hyperparams import *
from src.utils import *
from src.module import *
from src.dataloader import *
from src.teacher import get_kernel_features, fit_logistic

args = get_hyperparams()
args = device_setting(args)
seed_everything(args.seed)
args = override_to_best_hyperparams(args)

datasets = get_dataset(args)
args, data, data_val, data_test = set_dataset(args, datasets)
adj = normalize_adj_sparse(data).to(data.x.device).double()

src, dst = data.edge_index
homophily = (data.y[src] == data.y[dst]).double().mean().item()

def deficit(Phi, hops):
    M = Phi
    for _ in range(hops):
        M = torch.sparse.mm(adj, M)
    total = ((Phi - Phi.mean(0)) ** 2).sum(1).mean()
    return (((Phi - M) ** 2).sum(1).mean() / total).item()

def prediction_deficit(Phi, hops):
    y_train = F.one_hot(data.y[data.train_mask], args.num_class).to(Phi.dtype)
    W = fit_logistic(Phi[data.train_mask], y_train, args.gamma)
    M = Phi
    for _ in range(hops):
        M = torch.sparse.mm(adj, M)
    p_self, p_nb = F.softmax(Phi @ W, dim=1), F.softmax(M @ W, dim=1)
    disagree = (p_self.argmax(1) != p_nb.argmax(1)).double().mean().item()
    kl = (p_nb * (p_nb.clamp(min=1e-12).log() - p_self.clamp(min=1e-12).log())).sum(1).mean().item()
    acc_self = (p_self.argmax(1) == data.y).double().mean().item()
    acc_nb = (p_nb.argmax(1) == data.y).double().mean().item()
    return disagree, kl, acc_self, acc_nb

X = data.x.double()
Phi = get_kernel_features(X, args.teacher_kernel, args.basis).to(X.dtype)
print(f'{args.dataset_name}: nodes {len(X)}  edge homophily {homophily:.3f}  kernel {args.teacher_kernel} gamma {args.gamma:g}')
for hops in [1, 2]:
    dis, kl, a_self, a_nb = prediction_deficit(Phi, hops)
    print(f'  hops {hops}:  feature deficit  kernel {deficit(Phi, hops):.3f}  raw {deficit(X, hops):.3f}   |   '
          f'prediction deficit  argmax disagree {100 * dis:.1f}%  KL(nb||self) {kl:.3f}   acc self {100 * a_self:.1f}%  nb {100 * a_nb:.1f}%')
