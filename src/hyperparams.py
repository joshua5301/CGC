import argparse

def get_hyperparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="reddit", help='cora, citeseer, arxiv, flickr, reddit')
    parser.add_argument('--ratio', type=float, default=0.001)
    parser.add_argument('--raw_data_dir', type=str, default="./data/")
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=10, help='evaluate val/test every k epochs (best-val epoch)')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_dim', type=int, default=256)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropouts', type=str, default=None, help='comma-separated dropout values')
    parser.add_argument('--teacher_kernel', type=str, default=None, help='linear, erf, relu')
    parser.add_argument('--gamma', type=float, default=None, help='teacher regularization coefficient')
    parser.add_argument('--T', type=float, default=None, help='label smoothing/sharpening temperature')
    parser.add_argument('--kl_weight', type=float, default=None, help='weight of label KL loss during clustering')
    parser.add_argument('--basis', type=int, default=3000, help='basis number for teacher kernel')
    parser.add_argument('--edges', type=str, default='none', choices=['none', 'coarsen', 'ot_1hop', 'structure_identity', 'structure_median'],
                        help="condensed edges: none (A' = I on A^2 X), coarsen (A' = cell-averaged propagation matrix on A X), "
                             "ot_1hop (1-hop neighbourhood-OT objective on raw X: representatives, row-stochastic A' and partition by block descent)")
    parser.add_argument('--root_weight', type=float, default=1.0, help='ot_1hop: alpha (root distance)')
    parser.add_argument('--neighbor_weight', type=float, default=1.0, help='ot_1hop: beta (neighbourhood W1)')
    parser.add_argument('--label_weight', type=float, default=1.0, help='ot_1hop: mu (label KL)')
    parser.add_argument('--outer_iters', type=int, default=5, help='ot_1hop: outer block-descent iterations')
    parser.add_argument('--struct_student', type=str, default='gcn', choices=['gcn', 'faithful'],
                        help='structure_*: the condensed graph is the objective\'s own (raw-X medians, Q as given); gcn = main-table GCN served on (X, A_hat), '
                             'faithful = bias-free mean-aggregation student served on (X, P) [the bound applies]')
    parser.add_argument('--struct_init', type=str, default='kmeans', choices=['kmeans', 'grip'],
                        help='structure_*: initial partition -- k-means on raw X (default; the GRIP partition and kl_weight are not used) or the GRIP partition on A^2 X')
    parser.add_argument('--struct_coef', type=str, default='surrogate', choices=['surrogate', 'bound'],
                        help='structure_*: surrogate = alpha/beta/mu from --root_weight/--neighbor_weight/--label_weight; '
                             'bound = alpha, beta from the explicit constants (c_P, K=2, --struct_abar, R = max ||x||, m), mu = 1')
    parser.add_argument('--struct_abar', type=float, default=1.0, help='structure_* bound mode: common layer bound Lip(sigma) ||W||_2')
    parser.add_argument('--ot_level', type=str, default='raw', choices=['raw', 'prop1'],
                        help='ot_1hop: structural features, raw X or the one-hop mean P X (so that the student sees two hops, as the A\' = I path does)')
    parser.add_argument('--ot_solver', type=str, default='sinkhorn', choices=['exact', 'sinkhorn'],
                        help='ot_1hop: exact (LP / network simplex, CPU reference) or sinkhorn (entropic, GPU)')
    parser.add_argument('--ot_eps', type=float, default=0.02, help='ot_1hop sinkhorn: entropic scale, relative to the mean transport cost')
    parser.add_argument('--sink_iters', type=int, default=100, help='ot_1hop sinkhorn: Sinkhorn / IBP iterations')
    parser.add_argument('--candidates', type=int, default=16, help='ot_1hop sinkhorn: cells scored per node in the assignment block')
    parser.add_argument('--max_lp_variables', type=int, default=2_000_000, help='ot_1hop: guard on the largest cell LP')
    parser.add_argument('--gammas', type=str, default='0.0001,0.001,0.01,0.1,1', help='diagnostics: comma list of gamma values')

    args = parser.parse_args()
    return args

BEST_HYPERPARAMS_DICT = {
    # val-selected per density from the shared grid: gamma {0.01, 0.1, 1}, T {0.2, 0.5, 1, 2}, kl {0.1, 0.2, 0.5, 1, 2}; dropout {0.1, 0.5, 0.9}
    #                       kernel   gamma   T     kl    dropout
    ('cora',     0.013):  ('relu',  0.01,   1.0,  2.0,  '0.9'),
    ('cora',     0.026):  ('relu',  0.01,   1.0,  0.2,  '0.9'),
    ('cora',     0.052):  ('relu',  0.01,   1.0,  1.0,  '0.9'),
    ('citeseer', 0.009):  ('erf',   0.1,    0.2,  0.2,  '0.1'),
    ('citeseer', 0.018):  ('erf',   0.1,    0.2,  0.2,  '0.1'),
    ('citeseer', 0.036):  ('erf',   0.1,    0.2,  0.1,  '0.5'),
    ('arxiv',    0.0005): ('relu',  0.01,   0.2,  0.5,  '0.5'),
    ('arxiv',    0.0025): ('relu',  0.01,   0.2,  0.2,  '0.5'),
    ('arxiv',    0.005):  ('relu',  0.01,   0.2,  0.5,  '0.5'),
    ('flickr',   0.001):  ('relu',  0.1,    0.5,  0.1,  '0.5'),
    ('flickr',   0.005):  ('relu',  1.0,    2.0,  0.1,  '0.1'),
    ('flickr',   0.01):   ('relu',  1.0,    1.0,  0.5,  '0.1'),
    ('reddit',   0.0005): ('erf',   0.1,    2.0,  0.5,  '0.1'),
    ('reddit',   0.001):  ('erf',   0.1,    2.0,  2.0,  '0.1'),
    ('reddit',   0.002):  ('erf',   0.01,   2.0,  1.0,  '0.1'),
}

def override_to_best_hyperparams(args):
    best = BEST_HYPERPARAMS_DICT[(args.dataset_name, args.ratio)]
    for name, value in zip(['teacher_kernel', 'gamma', 'T', 'kl_weight', 'dropouts'], best):
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args
