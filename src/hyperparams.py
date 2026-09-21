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
    parser.add_argument('--gammas', type=str, default='0.0001,0.001,0.01,0.1,1', help='diagnostics: comma list of gamma values')

    args = parser.parse_args()
    return args

BEST_HYPERPARAMS_DICT = {
    #                       kernel   gamma   T     kl    dropout
    ('cora',     0.013):  ('relu',  0.0001, 1.0,  0.2,  '0.9'),
    ('cora',     0.026):  ('relu',  0.01,   1.0,  0.2,  '0.9'),
    ('cora',     0.052):  ('relu',  0.001,  1.0,  1.0,  '0.9'),
    ('citeseer', 0.009):  ('erf',   0.1,    0.2,  0.2,  '0.1'),
    ('citeseer', 0.018):  ('erf',   0.1,    0.2,  0.2,  '0.1'),
    ('citeseer', 0.036):  ('erf',   0.1,    0.02, 0.2,  '0.5'),
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
