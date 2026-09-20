import argparse

def para():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="reddit", help= 
    'cora, citeseer, arxiv, flickr, reddit')
    parser.add_argument('--ratio', type=float, default= 0.001)
    # cora 0.026 citeseer 0.018  arxiv 0.0025 flickr 0.005  reddit 0.001
    parser.add_argument('--raw_data_dir', type=str, default="./data/")
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=10, help='evaluate val/test every k epochs (best-val epoch)')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_dim', type=int, default=256)

    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--teacher_kernel', type=str, default="erf", help='linear, erf, relu')
    parser.add_argument('--gamma', type=float, default=0.1, help='teacher regularization coefficient')
    parser.add_argument('--T', type=float, default=1.0, help='label smoothing/sharpening temperature')
    parser.add_argument('--kl_weight', type=float, default=0.5, help='weight of label KL loss during clustering')
    parser.add_argument('--basis', type=int, default=3000, help='Nystrom inducing points of the teacher (random pool rows)')

    args = parser.parse_args()
    return args

HYPER = {
    ('cora',     0.013):  ('relu',  0.1,    1.0,  1.0,  0.9),
    ('cora',     0.026):  ('relu',  0.01,   2.0,  0.2,  0.9),
    ('cora',     0.052):  ('relu',  0.01,   1.0,  2.0,  0.9),
    ('citeseer', 0.009):  ('erf',   10.0,   0.25, 0.2,  0.3),
    ('citeseer', 0.018):  ('erf',   10.0,   0.25, 0.2,  0.1),
    ('citeseer', 0.036):  ('erf',   3.0,    0.25, 0.2,  0.3),
    ('arxiv',    0.0005): ('erf',   1e-3,   0.25, 0.5,  0.7),
    ('arxiv',    0.0025): ('relu',  1e-4,   0.25, 0.2,  0.3),
    ('arxiv',    0.005):  ('erf',   1e-3,   1.0,  0.5,  0.1),
    ('flickr',   0.001):  ('relu',  0.01,   1.0,  0.3,  0.3),
    ('flickr',   0.005):  ('relu',  0.1,    1.0,  0.3,  0.0),
    ('flickr',   0.01):   ('relu',  0.1,    1.0,  0.3,  0.0),
    ('reddit',   0.0005): ('erf',   1e-3,   1.0,  0.3,  0.1),
    ('reddit',   0.001):  ('erf',   1e-4,   1.0,  1.0,  0.3),
    ('reddit',   0.002):  ('erf',   0.01,   1.0,  1.0,  0.1),
}

def hyperpara(args):
    key = (args.dataset_name, args.ratio)
    args.teacher_kernel, args.gamma, args.T, args.kl_weight, args.dropout = HYPER[key]
    return args
