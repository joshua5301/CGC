import argparse
import sys

def para():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="reddit", help= 
    'cora, citeseer, pubmed, arxiv, flickr, reddit, products')
    parser.add_argument('--ratio', type=float, default= 0.001)
    # cora 0.026 citeseer 0.018  arxiv 0.0025 flickr 0.005  reddit 0.001
    parser.add_argument('--raw_data_dir', type=str, default="./data/")
    parser.add_argument('--result_path', type=str, default="./results")
    parser.add_argument('--cond_folder', type=str, default="./cond_graph/")
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_dim', type=int, default=256)
    parser.add_argument('--test_gnn', type=str, default="gcn",
                        help='gcn, sage, gat, cheby, appnp, or comma list / "all"')
    parser.add_argument('--test_gnn_idx', type=int, default=0)

    parser.add_argument('--kernel', type=str, default="gcn", help='gcn, ppr, heat, cheby, sage')
    parser.add_argument('--conv_depth', type=int, default=2, help= 'number of conv depth of the original graph')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--clustering', type=str, default='spectral', help='spectral, kmeans')
    parser.add_argument('--generate_adj', type=int, default=1, help='generate the condensed graph')
    parser.add_argument('--adj_T', type=float, default=0.95, help='threshold for condensed graph')
    parser.add_argument('--alpha', type=float, default=3, help='weight for smooth loss')
    parser.add_argument('--tau', type=float, default=10, help='the denoising ratio')
    parser.add_argument('--aug_ratio', type=float, default=0.55, help='the augmentation ratio')

    parser.add_argument('--landmark', type=str, default='cgc',
                        help='cgc, class_kmeans, kmeans, random, random_split, easy, hard')
    parser.add_argument('--cand_mult', type=float, default=4.0,
                        help='candidate over-generation factor for --landmark easy/hard')
    parser.add_argument('--label_mode', type=str, default='onehot', help='onehot, closed, logistic, logistic_mean, probe, probe_mean, ridge, ridge_mean, restricted, weighted, gcn_mean (full-graph GCN teacher), mlp_mean (MLP-on-H teacher), cs, cs_loss')
    parser.add_argument('--ce_steps', type=int, default=200)
    parser.add_argument('--tangent', type=float, default=0.0,
                        help='first-order condensation: downstream perturbs each node along its '
                             'cell principal axis by N(0,(tangent*sigma)^2); 0 = off')
    parser.add_argument('--cell_k', type=int, default=0,
                        help='overlapping cells: average features/labels over the K nearest pool '
                             'nodes of each centre instead of the Voronoi cell (0 = off)')
    parser.add_argument('--feat_sub', type=int, default=0,
                        help='1 = project condensed features onto the linear teacher subspace '
                             'span(W) (rank c); stores c coords/node + shared basis')
    parser.add_argument('--self_rounds', type=int, default=0,
                        help='iterated condensation: after each round the ensemble of trained students '
                             'becomes the teacher and the cell-mean labels are recomputed')
    parser.add_argument('--self_ratios', type=str, default='',
                        help='comma list of ratios for rounds 1..R: re-condense at each ratio with the '
                             'previous students as teacher (shrinking schedule); overrides --self_rounds')
    parser.add_argument('--self_consistent', type=int, default=0,
                        help='1 = distil the student ensemble into an H-space probe before cell-averaging '
                             '(keeps the teacher a function of what the next student sees)')
    parser.add_argument('--self_full', type=int, default=0,
                        help='control for --self_rounds: retrain on the FULL graph with the soft pseudo-labels '
                             '(no condensation) instead of on the condensed set')
    parser.add_argument('--gen_static', type=int, default=0,
                        help='materialise K teacher-labelled draws x_j + s_j*eps per cell as static '
                             'nodes (centroids dropped); 0 = off')
    parser.add_argument('--gen', type=float, default=0.0,
                        help='parametric condensation: downstream samples x ~ N(x_j, (gen*s_j)^2) per '
                             'epoch and labels it with the condensed-set teacher; 0 = off')
    parser.add_argument('--tan_frac', type=float, default=1.0,
                        help='keep tangent info only for the top fraction of cells by sigma*|g|')
    parser.add_argument('--tan_rank', type=int, default=0,
                        help='project cell directions onto a shared rank-r basis (0 = full)')
    parser.add_argument('--tan_code', type=int, default=0,
                        help='codebook of K shared directions; each cell stores one index (0 = off)')
    parser.add_argument('--tan_label', type=int, default=1,
                        help='tangent: 1 = move label with g_j (condition C), 0 = input only (B)')
    parser.add_argument('--tan_static', type=int, default=0,
                        help='1 = centre + two endpoints x_j +- sigma_j u_j as nodes, 2 = endpoints only')
    parser.add_argument('--tan_static_edge', type=int, default=0,
                        help='1 = also connect each endpoint to its centre (star)')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='downstream mixup Beta(a,a) on the condensed set; 0 = off')
    parser.add_argument('--lam_p', type=float, default=0.0,
                        help='weight of the teacher posterior block when clustering; '
                             '0 = feature space only (default), large = posterior space')
    parser.add_argument('--avg_pool', type=str, default='all',
                        help='label averaging set: all, or unlabeled (out-of-sample only)')
    parser.add_argument('--expert_basis', type=int, default=0,
                        help='logistic_mean: kernel basis size, 0 = use the landmarks')
    parser.add_argument('--w_steps', type=int, default=300, help='weighted: Adam steps')
    parser.add_argument('--w_lr', type=float, default=0.05, help='weighted: Adam lr')
    parser.add_argument('--w_mu', type=float, default=0.0,
                        help='weighted: trust-region pull toward uniform weights')
    parser.add_argument('--w_target', type=str, default='both',
                        help='weighted: optimise both, feat only, or label only')
    parser.add_argument('--w_batch', type=int, default=20000,
                        help='weighted: labelled nodes per step (0 = all)')
    parser.add_argument('--label_feat', type=str, default='last', help='first (raw X), last, mean, concat')
    parser.add_argument('--target_maxp', type=float, default=0.0,
                        help='>0 auto-selects gamma by bisection to hit this target sharpness')
    parser.add_argument('--maxp_iters', type=int, default=6)
    parser.add_argument('--cs_a1', type=float, default=0.8)
    parser.add_argument('--cs_a2', type=float, default=0.8)
    parser.add_argument('--cs_iters', type=int, default=50)
    parser.add_argument('--cs_folds', type=int, default=2,
                        help='cs_loss: train split; C&S injects train-minus-f, loss scored on f')
    parser.add_argument('--cs_reset', type=int, default=1,
                        help='0 disables the Z[train]=Y_L reset (needed for inductive splits)')
    parser.add_argument('--cs_scale', type=float, default=1.0)
    parser.add_argument('--teacher', type=str, default='none', help='none, probe, ridge')
    parser.add_argument('--teacher_temp', type=float, default=1.0)
    parser.add_argument('--teacher_gamma', type=float, default=1e-2)
    parser.add_argument('--teacher_folds', type=int, default=0,
                        help='>1 uses an out-of-fold teacher for train rows')
    parser.add_argument('--distill_pool', type=str, default='train', help='train, all')
    parser.add_argument('--label_prior', type=str, default='none', help='none, cluster')
    parser.add_argument('--no_hyperpara', type=int, default=0,
                        help='1 skips the per-dataset hyperpara table entirely')
    parser.add_argument('--preset', type=str, default='', help="'unified' sets the P-based pipeline")
    parser.add_argument('--adj_mode', type=str, default='cosine',
                        help='cosine, coarsen, commute, tangent')
    parser.add_argument('--tan_edge_k', type=int, default=2,
                        help='tangent adj: max neighbours per direction')
    parser.add_argument('--tan_edge_T', type=float, default=1.0,
                        help='tangent adj: keep neighbour if within T*sigma_j of the query point')
    parser.add_argument('--relabel', type=int, default=0,
                        help='1 = re-evaluate teacher at the A-propagated position (cell mean kept)')
    parser.add_argument('--adj_steps', type=int, default=300)
    parser.add_argument('--adj_lr', type=float, default=0.05)
    parser.add_argument('--adj_l1', type=float, default=0.0)
    parser.add_argument('--adj_init', type=str, default='zeros', help='zeros, coarsen')
    parser.add_argument('--cond_feat', type=str, default='solve', help='solve, prop, raw')
    parser.add_argument('--whiten', type=float, default=0.0, help='0=none, 1=full; clustering metric only')
    parser.add_argument('--label_kernel', type=str, default='linear', help='linear, erf, arccos, rbf')
    parser.add_argument('--h_pool', type=str, default='train', help='train, all, train_unlabeled')
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=1e-2)
    parser.add_argument('--constraint', type=str, default='none', help='none, balance, row, simplex')
    parser.add_argument('--head', type=str, default='mse', help='ce, mse, mse_logit')

    args = parser.parse_args()
    args.explicit = {a.split('=')[0][2:].replace('-', '_') for a in sys.argv[1:] if a.startswith('--')}
    if args.preset == 'unified':
        args.landmark, args.h_pool = 'kmeans', 'all'
        args.generate_adj, args.adj_mode, args.cond_feat = 1, 'commute', 'raw'
        args.label_mode, args.head = 'probe', 'ce'
    return args



def apply_hyperpara(args):
    if args.no_hyperpara:
        return args
    keep = {k: getattr(args, k) for k in getattr(args, 'explicit', set()) if hasattr(args, k)}
    args = hyperpara(args) if args.generate_adj == 1 else hyperpara_noadj(args)
    for k, v in keep.items():
        setattr(args, k, v)
    return args


def hyperpara(args):
    if args.dataset_name == 'cora':
        
        if args.ratio == 0.013:
            args.epoch=1200
            args.dropout = 0.8
            args.adj_T = 0.8
            args.alpha = 1
            args.tau = 0.5
            args.aug_ratio = 0.5

        if args.ratio == 0.026:
            args.dropout = 0.8
            args.adj_T = 0.8
            args.alpha = 7
            args.tau = 1        
            args.aug_ratio = 0.5

        if args.ratio == 0.052:
            args.dropout = 0.8  
            args.adj_T = 0.9    
            args.alpha = 7      
            args.tau = 0.1  
            args.aug_ratio = 0.3



    if args.dataset_name == 'citeseer':

        if args.ratio == 0.009:
            args.adj_T = 0.8
            args.alpha = 2
            args.tau = 0.05 
            args.aug_ratio = 0.6

        if args.ratio == 0.018:
            args.adj_T = 0.8
            args.alpha = 1
            args.tau = 0.05 
            args.aug_ratio = 0.1

        if args.ratio == 0.036:
            args.adj_T = 0.75
            args.alpha = 2.
            args.tau = 0.5      
            args.aug_ratio = 0.5


    if args.dataset_name == 'arxiv':
        
        if args.ratio == 0.0005:
            args.adj_T = 0.92
            args.alpha = 3
            args.tau = 5
            args.aug_ratio = 0.4

        if args.ratio == 0.0025:
            args.weight_decay=5e-3
            args.epoch = 1000
            args.adj_T = 0.92
            args.alpha = 3
            args.tau = 15
            args.aug_ratio = 0.4

        if args.ratio == 0.005:
            args.weight_decay=5e-3
            args.lr = 0.01
            args.dropout = 0.4
            args.adj_T = 0.92
            args.alpha = 3
            args.tau = 1
            args.aug_ratio = 3

    if args.dataset_name == 'flickr':

        if args.ratio == 0.001:
            args.lr=0.001
            args.epoch = 1000
            args.dropout = 0.47
            args.adj_T = 0.996
            args.alpha = 1
            args.tau = 0.1
            args.aug_ratio = 0.5

        if args.ratio == 0.005:
            args.dropout = 0.6
            args.adj_T = 0.996
            args.alpha = 1
            args.tau = 0.08
            args.aug_ratio = 0.3

        if args.ratio == 0.01:
            args.dropout = 0.9
            args.adj_T = 0.996
            args.alpha = 1
            args.tau = 0.01
            args.aug_ratio = 0.5

    if args.dataset_name == 'reddit':

        args.epoch = 1000
        if args.ratio == 0.0005:
            args.lr = 0.001
            args.dropout = 0.1
            args.adj_T = 0.95
            args.alpha = 3
            args.tau = 10
            args.aug_ratio = 0.2

        if args.ratio == 0.001:
            args.lr = 0.001
            args.dropout = 0.1
            args.adj_T = 0.95
            args.alpha = 3
            args.tau = 10
            args.aug_ratio = 0.55

        if args.ratio == 0.002:
            args.lr = 0.001
            args.dropout = 0.1
            args.adj_T = 0.95
            args.alpha = 3
            args.tau = 10
            args.aug_ratio = 0.75

    return args


def hyperpara_noadj(args):
    if args.dataset_name == 'cora':

        if args.ratio == 0.013:
            args.dropout = 0.8
            args.tau = 0.7
            args.aug_ratio = 0.3

        if args.ratio == 0.026:
            args.dropout = 0.8
            args.tau = 10
            args.aug_ratio = 0.7

        if args.ratio == 0.052:
            args.dropout = 0.9      
            args.tau = 0.7
            args.aug_ratio = 0.3

    if args.dataset_name == 'citeseer':

        if args.ratio == 0.009:
            args.tau = 0.1 
            args.aug_ratio = 0.6

        if args.ratio == 0.018:
            args.lr = 0.02
            args.dropout = 0.0
            args.tau = 0.1
            args.aug_ratio = 0.8

        if args.ratio == 0.036:  
            args.weight_decay=0.08  
            args.tau = 10     
            args.aug_ratio = 0.7

    if args.dataset_name == 'arxiv':

        if args.ratio == 0.0005:
            args.epoch = 1000
            args.dropout = 0.7  
            args.tau = 0.3
            args.aug_ratio = 0.7

        if args.ratio == 0.0025:
            args.tau = 0.3
            args.aug_ratio = 0.4

        if args.ratio == 0.005:
            args.tau = 10
            args.aug_ratio = 0.8

    if args.dataset_name == 'flickr':

        if args.ratio == 0.001:
            args.dropout = 0.64 
            args.tau = 0.1 
            args.aug_ratio = 0.4

        if args.ratio == 0.005:
            args.dropout = 0.7
            args.tau = 0.08 
            args.aug_ratio = 0.5

        if args.ratio == 0.01:
            args.dropout = 0.9
            args.tau = 0.005
            args.aug_ratio = 0.6



    if args.dataset_name == 'reddit':

        if args.ratio == 0.0005:
            args.lr = 0.001
            args.dropout = 0.1
            args.tau = 10
            args.aug_ratio = 0.8

        if args.ratio == 0.001:
            args.lr = 0.001
            args.dropout = 0.1
            args.tau = 10
            args.aug_ratio = 0.2

        if args.ratio == 0.002:
            args.lr = 0.001
            args.dropout = 0.1
            args.tau = 10
            args.aug_ratio = 0.2


    return args
