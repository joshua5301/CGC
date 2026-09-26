import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.teacher import fit_logistic, get_kernel_features


def fit_gcn_probe_teacher(x, edge_index, train_ids, train_labels, val_ids, val_labels,
                          hidden=256, dropout=.5, lr=.01, weight_decay=5e-4,
                          epochs=1000, eval_every=10, seed=0, device='cuda'):
    from src.gnn_distance_probe import ProbeGNN

    if np.intersect1d(train_ids, val_ids).size or not len(train_ids) or not len(val_ids):
        raise ValueError('Require disjoint nonempty training and validation sets')
    if epochs < 1 or eval_every < 1:
        raise ValueError('Require positive epochs and evaluation interval')
    classes = np.unique(train_labels)
    if not np.isin(val_labels, classes).all():
        raise ValueError('Validation contains classes absent from training')
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    train = torch.as_tensor(train_ids, dtype=torch.long, device=device)
    val = torch.as_tensor(val_ids, dtype=torch.long, device=device)
    labels = torch.as_tensor(np.searchsorted(classes, train_labels), dtype=torch.long, device=device)
    validation = torch.as_tensor(np.searchsorted(classes, val_labels), dtype=torch.long, device=device)
    network = ProbeGNN('gcn', x.shape[1], hidden, len(classes), dropout).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    rows, best_key = [], (-1., -float('inf'))
    for epoch in tqdm(range(1, epochs + 1), desc='GCN teacher training'):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(network(x, edges)[train], labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite GCN teacher loss')
        loss.backward()
        optimizer.step()
        if epoch % eval_every and epoch != epochs:
            continue
        network.eval()
        with torch.no_grad():
            logits = network(x, edges)
            accuracy = float((logits[val].argmax(1) == validation).double().mean())
            ce = float(F.cross_entropy(logits[val], validation))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite GCN teacher logits')
        rows.append(dict(epoch=epoch, val_accuracy=100 * accuracy, val_ce=ce))
        if (accuracy, -ce) > best_key:
            best_key, best_epoch = (accuracy, -ce), epoch
            best_logits = logits.detach().cpu().clone()
            best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
    config = dict(model='gcn', layers=2, hidden=hidden, dropout=dropout, lr=lr,
                  weight_decay=weight_decay, epochs=epochs, eval_every=eval_every,
                  seed=seed, T=1., validation_selection=True,
                  selection='highest_validation_accuracy_then_lowest_ce_then_earliest_epoch',
                  best_epoch=best_epoch, val_accuracy=100 * best_key[0], val_ce=-best_key[1],
                  classes=classes.tolist(), training_label_override=False,
                  training='uniform_ce_on_original_training_labels', features='raw')
    return dict(probabilities=best_logits.softmax(1).numpy(), logits=best_logits.numpy(),
                sweep=pd.DataFrame(rows), config=config, state_dict=best_state)


def tune_probe_teacher(features, train_ids, train_labels, val_ids, val_labels,
                       gammas, kernel='relu', basis=3000, seed=0, device='cuda'):
    gammas = sorted(set(float(g) for g in gammas))
    if not gammas or any(not np.isfinite(g) or g < 0 for g in gammas):
        raise ValueError('Require finite nonnegative gamma candidates')
    if np.intersect1d(train_ids, val_ids).size or not len(train_ids) or not len(val_ids):
        raise ValueError('Require disjoint nonempty training and validation sets')
    classes = np.unique(train_labels)
    if not np.isin(val_labels, classes).all():
        raise ValueError('Validation contains classes absent from training')
    seed_everything(seed)
    features = torch.as_tensor(features, dtype=torch.float64, device=device)
    phi = get_kernel_features(features, kernel, basis)
    train = torch.as_tensor(train_ids, dtype=torch.long, device=device)
    val = torch.as_tensor(val_ids, dtype=torch.long, device=device)
    labels = F.one_hot(torch.as_tensor(np.searchsorted(classes, train_labels), device=device), len(classes)).double()
    validation = torch.as_tensor(np.searchsorted(classes, val_labels), device=device)
    best_accuracy, best_gamma, best_logits = -1., None, None
    rows = []
    for gamma in tqdm(gammas, desc='Teacher gamma validation'):
        weights = fit_logistic(phi[train], labels, gamma)
        with torch.no_grad():
            logits = phi @ weights
            accuracy = float((logits[val].argmax(1) == validation).double().mean())
            ce = float(F.cross_entropy(logits[val], validation))
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f'Nonfinite teacher logits at gamma={gamma}')
        rows.append(dict(gamma=gamma, val_accuracy=100 * accuracy, val_ce=ce))
        if accuracy > best_accuracy:
            best_accuracy, best_gamma = accuracy, gamma
            best_logits = logits.detach().cpu()
    config = dict(kernel=kernel, basis=basis, seed=seed, gamma=best_gamma,
                  gammas=gammas, T=1., validation_selection=True,
                  selection='highest_validation_accuracy_then_smallest_gamma',
                  val_accuracy=100 * best_accuracy, classes=classes.tolist(),
                  training_label_override=False)
    return dict(probabilities=best_logits.softmax(1).float().numpy(),
                logits=best_logits.numpy(), sweep=pd.DataFrame(rows), config=config)
