import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.teacher import fit_logistic, get_kernel_features


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
