import numpy as np
import torch

from src.grip_seed_structure import describe_partition
from src.partition import geometric_medians, cell_means, partition_cost


def test_cost_decomposition_and_allocation():
    x = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [8., 1.], [9., 1.]], dtype=torch.float64)
    q = torch.tensor([[.9, .1], [.8, .2], [.7, .3], [.2, .8], [.1, .9]], dtype=torch.float64)
    a = torch.tensor([0, 0, 0, 1, 2])
    centers = geometric_medians(x, a, 3)
    labels = cell_means(q, a, 3)
    ds = float((x - geometric_medians(x, torch.zeros(5, dtype=torch.long), 1)).norm(dim=1).mean())
    ks = float((q * (q.log() - q.mean(0).log())).sum(1).mean())
    value = partition_cost(x, q, a, centers, labels, ds, ks, .2)
    artifact = dict(assignment=a, x=centers.float(), final_J=value['J'], dist_scale=ds, kl_scale=ks)
    stats, cells, classes = describe_partition(x, q, artifact, .2)
    np.testing.assert_allclose(cells.total_J.sum(), value['J'], atol=1e-7)
    np.testing.assert_allclose(classes.total_J.sum(), value['J'], atol=1e-7)
    np.testing.assert_allclose(classes.soft_share_shift.sum(), 0, atol=1e-12)
    assert cells['size'].tolist() == [3, 1, 1]
    assert classes.representative_count.tolist() == [1, 2]
    assert classes.node_count.tolist() == [3, 2]
    assert stats['label_mass_tv'] > 0
    assert stats['size_gini'] > 0
    assert stats['margin_valid_representatives'] == 2
