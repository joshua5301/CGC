import torch

from src.grip_candidates import candidate_pool, coverage_seeds
from src.partition import kmeans_init, partition


def test_pool_size_reproducibility_and_train_membership():
    mask = torch.tensor([True, False, True, False, True, False])
    assert candidate_pool(mask, 'train').tolist() == [0, 2, 4]
    first = candidate_pool(mask, 'random', 17)
    assert len(first) == 3 and len(first.unique()) == 3
    assert torch.equal(first, candidate_pool(mask, 'random', 17))


def test_coverage_matches_exhaustive_greedy_costs():
    x = torch.tensor([[0.], [1.], [3.], [8.], [10.]], dtype=torch.double)
    candidates = torch.tensor([4, 0, 2])
    selected, history = coverage_seeds(x, candidates, 3, block_size=2)
    current = []
    for step, picked in enumerate(selected.tolist()):
        costs = {}
        for candidate in sorted(candidates.tolist()):
            if candidate not in current:
                costs[candidate] = float(torch.cdist(x, x[current + [candidate]]).square().min(1).values.mean())
        expected = min(costs, key=costs.get)
        assert picked == expected
        assert abs(history[step] - costs[expected]) < 1e-12
        current.append(picked)


def test_cached_assignment_preserves_original_partition():
    x = torch.tensor([[0., 0.], [.1, 0.], [.2, .1], [4., 4.], [4.1, 4.], [4., 4.1]])
    q = torch.tensor([[.9, .1]] * 3 + [[.1, .9]] * 3)
    assignment = kmeans_init(x, 2, seed=1234)
    direct = partition(x, q, 2, return_state=True)
    cached = partition(x, q, 2, return_state=True, initial_assignment=assignment)
    for a, b in zip(direct[:3], cached[:3]):
        assert torch.equal(a, b)
    assert direct[3] == cached[3]
