import numpy as np
import torch

import src.partition as partition


def test_passes_seeded_centers_to_faiss(monkeypatch):
    received = {}

    class FakeKmeans:
        def __init__(self, *args, **kwargs):
            from types import SimpleNamespace
            self.cp = SimpleNamespace()
            self.index = self

        def train(self, x, init_centroids=None):
            received['centers'] = init_centroids
            received['seed'] = self.cp.seed

        def search(self, x, k):
            return None, np.zeros((len(x), 1), dtype=np.int64)

    monkeypatch.setattr(partition.faiss, 'Kmeans', FakeKmeans)
    x = torch.tensor([[0.], [1.], [4.], [10.]])
    partition.kmeans_init(x, 2, seed=1234, plus_plus=True)
    expected, _ = partition.kmeans_plusplus(x.numpy(), 2, random_state=1234, n_local_trials=1)
    np.testing.assert_array_equal(received['centers'], expected)
    assert received['seed'] == 1234
    partition.kmeans_init(x, 2, seed=0)
    assert received['centers'] is None
    assert received['seed'] == 0
