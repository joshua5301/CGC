import numpy as np
import scipy.sparse as sp
import pytest

from src.partition_hybrid import HybridIdentity, cut_cost, move_cut_deltas


def toy():
    rng = np.random.default_rng(7)
    P = rng.random((19, 19))
    P[P < .8] = 0
    P[0] = 0  # isolated node with a self transition
    np.fill_diagonal(P, 1.)
    P /= P.sum(1, keepdims=True)
    H = rng.normal(size=(19, 4))
    F = rng.random((19, 3))+.1
    F /= F.sum(1, keepdims=True)
    a = np.arange(19) % 4
    C = np.stack([H[a == j].mean(0) for j in range(4)])
    Y = np.stack([F[a == j].mean(0) for j in range(4)])
    return H, F, sp.csr_matrix(P), a, C, Y


def test_cut_matches_full_ps_minus_s_and_every_move():
    H, F, P, a, C, Y = toy()
    S = np.eye(len(C))[a]
    assert np.isclose(cut_cost(P, a), abs(P@S-S).sum()/2)
    initial = cut_cost(P, a)
    for node in range(len(a)):
        delta = move_cut_deltas(P, P.T.tocsr(), a, node, len(C))
        for dest in range(len(C)):
            trial = a.copy()
            trial[node] = dest
            assert np.isclose(delta[dest], cut_cost(P, trial)-initial, atol=1e-12)
    assert np.all(move_cut_deltas(P, P.T.tocsr(), a, 0, len(C)) <= 1.)


@pytest.mark.parametrize('lam', [0., .01, 1., 100.])
def test_descent_preserves_budget_and_identity_objective(lam):
    H, F, P, a, C, Y = toy()
    model = HybridIdentity(H, F, P, a, C, Y, 2., .3, lam, .5)
    result = model.run(8, log=lambda x: None)
    history = result['history']
    assert np.all(np.diff([r['J'] for r in history]) <= 1e-9)
    assert (result['counts'] > 0).all()
    assert result['counts'].sum() == len(H)
    assert np.allclose(result['y'].sum(1), 1.)
    for row in history:
        assert np.isclose(row['J'], row['feature']/2+.5*row['kl']/.3+lam*row['structure'])
    S = np.eye(len(C))[result['assign']]
    assert np.isclose(history[-1]['residual_l1'], abs(P@S-S).sum()/len(H))


def test_generic_bounded_message_recursion_with_propagated_initial_error():
    # Nonlinear message BEFORE aggregation; not restricted to linear GCN messages.
    H, F, P, a, C, Y = toy()
    X = H
    Z = P@P@X
    centers = np.stack([Z[a == j].mean(0) for j in range(len(C))])
    S = np.eye(len(C))[a]
    residual = abs(P@S-S).sum(1)
    bound = np.linalg.norm(X-Z, axis=1)+np.linalg.norm(Z-centers[a], axis=1)
    h, hc = X.copy(), centers.copy()
    for _ in range(3):
        # tanh: 1-Lipschitz, each message has Euclidean norm <= sqrt(d).
        h = .3*h + .7*(P@np.tanh(h))
        hc = .3*hc + .7*np.tanh(hc)
        bound = .3*bound+.7*(P@bound)+.7*np.sqrt(X.shape[1])*residual
        assert np.all(np.linalg.norm(h-hc[a], axis=1) <= bound+1e-12)


def test_smoothing_remainder_cannot_be_omitted():
    X = np.array([[-1.], [1.]])
    P = np.ones((2, 2))/2
    S = np.ones((2, 1))
    assert np.allclose(P@X, 0)
    assert np.allclose(P@S-S, 0)
    assert np.allclose(P@np.maximum(X, 0), .5)


def test_selection_ignores_test_and_ties_prefer_baseline():
    from hybrid_grip import choose_variant
    rows = [dict(variant=n, seed=s, val=80., test=t) for n, t in
            [('grip', 1.), ('hybrid_0', 99.), ('hybrid_1', 100.)] for s in range(3)]
    assert choose_variant(rows) == 'grip'
    rows[-1]['val'] = 81.
    assert choose_variant(rows) == 'hybrid_1'
