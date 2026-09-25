import numpy as np
import torch

from src.grip_confusion_study import confusion_metrics
import src.risk_experiment as experiment


def test_confusion_orientation_and_contributions():
    matrix = np.array([[2, 1, 0, 0], [3, 4, 0, 0], [0, 0, 5, 0], [2, 0, 0, 6]])
    stats, classes = confusion_metrics(matrix)
    assert stats['error_1_to_0'] == 3
    assert stats['error_3_to_0'] == 2
    assert stats['error_13_to_0'] == 5
    np.testing.assert_allclose(stats['class0_precision'], 200 / 7)
    np.testing.assert_allclose(classes.accuracy_contribution.sum(), stats['accuracy'])


def test_returned_model_is_validation_best(monkeypatch):
    states = []
    monkeypatch.setattr(experiment, 'GCN', lambda nin, hidden, nout, layers, dropout: torch.nn.Linear(nin, nout))
    monkeypatch.setattr(experiment, '_forward', lambda model, x: model(x).log_softmax(1))

    def accuracy(model, evaluation):
        states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return .8 if len(states) == 1 else .4

    monkeypatch.setattr(experiment, '_accuracy', accuracy)
    val, test, epoch, model = experiment._train_student(torch.eye(2), torch.eye(2), None,
        dict(dropout=.5, lr=.01, weight_decay=0.), 0,
        dict(epochs=2, eval_every=1, hidden=4, loss_weighting='uniform'), return_model=True)
    assert val == .8 and test is None and epoch == 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, states[0][key], rtol=0, atol=0)
