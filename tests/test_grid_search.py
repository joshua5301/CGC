from src.grid_search import GridStudy


def test_grid_exhaustion_and_resume(tmp_path):
    space = {'a': [1, 2], 'b': [10, 20]}
    seen = []

    def objective(trial):
        seen.append(trial.params.copy())
        trial.set_user_attr('config', trial.params)
        return trial.suggest_categorical('a', space['a']) + trial.params['b']

    study = GridStudy(space, tmp_path)
    study.optimize(objective)
    assert seen == [{'a': 1, 'b': 10}, {'a': 1, 'b': 20},
                    {'a': 2, 'b': 10}, {'a': 2, 'b': 20}]
    resumed = GridStudy(space, tmp_path)
    resumed.optimize(objective)
    assert len(seen) == 4
    assert resumed.best_trial.value == 22
    assert resumed.best_trial.user_attrs['config'] == {'a': 2, 'b': 20}


def test_interrupted_combination_is_retried(tmp_path):
    study = GridStudy({'a': [1, 2]}, tmp_path)

    def interrupted(trial):
        if trial.number == 1:
            raise RuntimeError('interrupted')
        return trial.params['a']

    try:
        study.optimize(interrupted)
    except RuntimeError:
        pass
    seen = []
    resumed = GridStudy({'a': [1, 2]}, tmp_path)
    resumed.optimize(lambda trial: seen.append(trial.number) or trial.params['a'])
    assert seen == [1]
    assert resumed.best_trial.value == 2
