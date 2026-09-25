import json
import math
from itertools import product
from types import SimpleNamespace

import pandas as pd
from tqdm.auto import tqdm


class GridTrial:
    def __init__(self, number, params):
        self.number, self.params, self.user_attrs = number, params, {}

    def suggest_categorical(self, name, choices):
        return self.params[name]

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class GridStudy:
    def __init__(self, space, directory):
        self.space, self.directory = space, directory / 'grid'
        self.directory.mkdir(exist_ok=True)
        self.trials = [SimpleNamespace(**json.loads(p.read_text(encoding='utf-8')))
                       for p in sorted(self.directory.glob('*.json'))]
        self.trials.sort(key=lambda t: t.number)

    @property
    def best_trial(self):
        return max(self.trials, key=lambda t: (t.value, -t.number))

    def optimize(self, objective):
        completed = {t.number for t in self.trials}
        total = math.prod(len(v) for v in self.space.values())
        with tqdm(total=total, initial=len(completed), desc='Grid validation') as progress:
            for number, values in enumerate(product(*self.space.values())):
                if number in completed:
                    continue
                trial = GridTrial(number, dict(zip(self.space, values)))
                value = float(objective(trial))
                if not math.isfinite(value):
                    raise FloatingPointError('Nonfinite grid validation score')
                record = dict(number=number, params=trial.params,
                              user_attrs=trial.user_attrs, value=value)
                path = self.directory / f'{number:08d}.json'
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(record), encoding='utf-8')
                temporary.replace(path)
                self.trials.append(SimpleNamespace(**record))
                progress.update()
                progress.set_postfix(best_val=self.best_trial.value)

    def trials_dataframe(self):
        return pd.DataFrame([dict(number=t.number, value=t.value, state='COMPLETE',
                                  **{f'params_{k}': v for k, v in t.params.items()},
                                  **{f'user_attrs_{k}': v for k, v in t.user_attrs.items()})
                             for t in self.trials])
