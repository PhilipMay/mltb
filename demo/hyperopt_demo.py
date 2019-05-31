from mltb.hyperopt import fmin
from hyperopt import tpe, hp, STATUS_OK


def objective(x):
    return {
        'loss': x ** 2,
        'status': STATUS_OK,
        'other_stuff': {'type': None, 'value': [0, 1, 2]},
        }


best, trials = fmin(objective,
    space=hp.uniform('x', -10, 10),
    algo=tpe.suggest,
    max_evals=100,
    filename='trials_file')

print(best)
print(trials.trials)
