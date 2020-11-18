import optuna
import time
import numpy as np
from mltb.omlflow import OptunaMLflow


def objective(trial):
    with OptunaMLflow(trial, "", num_name_digits=5) as om:
        results = []
        for i in range(3):
            x = trial.suggest_uniform('x', -10, 10)
            om.log_param('x', x)
            result = (x - 2) ** 2
            om.log_iter(i, {'x': result})
            results.append(result)
        result = np.mean(results)
        om.log_metric('result', result)
        time.sleep(2)
        return result


study = optuna.create_study(study_name='_test_mlflow_context_namager_01')
study.optimize(objective, n_trials=5)