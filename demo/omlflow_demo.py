import optuna
import time
import numpy as np
from mltb.omlflow import OptunaMLflow


@OptunaMLflow()
def objective(omlflow):
    results = []
    for i in range(3):
        x = omlflow.suggest_uniform("x", -10, 10)
        result = (x - 2) ** 2
        omlflow.log_iter({"x": result}, i)
        results.append(result)
    result = np.mean(results)
    time.sleep(2)
    return result


study = optuna.create_study(study_name="_test_mlflow_context_namager_01")
study.optimize(objective, n_trials=5)
