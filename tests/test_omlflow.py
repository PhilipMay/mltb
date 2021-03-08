import optuna
import numpy as np
from mlflow.tracking import MlflowClient

from mltb.omlflow import OptunaMLflow


def _objective_func(trial, tracking_uri):
    with OptunaMLflow(trial, tracking_uri) as om:
        x = trial.suggest_uniform("x", -10, 10)
        om.log_param("x", x)
        results = []

        # do 3 folds
        for i in range(3):
            result = (x - 2) ** 2
            om.log_iter({"x": result}, i)
            results.append(result)

        result = np.mean(results)
        om.log_metric("result", result)
        return result


def test_study_name(tmpdir):
    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    n_trials = 2

    study = optuna.create_study(study_name=study_name)
    study.optimize(_objective_func, n_trials=n_trials)

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)
    assert len(run_infos) == n_trials
