import optuna
import numpy as np
from mlflow.tracking import MlflowClient

from mltb.omlflow import OptunaMLflow


def _objective_func_factory(tracking_uri, num_folds):
    @OptunaMLflow(tracking_uri=tracking_uri)
    def _objective_func(omlflow):
        x = omlflow.suggest_uniform("x", -10, 10)
        results = []

        # do folds
        for i in range(num_folds):
            result = (x - 2) ** 2
            omlflow.log_iter({"result": result}, i)
            results.append(result)

        result = np.mean(results)
        return result

    return _objective_func


def test_study_name(tmpdir):
    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    n_trials = 2
    num_folds = 3

    study = optuna.create_study(study_name=study_name)
    study.optimize(_objective_func_factory(tracking_file_name, num_folds), n_trials=n_trials)

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)
    assert len(run_infos) == n_trials + n_trials * num_folds

    first_run_id = run_infos[-1].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()
    assert "x" in first_run_dict["data"]["params"]
    assert first_run_dict["data"]["tags"]["direction"] == "MINIMIZE"
