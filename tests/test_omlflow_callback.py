import tempfile
from pathlib import Path
from typing import Dict

import optuna
import pytest
import sklearn.model_selection
import torch
from mlflow.tracking import MlflowClient
from transformers import (
    ElectraConfig,
    ElectraForSequenceClassification,
    ElectraTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from mltb.integration.transformers import OMLflowCallback
from mltb.omlflow import OptunaMLflow

# global parameters for training
MAX_STEPS = 4
# an eval step is only logged if it coincides with a logging step
# logging steps should divide eval steps otherwise some evals are lost
LOGGING_STEPS = EVAL_STEPS = 2
assert EVAL_STEPS % LOGGING_STEPS == 0, f"Logging steps {LOGGING_STEPS} should divide eval steps {EVAL_STEPS}"
N_EVAL_STEPS = MAX_STEPS // EVAL_STEPS
LAST_EVAL_STEP = N_EVAL_STEPS * EVAL_STEPS
N_TRIALS = 2


@pytest.fixture(scope="module")
def config():
    return ElectraConfig()


@pytest.fixture(scope="module")
def model(config):
    return ElectraForSequenceClassification(config)


@pytest.fixture(scope="module")
def output_path():
    tempdir = tempfile.mkdtemp()
    return Path(tempdir)


@pytest.fixture(scope="module")
def tracking_file_name(output_path):
    return f"file:{output_path.as_posix()}"


def test_omlflow_callback(config, output_path, model, tracking_file_name):
    study_name = "test_omlflow_callback"
    study = optuna.create_study(study_name=study_name)
    train_func = _train_func_factory(config, output_path, model, tracking_file_name)
    study.optimize(train_func, n_trials=N_TRIALS)

    mlflow_client = MlflowClient(tracking_file_name)

    # test that there is 1 MLflow experiment, corresponding to the Optuna trial
    experiments = mlflow_client.list_experiments()
    assert len(experiments) == 1
    experiment = experiments[0]
    assert experiment.name == study_name

    # each trial should be recorded an an MLflow run
    experiment_id = experiment.experiment_id
    run_infos = mlflow_client.list_run_infos(experiment_id)
    assert len(run_infos) == N_TRIALS

    # test that the callback correctly recorded the target metric for each eval step
    run_info = run_infos[0]
    metric_history = mlflow_client.get_metric_history(run_id=run_info.run_id, key="eval_acc")
    assert len(metric_history) == N_EVAL_STEPS
    eval_steps = range(EVAL_STEPS, LAST_EVAL_STEP + EVAL_STEPS, EVAL_STEPS)
    for step, metric_snapshot in zip(eval_steps, metric_history):
        assert metric_snapshot.step == step

    # test that the model training args are recorded as MLflow params
    run = mlflow_client.get_run(run_info.run_id)
    run_dict = run.to_dictionary()
    assert int(run_dict["data"]["params"]["eval_steps"]) == EVAL_STEPS


def _train_func_factory(config, output_path, model, tracking_file_name):
    @OptunaMLflow(num_name_digits=3, enforce_clean_git=False, tracking_uri=tracking_file_name)
    def train(trial: OptunaMLflow):
        test_sentence = "This is a single test sentence"
        tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
        trivial_vocab = "\n".join(test_sentence.lower().split(" ") + tokens)

        vocab_path = output_path / "vocab.txt"
        with vocab_path.open("w") as f:
            f.write(trivial_vocab)

        tokenizer = ElectraTokenizer(vocab_path.as_posix())
        encoded_text = tokenizer(
            [test_sentence],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_token_type_ids=False,
        )
        # trivial labels with trivial text
        dataset = LabeledDataset(encoded_text, [0])

        training_args = TrainingArguments(
            max_steps=MAX_STEPS,
            logging_steps=LOGGING_STEPS,
            evaluation_strategy="steps",
            eval_steps=EVAL_STEPS,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=0.001,
            output_dir=output_path.as_posix(),
            report_to=["none"],
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            compute_metrics=_compute_metrics,
            callbacks=[OMLflowCallback(trial, log_training_args=True, log_model_config=False)],
        )
        trainer.train()

        final_eval_metric_candidates = [
            d for d in trainer.state.log_history if (d["step"] == LAST_EVAL_STEP) and ("eval_acc" in d)
        ]
        assert (
            len(final_eval_metric_candidates) == 1
        ), f"Expected only one final metric dict, got: {final_eval_metric_candidates}"
        final_metrics = final_eval_metric_candidates[0]

        return final_metrics["eval_acc"]

    return train


def _compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"acc": sklearn.metrics.accuracy_score(labels, preds)}


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
