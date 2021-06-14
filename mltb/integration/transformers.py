# Copyright (c) 2021 by Timothy Wolff-Piggott
# This software is distributed under the terms of the BSD 2-Clause License.
# For details see the LICENSE file in the root directory.

import logging
import os

import mlflow
from optuna._imports import try_import

from mltb.omlflow import OptunaMLflow

with try_import() as _imports:
    import transformers
_imports.check()

_logger = logging.getLogger(__name__)


class OMLflowCallback(transformers.TrainerCallback):
    """
    Class based on `transformers.TrainerCallback`; integrates with OptunaMLflow
    to send the logs to `MLflow` and `Optuna` during model training.
    """

    def __init__(
        self,
        trial: OptunaMLflow,
        log_training_args: bool = True,
        log_model_config: bool = True,
    ):
        """
        Check integration package dependencies and initialize class.

        Args:
            trial: OptunaMLflow object
            log_training_args: Whether to log all Transformers TrainingArguments as MLflow params
            log_model_config: Whether to log the Transformers model config as MLflow params
        """
        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH

        self._initialized = False
        self._log_artifacts = False
        self._ml_flow = mlflow
        self._trial = trial
        self._log_training_args = log_training_args
        self._log_model_config = log_model_config

    def setup(self, args, state, model):
        """
        Setup the optional MLflow integration.

        Environment:
            HF_MLFLOW_LOG_ARTIFACTS (:obj:`str`, `optional`):
                Whether to use MLflow .log_artifact() facility to log artifacts.
                This only makes sense if logging to a remote server, e.g. s3 or GCS.
                If set to `True` or `1`, will copy whatever is in TrainerArgument's output_dir
                to the local or remote artifact storage. Using it without a remote storage will
                just copy the files to your artifact location.
        """
        log_artifacts = os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper()
        if log_artifacts in {"TRUE", "1"}:
            self._log_artifacts = True
        if state.is_world_process_zero:
            combined_dict = dict()
            if self._log_training_args:
                _logger.info("Logging training arguments.")
                combined_dict.update(args.to_dict())
            if self._log_model_config and hasattr(model, "config") and model.config is not None:
                _logger.info("Logging model config.")
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    _logger.warning(
                        f"Trainer is attempting to log a value of "
                        f'"{value}" for key "{name}" as a parameter. '
                        f"MLflow's log_param() only accepts values no longer than "
                        f"250 characters so we dropped this attribute."
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                self._trial.log_params(dict(combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH]))
        self._initialized = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """
        Call setup if not yet initialized.
        """
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        """
        Log all metrics from Transformers logs as MLflow metrics at the appropriate step.
        """
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            metrics_to_log = dict()
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    metrics_to_log[k] = v
                else:
                    _logger.warning(
                        f"Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a metric. '
                        f"MLflow's log_metric() only accepts float and "
                        f"int types so we dropped this attribute."
                    )
            self._trial.log_metrics(metrics_to_log, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """
        Log the training output as MLflow artifacts if logging artifacts is enabled.
        """
        if self._initialized and state.is_world_process_zero:
            if self._log_artifacts:
                _logger.info("Logging artifacts. This may take time.")
                self._ml_flow.log_artifacts(args.output_dir)
