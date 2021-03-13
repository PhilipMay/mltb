import logging
import optuna
import platform
import textwrap
import traceback
import warnings
import git
import os
from functools import wraps
import sys

import mlflow
from mlflow.entities import RunStatus
from mlflow.tracking.context.default_context import _get_main_file


_logger = logging.getLogger(__name__)


class OptunaMLflow(object):
    """Wrapper to log to Optuna and MLflow at the same time."""

    def __init__(
        self,
        tracking_uri=None,
        num_name_digits=3,
        enforce_clean_git=False,
        optuna_result_name="optuna_result"
    ):
        """Init class.

        Args:
            tracking_uri ([str], optional): See `MLflow documentation
                <https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri>`_.
                Defaults to ``None`` which logs to the default locale folder ``./mlruns`` or
                uses the ``MLFLOW_TRACKING_URI`` environment variable if it is available.
            num_name_digits (int, optional): [description]. Defaults to 3.
            enforce_clean_git (bool, optional): Check and enforce that the GIT repository has no
                uncommited changes. Defaults to False. Also see `git.repo.base.Repo.is_dirty
                <https://gitpython.readthedocs.io/en/stable/reference.html#git.repo.base.Repo.is_dirty>`_
        """
        self._tracking_uri = tracking_uri
        self._num_name_digits = num_name_digits
        self._enforce_clean_git = enforce_clean_git
        self._max_mlflow_tag_length = 5000
        self._iter_metrics = {}
        self._next_iter_num = 0
        self._hostname = None
        self._optuna_result_name = optuna_result_name

    def __call__(self, func):

        @wraps(func)
        def objective_decorator(trial):
            """Decorator for optuna objective function.

            Args:
                trial ([``optuna.trial.Trial``]): The optuna trial to use.
            """
            self._trial = trial

            # check if GIT repo is clean
            if self._enforce_clean_git:
                self._check_repo_is_dirty()

            try:
                # set tracking_uri for MLflow
                if self._tracking_uri is not None:
                    mlflow.set_tracking_uri(self._tracking_uri)

                mlflow.set_experiment(self._trial.study.study_name)

                digits_format_string = "{{:0{}d}}".format(self._num_name_digits)
                mlflow.start_run(run_name=digits_format_string.format(self._trial.number))
                _logger.info("Run {} started.".format(self._trial.number))

                # TODO: use set_tags with dict
                self.set_tag("hostname", self._get_hostname())
                self.set_tag("process_id", os.getpid())
            except Exception as e:
                _logger.error(
                    "Exception raised during MLflow communication! Exception: {}".format(e),
                    exc_info=True,
                )

            try:
                # call objective function
                result = func(self)

                # log the result to MLflow but not optuna
                self.log_metric(self._optuna_result_name, result, optuna_log=False)

                # extract tags from trial
                tags = {}
                # Set direction and convert it to str and remove the common prefix.
                study_direction = self._trial.study.direction
                if isinstance(study_direction, optuna._study_direction.StudyDirection):
                    tags["direction"] = str(study_direction).split(".")[-1]

                distributions = {(k + "_distribution"): str(v) for (k, v) in self._trial.distributions.items()}
                tags.update(distributions)
                self.set_tags(tags, optuna_log=False)
                self._end_run("FINISHED")

                return result
            except:  # noqa: E722
                exc_type, exc_value, exc_traceback = sys.exc_info()
                exc_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                self.set_tag("exception", exc_text)
                if exc_type is KeyboardInterrupt:
                    self._end_run(RunStatus.to_string(RunStatus.KILLED))
                else:
                    self._end_run(RunStatus.to_string(RunStatus.FAILED))
                raise  # raise exception again

        return objective_decorator

    #####################################
    # MLflow wrapper functions
    #####################################

    def log_metric(self, key, value, step=None, optuna_log=True):
        """Wrapper of the corresponding MLflow function.

        The data is also added to Optuna as an user attribute.

        Args:
            optuna_log (bool, optional): Internal parameter that should be ignored by the API user.
                Defaults to True.
        """
        if optuna_log:
            self._trial.set_user_attr(key, value)
        _logger.info(f"Metric: {key}: {value} at step: {step}")
        try:
            mlflow.log_metric(key, value, step=None)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    def log_metrics(self, metrics, step=None, optuna_log=True):
        """Wrapper of the corresponding MLflow function.

        The data is also added to Optuna as an user attribute.

        Args:
            metrics ([Dict]): Dict of metrics.
            optuna_log (bool, optional): Internal parameter that should be ignored by the API user.
                Defaults to True.
        """
        for key, value in metrics.items():
            if optuna_log:
                self._trial.set_user_attr(key, value)
            _logger.info(f"Metric: {key}: {value} at step: {step}")
        try:
            mlflow.log_metrics(metrics, step=None)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    def log_param(self, key, value, optuna_log=True):
        """Wrapper of the corresponding MLflow function.

        The data is also added to Optuna as an user attribute.

        Args:
            optuna_log (bool, optional): Internal parameter that should be ignored by the API user.
                Defaults to True.
        """
        if optuna_log:
            self._trial.set_user_attr(key, value)
        _logger.info(f"Param: {key}: {value}")
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    def log_params(self, params):
        """Wrapper of the corresponding MLflow function.

        The data is also added to Optuna as an user attribute.

        Args:
            params ([Dict]): Dict of params.
        """
        for key, value in params.items():
            self._trial.set_user_attr(key, value)
            _logger.info(f"Param: {key}: {value}")
        try:
            mlflow.log_params(params)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    def set_tag(self, key, value, optuna_log=True):
        """Wrapper of the corresponding MLflow function.

        The data is also added to Optuna as an user attribute.

        Args:
            optuna_log (bool, optional): Internal parameter that should be ignored by the API user.
                Defaults to True.
        """
        if optuna_log:
            self._trial.set_user_attr(key, value)
        _logger.info(f"Tag: {key}: {value}")
        value = str(value)  # make sure it is a string
        if len(value) > self._max_mlflow_tag_length:
            value = textwrap.shorten(value, self._max_mlflow_tag_length)
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    def set_tags(self, tags, optuna_log=True):
        """Wrapper of the corresponding MLflow function.

        The data is also added to Optuna as an user attribute.

        Args:
            tags ([Dict]): Dict of tags.
            optuna_log (bool, optional): Internal parameter that should be ignored by the API user.
                Defaults to True.
        """
        for key, value in tags.items():
            if optuna_log:
                self._trial.set_user_attr(key, value)
            _logger.info(f"Tag: {key}: {value}")
            value = str(value)  # make sure it is a string
            if len(value) > self._max_mlflow_tag_length:
                tags[key] = textwrap.shorten(value, self._max_mlflow_tag_length)
        try:
            mlflow.set_tags(tags)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    def log_iter(self, metrics, step=None):  # TODO: add params and tags?
        """"Log an iteration or fold as a nasted run."""
        for key, value in metrics.items():
            value_list = self._iter_metrics.get(key, [])
            value_list.append(value)
            self._iter_metrics[key] = value_list
            self._trial.set_user_attr("{}_iter".format(key), value_list)
            _logger.info(f"Iteration metric: {key}: {value} at step: {step}")
        digits_format_string = "{{:0{0}d}}-{{:0{0}d}}".format(self._num_name_digits)
        if step is None:
            step = self._next_iter_num
            self._next_iter_num += 1
        try:
            with mlflow.start_run(run_name=digits_format_string.format(self._trial.number, step), nested=True):
                self.log_metrics(metrics, step=step, optuna_log=False)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    def _end_run(self, status, exc_text=None):
        try:
            mlflow.end_run(status)
            if exc_text is None:
                _logger.info("Run finished with status: {}".format(status))
            else:
                _logger.error("Run finished with status: {}, exc_text:{}".format(status, exc_text))
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    #####################################
    # util functions
    #####################################

    def _get_hostname(self):
        if self._hostname is None:
            hostname = "unknown"
            try:
                hostname = platform.node()
            except Exception as e:
                warn_msg = "Exception while getting hostname! {}".format(e)
                _logger.warn(warn_msg)
                warnings.warn(warn_msg, RuntimeWarning)
            self._hostname = hostname
        return self._hostname

    def _check_repo_is_dirty(self):
        path = _get_main_file()
        if os.path.isfile(path):
            path = os.path.dirname(path)
        repo = git.Repo(path, search_parent_directories=True)
        if repo.is_dirty():
            error_message = "Git repository '{}' is dirty!".format(path)
            _logger.error(error_message)
            raise RuntimeError(error_message)

    #####################################
    # Optuna wrapper functions
    #####################################

    def report(self, value, step):
        """Wrapper of the corresponding Optuna function."""
        self._trial.report(value, step)

    def should_prune(self):
        """Wrapper of the corresponding Optuna function."""
        return self._trial.should_prune()

    def suggest_categorical(self, name, choices):
        """Wrapper of the corresponding Optuna function."""
        result = self._trial.suggest_categorical(name, choices)
        self.log_param(name, result, optuna_log=False)
        return result

    def suggest_discrete_uniform(self, name, low, high, q):
        """Wrapper of the corresponding Optuna function."""
        result = self._trial.suggest_discrete_uniform(name, low, high, q)
        self.log_param(name, result, optuna_log=False)
        return result

    def suggest_int(self, name, low, high, step=1, log=False):
        """Wrapper of the corresponding Optuna function."""
        result = self._trial.suggest_int(name, low, high, step, log)
        self.log_param(name, result, optuna_log=False)
        return result

    def suggest_loguniform(self, name, low, high):
        """Wrapper of the corresponding Optuna function."""
        result = self._trial.suggest_loguniform(name, low, high)
        self.log_param(name, result, optuna_log=False)
        return result

    def suggest_uniform(self, name, low, high):
        """Wrapper of the corresponding Optuna function."""
        result = self._trial.suggest_uniform(name, low, high)
        self.log_param(name, result, optuna_log=False)
        return result
