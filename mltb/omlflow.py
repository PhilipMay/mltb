import logging
import optuna
import platform
import textwrap
import traceback
import warnings

import mlflow
from mlflow.tracking.context.default_context import _get_user


_logger = logging.getLogger(__name__)


class OptunaMLflow(object):

    def __init__(self, trial, tracking_uri, nun_name_digits=3):
        self._trial = trial
        self._tracking_uri = tracking_uri
        self._nun_name_digits = nun_name_digits
        self._max_mlflow_tag_length = 5000
        self._fold_metrics = {}

    #####################################
    # MLflow wrapper functions
    #####################################

    def end_run(self, status):
        try:
            mlflow.end_run(status)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    def log_metric(self, key, value, step=None):
        print('log_metric', value)
        self._trial.set_user_attr(key, value)
        try:
            mlflow.log_metric(key, value, step=None)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    def log_metrics(self, metrics, step=None, optuna_log=True):
        print('log_metrics', metrics)
        if optuna_log:
            for key, value in metrics.items():
                self._trial.set_user_attr(key, value)
        try:
            mlflow.log_metrics(metrics, step=None)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    def log_param(self, key, value, optuna_log=True):
        print('log_param', value)
        if optuna_log:
            self._trial.set_user_attr(key, value)
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    def log_params(self, params):
        print('log_params', params)
        for key, value in params.items():
            self._trial.set_user_attr(key, value)
        try:
            mlflow.log_params(params)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    def set_tag(self, key, value, optuna_log=True):
        print('set_tag', value)
        if optuna_log:
            self._trial.set_user_attr(key, value)
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
        print('set_tags', tags)
        for key, value in tags.items():
            if optuna_log:
                self._trial.set_user_attr(key, value)
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

    def log_fold(self, step, metrics):  # TODO: add params and tags?
        print('log_fold', step, metrics)
        for key, value in metrics.items():
            value_list = self._fold_metrics.get(key, [])
            value_list.append(value)
            self._fold_metrics[key] = value_list
            self._trial.set_user_attr("{}_folds".format(key), value_list)
        digits_format_string = "{{:0{0}d}}-{{:0{0}d}}".format(self._nun_name_digits)
        try:
            with mlflow.start_run(
                run_name=digits_format_string.format(self._trial.number, step),
                nested=True
            ):
                # overwrite user with user + hostname
                self.set_tag('mlflow.user', self._get_user_hostname(), optuna_log=False)

                self.log_metrics(metrics, step=step, optuna_log=False)
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )

    #####################################
    # util functions
    #####################################

    def _get_user_hostname(self):
        # TODO: user caching here?
        user = 'unknown'
        hostname = 'unknown'
        try:
            user = _get_user()
            hostname = platform.node()
        except Exception as e:
            warnings.warn('Exception while getting user and hostname! {}'
                          .format(e), RuntimeWarning)
        return '{}@{}'.format(user, hostname)

    #####################################
    # context manager functions
    #####################################

    def __enter__(self):
        print('enter')
        try:
            # This sets the tracking_uri for MLflow.
            if self._tracking_uri is not None:
                mlflow.set_tracking_uri(self._tracking_uri)

            mlflow.set_experiment(self._trial.study.study_name)

            digits_format_string = "{{:0{}d}}".format(self._nun_name_digits)
            mlflow.start_run(run_name=digits_format_string.format(self._trial.number))

            # overwrite user with user + hostname
            self.set_tag('mlflow.user', self._get_user_hostname())
        except Exception as e:
            _logger.error(
                "Exception raised during MLflow communication! Exception: {}".format(e),
                exc_info=True,
            )
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:  # no exception

            # extract tags from trial
            tags = {}
            # Set direction and convert it to str and remove the common prefix.
            study_direction = self._trial.study.direction
            if isinstance(study_direction, optuna._study_direction.StudyDirection):
                tags["direction"] = str(study_direction).split(".")[-1]

            distributions = {
                (k + "_distribution"): str(v) for (k, v) in self._trial.distributions.items()
            }
            tags.update(distributions)
            self.set_tags(tags)

            self.end_run('FINISHED')
            print('exit', type, '#', exc_value, '#', tb)
        else:  # exception
            print('Exception!', exc_type, '#', exc_value, '#', tb)
            print(type(exc_type))
            exc_text = ''.join(traceback.format_exception(exc_type, exc_value, tb))
            self.set_tag('exception', exc_text)
            if exc_type is KeyboardInterrupt:
                print('set KILLED')
                self.end_run('KILLED')
            else:
                print('set FAILED')
                self.end_run('FAILED')
            return False

    #####################################
    # Optuna wrapper functions
    #####################################

    def report(self, value, step):
        self._trial.report(value, step)

    def should_prune(self):
        return self._trial.should_prune()

    def suggest_categorical(self, name, choices):
        result = self._trial.suggest_categorical(name, choices)
        self.log_param(name, result, optuna_log=False)
        return result

    def suggest_discrete_uniform(self, name, low, high, q):
        result = self._trial.suggest_discrete_uniform(name, low, high, q)
        self.log_param(name, result, optuna_log=False)
        return result

    def suggest_int(self, name, low, high, step=1, log=False):
        result = self._trial.suggest_int(name, low, high, step, log)
        self.log_param(name, result, optuna_log=False)
        return result

    def suggest_loguniform(self, name, low, high):
        result = self._trial.suggest_loguniform(name, low, high)
        self.log_param(name, result, optuna_log=False)
        return result

    def suggest_uniform(self, name, low, high):
        result = self._trial.suggest_uniform(name, low, high)
        self.log_param(name, result, optuna_log=False)
        return result


# TODO: git dirty check: https://gitpython.readthedocs.io/en/stable/reference.html?highlight=dirty#git.repo.base.Repo.is_dirty
# TODO: Finish nested metrics for crossvalidation or repeated training
