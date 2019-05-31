"""Hyperopt tools."""

import joblib
import hyperopt


def fmin(fn,
         space,
         algo,
         max_evals,
         filename,
         rstate=None,
         pass_expr_memo_ctrl=None,
         verbose=0,
         max_queue_len=1,
         show_progressbar=True,
        ):
    """Minimize a function with hyperopt and save results to disk for later restart.

    Parameters
    ----------
    fn : callable (trial point -> loss)
        This function will be called with a value generated from `space`
        as the first and possibly only argument.  It can return either
        a scalar-valued loss, or a dictionary.  A returned dictionary must
        contain a 'status' key with a value from `STATUS_STRINGS`, must
        contain a 'loss' key if the status is `STATUS_OK`. Particular
        optimization algorithms may look for other keys as well.  An
        optional sub-dictionary associated with an 'attachments' key will
        be removed by fmin its contents will be available via
        `trials.trial_attachments`. The rest (usually all) of the returned
        dictionary will be stored and available later as some 'result'
        sub-dictionary within `trials.trials`.
    space : hyperopt.pyll.Apply node
        The set of possible arguments to `fn` is the set of objects
        that could be created with non-zero probability by drawing randomly
        from this stochastic program involving involving hp_<xxx> nodes
        (see `hyperopt.hp` and `hyperopt.pyll_utils`).
    algo : search algorithm
        This object, such as `hyperopt.rand.suggest` and
        `hyperopt.tpe.suggest` provides logic for sequential search of the
        hyperparameter space.
    max_evals : int
        Allow up to this many additional function evaluations before returning.
    filename : str, pathlib.Path, or file object
        Filename where to store the results for later restart. Results will be
        stored as a pickled hyperopt Trials object which is gziped.
    rstate : numpy.RandomState, default numpy.random or `$HYPEROPT_FMIN_SEED`
        Each call to `algo` requires a seed value, which should be different
        on each call. This object is used to draw these seeds via `randint`.
        The default rstate is
        `numpy.random.RandomState(int(env['HYPEROPT_FMIN_SEED']))`
        if the `HYPEROPT_FMIN_SEED` environment variable is set to a non-empty
        string, otherwise np.random is used in whatever state it is in.
    verbose : int
        Print out some information to stdout during search.
    pass_expr_memo_ctrl : bool, default False
        If set to True, `fn` will be called in a different more low-level
        way: it will receive raw hyperparameters, a partially-populated
        `memo`, and a Ctrl object for communication with this Trials
        object.
    max_queue_len : integer, default 1
        Sets the queue length generated in the dictionary or trials. Increasing this
        value helps to slightly speed up parallel simulatulations which sometimes lag
        on suggesting a new trial.
    show_progressbar : bool, default True
        Show a progressbar.

    Returns
    -------
    list : (argmin, trials)
        argmin : dictionary
            If return_argmin is True returns `trials.argmin` which is a dictionary.  Otherwise
            this function  returns the result of `hyperopt.space_eval(space, trails.argmin)` if there
            were succesfull trails. This object shares the same structure as the space passed.
            If there were no succesfull trails, it returns None.
        trials : hyperopt.Trials
            The hyperopt trials object that also gets stored to disk.
    """
    try:
        trials = joblib.load(filename)
        evals_loaded_trials = len(trials.statuses())
        max_evals += evals_loaded_trials
        print('{} evals loaded from trials file "{}".'.format(evals_loaded_trials, filename))
    except FileNotFoundError:
        trials = hyperopt.Trials()
        print('No trials file "{}" found. Created new trials object.'.format(filename))

    result = hyperopt.fmin(fn,
                  space,
                  algo,
                  max_evals,
                  trials=trials,
                  rstate=rstate,
                  pass_expr_memo_ctrl=pass_expr_memo_ctrl,
                  verbose=verbose,
                  return_argmin=True,
                  max_queue_len=max_queue_len,
                  show_progressbar=show_progressbar,
                )

    joblib.dump(trials, filename, compress=('gzip', 3))

    return result, trials
