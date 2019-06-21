"""A collection of generic machine learning tools."""
import itertools
import sys
from tqdm import tqdm
from scipy import stats
import joblib


def multi_param_call(function, param_dict, iterations, verbose=1):
    """Call function multiple times and return dict with results.

    Calls the given `function` `iterations` times for each entry (value) 
    in the `param_dict`.

    Parameters
    ----------
    function
        The function to call.
    param_dict : dict
        Dictionary with params that are used to call the function.
    iterations : int
        Number of iterations that function will be called for each entry in param_dict.
    verbose : int, optional
        If 1 a progress bar will be shown. If 2 detailed status will be print
        after each iteration. Default is 1.

    Returns
    -------
    dict
        Dict with result values of the `function` calls. If `function` returned just 
        one value the returned dict contains the same keys as `param_dict`. The value is
        an array with one result for each `function` call. The called `function` can also 
        return a dict with results as an alternative. This is if you want to return more 
        then just one result. In this case this result dict contains the same keys
        as the `function` returned on first level. As value it contains a second dict
        as the second level. This second level dict contains the same keys as 
        `param_dict`. The value is an array with one result for each call.
    """
    result = {}
    if verbose == 1:
        number_of_iterations = len(param_dict) * iterations
        pbar = tqdm(total=number_of_iterations, file=sys.stdout)
    for key, value in param_dict.items():
        for i in range(iterations):
            f_result = function(value)

            if isinstance(f_result, dict):
                for f_result_key, f_result_value in f_result.items():
                    sub_dict = result.setdefault(f_result_key, {})
                    sub_dict.setdefault(key, []).append(f_result_value)

            else: 
                result.setdefault(key, []).append(f_result)
            if verbose == 1:
                pbar.update(1)    
            elif verbose == 2:
                print("Done with iteration {} of {} for {}. Result: {}".format(i+1, iterations, key, f_result))
        if verbose == 1:
            pbar.write('Done with {}'.format(key))
    if verbose == 1:
        pbar.close()
    return result


def ttest_combinations(values_dict):
    """Do a t-test on values in a dict and compute the p-value.

    The t-test is computed on each combination of two array_like in the `values_dict`.
    If `values_dict` contains two entries just one p-value is computed. If it contains
    four values this will compute six p-values.

    Parameters
    ----------
    values_dict : dict with key as str and value as array_like
        Dictionary with values to compute the t-test on. At least two entries
        must be present in the dict.

    Returns
    -------
    dict
        Dict with results. Key is a tuple of the compared keys. Value is the p-value.

    See Also
    --------
    Also see the SkiPy function `scipy.stats.ttest_ind`.
    """
    result = {}
    for key_pair in itertools.combinations(values_dict.keys(), 2):
        key_0 = key_pair[0]
        key_1 = key_pair[1]
        result[key_pair] = stats.ttest_ind(values_dict[key_0], values_dict[key_1])[1]
    return result


def save_data_list(data, filename):
    data_list = []

    try:
        data_list = joblib.load(filename)
    except FileNotFoundError:
        pass

    data_list.append(data)
    print('Saving data list of lenth {}.'.format(len(data_list)))
    joblib.dump(data_list, filename, compress=('gzip', 3))


def load_data_list(filename):
    return joblib.load(filename)
