import itertools
from scipy import stats


def multi_param_call(function, param_dict, iterations, verbose=1):
    """Call function multiple times and return dict with results.

    Parameters
    ----------
    function
        The function to call.
    param_dict : dictionary
        Dictionary with params that are used to call the function.
    iterations : int
        Number of iterations that function will be called.
    verbose : int, optional
        If 1 status will be print, else not.
    """
    result = {}
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
                print("Done with %s - iteration %i of %i." % (key, i+1, iterations))
    return result

def ttest_combinations(values_dict):
    result = {}
    for key_pair in itertools.combinations(values_dict.keys(), 2):
        key_0 = key_pair[0]
        key_1 = key_pair[1]
        result[key_pair] = stats.ttest_ind(values_dict[key_0], values_dict[key_1])[1]
    return result