def multi_param_call(function, param_dict, iterations, verbose=1):
    """Call function multiple times and return dict with resultsself.

    Parameters
    ----------
    function
        The function to callself.
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
            result.setdefault(key, []).append(f_result)
            if verbose == 1:
                print("Done with %s - iteration %i of %i." % (key, i+1, iterations))
    return result