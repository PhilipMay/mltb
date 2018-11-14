def multi_param_call(function, param_dict, iterations):
    result = {}
    for key, value in param_dict.items():
        for i in range(iterations):
            f_result = function(value)
            result.setdefault(key, []).append(f_result)
            print("Done with %s - iteration %i of %i." % (key, i+1, iterations))
    return result