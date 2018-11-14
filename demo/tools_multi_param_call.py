import mltb

def dummy_function(params_dict):
    return params_dict['x'] ** params_dict['y']

params_a = {'x' : 2, 'y' : 3}
params_b = {'x' : 3, 'y' : 2}

params = {'a' : params_a, 'b' : params_b}

result = mltb.tools.multi_param_call(dummy_function, params, 10)

print(result)