import mltb

def dummy_function_a(params_dict):
    return params_dict['x'] ** params_dict['y']

params_a = {'x' : 2, 'y' : 3}
params_b = {'x' : 3, 'y' : 2}

params = {'a' : params_a, 'b' : params_b}

result = mltb.tools.multi_param_call(dummy_function_a, params, 10)

print(result)


def dummy_function_b(params_dict):

    val_a = params_dict['x'] ** params_dict['y']
    val_b = params_dict['x'] + 1

    return {'val_a' : val_a, 'val_b' : val_b}

params_a = {'x' : 2, 'y' : 3}
params_b = {'x' : 3, 'y' : 2}

params = {'a' : params_a, 'b' : params_b}

result = mltb.tools.multi_param_call(dummy_function_b, params, 10)

print(result)
print(result['val_a'])
print(result['val_b'])
