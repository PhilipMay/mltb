"""Tools."""
def append_dict_to_dict(target, source):
    if target == {}:
        for key, value in source.items():
            target[key] = [value]
    else:
        for key, value in target.items():
            value.append(source[key])

def append_to_dict(dict, key, value):
    if key in dict:
        list = dict[key]
        list.append(value)
    else:
        list = [value]
        dict[key] = list