import mltb

a = [2, 3, 4, 5, 6, 4, 3, 2]
b = [20, 30, 40, 50, 60, 40, 30, 20]
c = [4, 5, 4, 3, 2, 4, 1, 0, 5, 4, 3]

my_dict = {"a": a, "b": b, "c": c}

result = mltb.tools.ttest_combinations(my_dict)

print(result)
