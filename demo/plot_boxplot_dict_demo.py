import mltb

data = {"D1": [1, 2, 3, 2, 2, 3, 4, 2, 1, 2], "D2": [9, 7, 6, 7, 8, 9, 9, 8, 6, 6, 9]}
mltb.plot.boxplot_dict(data, title="My Title", xlabel="x label name", ylabel="y label name")
