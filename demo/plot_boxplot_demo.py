import mltb

data = [[1,2,3,2,2,3,4,2,1,2], [9,7,6,7,8,9,9,8,6,6,9]]
mltb.plot.boxplot(data, title="My Title", labels=["A1", "B2"], xlabel="x label name", ylabel='y label name')
