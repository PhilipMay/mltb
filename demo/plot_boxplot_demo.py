import mltb

import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# fake up some data
spread = np.random.rand(50) * 100
center1 = np.ones(25) * 50
center2 = np.ones(25) * 55
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = [np.concatenate((spread, center1, flier_high, flier_low)), np.concatenate((spread, center2))]

mltb.plot.boxplot(data, title="My Title", labels=["A1", "B2"])