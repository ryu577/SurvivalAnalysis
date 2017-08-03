import numpy as np
import scipy
import pandas as pd

data = pd.read_csv("F:\\data\\NSTWithNodeLevelFeatures.csv", sep = ',')

organic = np.array(data.ix[:,5][data.ix[:,3] == "Ready"])

for i in range(6, len(data.columns)):
	col = np.array(data.ix[:,i][data.ix[:,3] == "Ready"])
	col[np.isnan(col)] = 0
	corr = np.corrcoef(organic, col)[1,0]
	if abs(corr) > 0.1:
		print str(i) + "\t" + str(corr)


