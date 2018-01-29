import numpy as np
import scipy
import pandas as pd

data = pd.read_csv("F:\\data\\diskFailureData.tsv", sep = '\t', header=None)

organic = np.array(data.ix[:,0])
organic[np.isnan(organic)] = 0

for i in range(3, len(data.columns)):
	col = np.array(data.ix[:,i])
	col[np.isnan(col)] = 0
	corr = np.corrcoef(organic, col)[1,0]
	if abs(corr) > 0.3:
		print str(i) + "\t" + str(corr)
	

data = pd.read_csv("F:\\data\\NSTWithNodeLevelFeatures.csv", sep = '\t')

organic = np.array(data.ix[:,5])#[data.ix[:,3] == "Ready"])

for i in range(6, len(data.columns)):
	col = np.array(data.ix[:,i])#[data.ix[:,3] == "Ready"])
	col[np.isnan(col)] = 0
	corr = np.corrcoef(organic, col)[1,0]
	if abs(corr) > 0.03:
		#print str(i) + "\t" + str(corr)
		print str(i) + ",",


#####################################################
## Logistic regression sample - 

import numpy as np

#Construct some input toy data it is clear that the first three rows are different from the last two.
X = np.array(
	[
		[1,0.1,10.0],
		[1,0.15,11.0],
		[1,0.09,10.5],
		[1,10.0,0.1],
		[1,11.0,0.12]
	])

#Construct the toy labels. Again, first three are different from last two.
y = np.array([1,1,1,0,0])

#Initialize the logistic regression.
from sklearn.linear_model import LogisticRegression
est = LogisticRegression()

#Fit the logistic regression.
est.fit(X,y)

# This gives a prediction of the classes. If >0 implies class 1 and <0 implies class 0.
np.dot(X,est.coef_.T)

# And this gives the probabilities of belonging to class 1.
1/(1+np.exp(-np.dot(X,est.coef_.T)))


