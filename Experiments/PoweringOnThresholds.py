import sys
sys.path.append('../')
from ParametricNonParametricHybridApproach import *
from kusto_queries import *
from LogLogistic import *

def executeNSTQuery(query):
    t = []
    x = []
    response = kusto_client.execute(kusto_database, query)
    data = response.fetchall();
    time.sleep(0.6)
    for row in data:
        for j in range(int(row[2])):
            if row[0] == 'Ready':
            	t.append(float(row[1])*60)
        	elif row[0] == 'HumanInvestigate':
        		x.append(float(row[1])*60)
    return [np.array(t), np.array(x)]


if __name__ == '__main__':
    [t, x] = executeNSTQuery(allNSTQuery)
    ll = LogLogistic(ti=t, xi=x)
    costs = []
    for tau in np.arange(400, 1230, 30):
    	(p, t) = constructMatrices(tau, ti = t, xi = x, rebootCost = 199.997, distr = ll) ## TODO: Get HI cost.
    	costs.append(TimeToAbsorbing(p1,t1,2)[0])


