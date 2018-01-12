import matplotlib.pyplot as plt
import datetime
import numpy as np
import time
from MarkovChains import *
from GlobalThresholdModel import *

'''
Constructs the matrices and calculates costs corresponding to a range of thresholds.
'''
def constructFullMatricesByHW():
    ## First get global vectors.
    [powOnProbsBase, powOnTimesBase] = populateRawVecs(powOnAllTrnstns, {'PoweringOn'})
    [powOnProbs, powOnTimes] = populateRawVecsForFeature(powOnAllTrnstnsWHWQry, {'PoweringOn'}, 'Wiwynn Azure Gen5 1 Compute C1042H', powOnProbsBase * 10.0, powOnTimesBase)
    
    [unhlProbsBase, unhlTimesBase] = populateRawVecs(unhlAllTrnstns, {'Unhealthy'})
    [unhlProbs, unhlTimes] = populateRawVecsForFeature(unhlAllTrnstnsWHWQry, {'Unhealthy'}, 'Wiwynn Azure Gen5 1 Compute C1042H', unhlProbsBase * 10.0, unhlTimesBase)

    [deadProbsBase, deadTimesBase] = populateRawVecs(deadAllTranstns, {'Dead'})
	[deadProbs, deadTimes] = populateRawVecsForFeature(deadAllTrnstnsWHWQry, {'Dead'}, 'Wiwynn Azure Gen5 1 Compute C1042H', deadProbsBase * 10.0, deadTimesBase)

	hiProbs = np.array([0,0,0,0,0,1])
	hiTimes = np.array([0,0,0,0,0,getHiCost(hiCostQuery)*60])#####TODO!!

    [readyProbsBase, readyTimesBase] = populateRawVecs(readyQuery, {'Ready'})
	[readyProbs, readyTimes] = populateRawVecsForFeature(readyTransitionsWHWQry, {'Dead'}, 'Wiwynn Azure Gen5 1 Compute C1042H', deadProbsBase * 10.0, deadTimesBase)

    probs = np.matrix(np.array([unhlProbs, powOnProbs, bootingProbs, hiProbs, deadProbs, readyProbs]))
    times = np.matrix(np.array([unhlTimes, powOnTimes, bootingTimes, hiTimes, deadTimes, readyTimes]))

def populateRawVecsForFeature(query, ignore = {}, level = 'Wiwynn Gen6 Optimized', baseCounts = np.zeros(6), baseTimes = np.zeros(6)):
	mapping = {'Ready':5, 'HumanInvestigate' : 3, 'Dead' : 4, 'PoweringOn': 1, 'Unhealthy' : 0, 'Booting' : 2}
    powOnCounts = np.zeros(6)
    powOnTimes = np.zeros(6)
    response = kusto_client.execute(kusto_database, query)
    data = response.fetchall();
    time.sleep(0.6)
    for row in data:
    	if row[0] not in ignore and row[0] in mapping and row[1] == level:
            powOnCounts[mapping[row[0]]] += row[3]
            powOnTimes[mapping[row[0]]] += row[4]
    powOnTimes = (powOnTimes + baseTimes * baseCounts) / (powOnCounts + baseCounts)
    powOnCounts += baseCounts
    return [powOnCounts/sum(powOnCounts), powOnTimes]

##################################################
## Queries
allBootingEventsWHWQry = ("NodeStateTransitions | where ContainerCount > 0 and PreciseTimeStamp > ago(20d)\n" +
"| where OldState in (\"Booting\")\n" +
"| summarize by DurationInSeconds, ContainerCount, NodeId, PreciseTimeStamp = bin(PreciseTimeStamp, 1s), Cluster, NewState\n" +
"| join kind = inner (\n"+
"cluster(\"Azurecm\").database('AzureCM').TMMgmtNodeEventsEtwTable\n" +
"| where PreciseTimeStamp >= datetime(11-19-2017)\n" +
"and Message contains \"PxeInfo suggests the Node must be booting. Based on HealthTimeoutsRebooting and lastPxeReqArrivalTime\" and Tenant contains \"prdapp\"\n" +
"| project pxetime = bin(PreciseTimeStamp,1s) , Tenant , NodeId , Message\n" +
"| extend additionalt = todouble(extract(\"let's give it ([0-9]+)\", 1, Message))\n" +
"| where additionalt <= 900| extend trupxetime = pxetime - (900 - additionalt)*1s) on NodeId\n" +
"| extend delta = iff(pxetime - 20m < PreciseTimeStamp and PreciseTimeStamp < pxetime + 20m, 1, 0)\n" +
"| where (isnull(pxetime) or delta > 0)\n" +
"| project DurationInSeconds, NodeId, bin(PreciseTimeStamp,1s), bin(pxetime,1s), Cluster, ContainerCount, additionalt, \n" +
"trupxetime, pxeToNew = (PreciseTimeStamp - trupxetime)/1s , NewState\n" +
"| extend pxeToBooting = -DurationInSeconds + pxeToNew\n" +
"| join kind = leftouter \n" +
"(\n" +
    "CADDAILY | where StartTime > ago(20d)\n" +
    "| summarize by NodeId, Hardware_Model\n" +
")\n" +
"on NodeId\n" +
"| where pxeToNew > 0\n" +
"| summarize argmin(pxeToNew, additionalt, pxeToBooting) by DurationInSeconds, NodeId, bin(PreciseTimeStamp,1s), ContainerCount, NewState, Hardware_Model\n" +
"| project gammaprime = min_pxeToNew_pxeToBooting, T3Prime = DurationInSeconds, NewState, ContainerCount, Hardware_Model\n" +
"| extend T3 = gammaprime + T3Prime")

###################################################
## Transitions from old states.
powOnAllTrnstnsWHWQry = ("cluster(\"Azurecm\").database('AzureCM').NodeStateChangeDurationDetails | \n" +
"where PreciseTimeStamp > ago(20d)\n" +
"and oldState in (\"PoweringOn\", \"Recovering\") and Tenant contains \"prdapp\" \n" +
"| project NodeId = nodeId, Tenant, PreciseTimeStamp, oldState, newState, durtn = todouble(stateDurationMilliseconds)/1000\n" +
"| extend OldTime = PreciseTimeStamp - durtn * 1s\n" +
"| project NodeId, Tenant, OldTime, NewTime = PreciseTimeStamp, \n" +
"oldState = iff(oldState == \"Recevering\", \"PoweringOn\", oldState), \n" +
"newState= iff(newState == \"Recovering\", \"PoweringOn\", newState), \n" +
"DurationInMin = durtn/60.0 \n" +
"| where  not (DurationInMin < 0.1)\n" +
"| join kind = inner (\n" +
   "cluster('vmainsight').database('vmadb').DailyVMCountsPerNode |\n" +
   "where TimeStamp > ago(21d)\n" +
   "| project NodeId, VMCount, snapTime = TimeStamp \n" +
") on NodeId  \n" +
"| where (OldTime - snapTime) > -5m and (OldTime - snapTime) < 1440m\n" +
"| project NodeId, Tenant, OldTime, NewTime, oldState, newState, DurationInMin, \n" +
"delta = iff(OldTime < snapTime, 2000*(snapTime - OldTime)/1m, (OldTime - snapTime)/1m), VMCount \n" +
"| summarize argmin(delta, VMCount) by NodeId, Tenant, OldTime, NewTime, oldState, newState, DurationInMin\n" +
"| where min_delta_VMCount > 0\n" +
"| extend durtnMultVMs = DurationInMin*min_delta_VMCount \n" +
"| join kind = leftouter \n" +
"(\n" +
    "CADDAILY | where StartTime > ago(20d)\n" +
    "| summarize by NodeId, Hardware_Model\n" +
")\n" +
"on NodeId\n" +
"| summarize sumdurtn = sum(durtnMultVMs), sumvms = sum(min_delta_VMCount) by newState, Hardware_Model\n" +
"| extend avgt = sumdurtn / sumvms * 60\n" +
"| where newState in (\"Ready\", \"Booting\", \"HumanInvestigate\", \"Dead\", \"Unhealthy\")")

unhlAllTrnstnsWHWQry = ("cluster(\"Azurecm\").database('AzureCM').NodeStateChangeDurationDetails | \n" +
"where PreciseTimeStamp > ago(20d)\n" +
"and oldState == \"Unhealthy\" and Tenant contains \"prdapp\" \n" +
"| project NodeId = nodeId, Tenant, PreciseTimeStamp, oldState, newState, durtn = todouble(stateDurationMilliseconds)/1000\n" +
"| where durtn > 75\n" +
"| extend OldTime = PreciseTimeStamp - durtn * 1s\n" +
"| project NodeId, Tenant, OldTime, NewTime = PreciseTimeStamp, \n" +
"oldState = iff(oldState == \"Recevering\", \"PoweringOn\", oldState), \n" +
"newState= iff(newState == \"Recovering\", \"PoweringOn\", newState), \n" +
"DurationInMin = durtn/60.0 \n" +
"| where  not (DurationInMin < 0.1)\n" +
"| join kind = inner (\n" +
   "cluster('vmainsight').database('vmadb').DailyVMCountsPerNode |\n" +
   "where TimeStamp > ago(21d)\n" +
   "| project NodeId, VMCount, snapTime = TimeStamp \n" +
") on NodeId  \n" +
"| where (OldTime - snapTime) > -5m and (OldTime - snapTime) < 1440m\n" +
"| project NodeId, Tenant, OldTime, NewTime, oldState, newState, DurationInMin, \n" +
"delta = iff(OldTime < snapTime, 2000*(snapTime - OldTime)/1m, (OldTime - snapTime)/1m), VMCount \n" +
"| summarize argmin(delta, VMCount) by NodeId, Tenant, OldTime, NewTime, oldState, newState, DurationInMin\n" +
"| where min_delta_VMCount > 0\n" +
"| extend durtnMultVMs = DurationInMin*min_delta_VMCount \n" +
"| join kind = leftouter \n" +
"(\n" +
    "CADDAILY | where StartTime > ago(20d)\n" +
    "| summarize by NodeId, Hardware_Model\n" +
")\n" +
"on NodeId\n" +
"| summarize sumdurtn = sum(durtnMultVMs), sumvms = sum(min_delta_VMCount) by newState, Hardware_Model\n" +
"| extend avgt = sumdurtn / sumvms * 60\n" +
"| where newState in (\"Ready\", \"Booting\", \"HumanInvestigate\", \"Dead\", \"Unhealthy\", \"PoweringOn\", \"Recovering\")")

deadAllTrnstnsWHWQry = ("cluster(\"Azurecm\").database('AzureCM').NodeStateChangeDurationDetails | \n" +
"where PreciseTimeStamp > ago(20d)\n" +
"and oldState in (\"Dead\") and Tenant contains \"prdapp\" \n" +
"| project NodeId = nodeId, Tenant, PreciseTimeStamp, oldState, newState, durtn = todouble(stateDurationMilliseconds)/1000\n" +
"| extend OldTime = PreciseTimeStamp - durtn * 1s\n" +
"| project NodeId, Tenant, OldTime, NewTime = PreciseTimeStamp, \n" +
"oldState = iff(oldState == \"Recevering\", \"PoweringOn\", oldState), \n" +
"newState= iff(newState == \"Recovering\", \"PoweringOn\", newState), \n" +
"DurationInMin = durtn/60.0 \n" +
"| where  not (DurationInMin < 0.1)\n" +
"| join kind = inner (\n" +
   "cluster('vmainsight').database('vmadb').DailyVMCountsPerNode |\n" +
   "where TimeStamp > ago(21d)\n" +
   "| project NodeId, VMCount, snapTime = TimeStamp \n" +
") on NodeId  \n" +
"| where (OldTime - snapTime) > -5m and (OldTime - snapTime) < 1440m\n" +
"| project NodeId, Tenant, OldTime, NewTime, oldState, newState, DurationInMin, \n" +
"delta = iff(OldTime < snapTime, 2000*(snapTime - OldTime)/1m, (OldTime - snapTime)/1m), VMCount \n" +
"| summarize argmin(delta, VMCount) by NodeId, Tenant, OldTime, NewTime, oldState, newState, DurationInMin\n" +
"| where min_delta_VMCount > 0 and DurationInMin < 120\n" +
"| extend durtnMultVMs = DurationInMin*min_delta_VMCount \n" +
"| join kind = leftouter \n" +
"(\n" +
    "CADDAILY | where StartTime > ago(20d)\n" +
    "| summarize by NodeId, Hardware_Model\n" +
")\n" +
"on NodeId\n" +
"| summarize sumdurtn = sum(durtnMultVMs), sumvms = sum(min_delta_VMCount) by newState, Hardware_Model\n" +
"| extend avgt = sumdurtn / sumvms * 60\n" +
"| where newState in (\"Ready\", \"Booting\", \"HumanInvestigate\", \"Dead\", \"Unhealthy\", \"PoweringOn\")")

readyTransitionsWHWQry = ("cluster(\"Azurecm\").database('AzureCM').NodeStateChangeDurationDetails | \n" +
"where PreciseTimeStamp > ago(10d)\n" +
"and oldState == \"Ready\" and Tenant contains \"prdapp\" \n " +
"| where newState in (\"Ready\", \"Booting\", \"HumanInvestigate\", \"Dead\", \"Unhealthy\", \"PoweringOn\", \"Recovering\")\n" +
"| project NodeId = nodeId, Tenant, PreciseTimeStamp, oldState, newState, durtn = todouble(stateDurationMilliseconds)/1000\n" +
"| extend OldTime = PreciseTimeStamp - durtn * 1s \n" +
"| project NodeId, Tenant, OldTime, NewTime = PreciseTimeStamp, \n" +
"oldState = iff(oldState == \"Recevering\", \"PoweringOn\", oldState), \n" +
"newState= iff(newState == \"Recovering\", \"PoweringOn\", newState), \n" +
"DurationInMin = durtn/60.0\n" +
"| where  not (DurationInMin < 0.1)\n" +
"| join kind = inner (\n" +
   "cluster('vmainsight').database('vmadb').DailyVMCountsPerNode |\n" +
   "where TimeStamp > ago(21d)\n" +
   "| project NodeId, VMCount, snapTime = TimeStamp \n" +
") on NodeId  \n" +
"| where (OldTime - snapTime) > -5m and (OldTime - snapTime) < 1440m\n" +
"| project NodeId, Tenant, OldTime, NewTime, oldState, newState, DurationInMin, \n" +
"delta = iff(OldTime < snapTime, 2000*(snapTime - OldTime)/1m, (OldTime - snapTime)/1m), VMCount \n" +
"| summarize argmin(delta, VMCount) by NodeId, Tenant, OldTime, NewTime, oldState, newState, DurationInMin\n" +
"| where min_delta_VMCount > 0 and DurationInMin < 120\n" +
"| extend durtnMultVMs = DurationInMin*min_delta_VMCount\n" +
"| join kind = leftouter \n" +
"(\n" +
    "CADDAILY | where StartTime > ago(20d)\n" +
    "| summarize by NodeId, Hardware_Model\n" +
")\n" +
"on NodeId\n" +
"| summarize sumdurtn = sum(durtnMultVMs), sumvms = sum(min_delta_VMCount) by newState, Hardware_Model\n" +
"| extend avgt = sumdurtn / sumvms * 60")
###################################################

hiCostWHWQry = ("cluster(\"Azurecm\").database('AzureCM').ServiceHealingTriggerEtwTable | where Tenant contains \"prdapp\" and TriggerType == \"Node\"\n" +
"and PreciseTimeStamp > ago(20d)\n" +
"| project shtime = PreciseTimeStamp, Cluster = Tenant, role = split(split(RoleInstanceName,\";\")[0], \":\")[1], TenantName\n" +
"| extend ResourceId = strcat(TenantName, \":\", role) \n" +
"| join kind=inner\n" +
"(\n" +
    "cluster('vmainsight').database('vmadb').VMALENS | where StartTime > ago(20d) \n" +
    "| project ResourceId, StartTime, \n" +
    "EndTime, DurationInMin, RCA, NodeId, Hardware_Model \n" +
")\n" +
"on ResourceId\n" +
"| where StartTime < shtime and shtime < EndTime\n" +
"| extend shdurtn = (EndTime - shtime)/1m\n" +
"| project shdurtn, Hardware_Model \n" +
"| summarize avg(shdurtn), count(), stdev(shdurtn) by Hardware_Model")



