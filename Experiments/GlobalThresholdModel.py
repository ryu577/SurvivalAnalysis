from kusto_client import KustoClient
import matplotlib.pyplot as plt
import datetime
import numpy as np
import time
from MarkovChains import *
import getpass

kusto_cluster = 'https://vmainsight.kusto.windows.net'
# In case you want to authenticate with AAD application.
passwd = getpass.getpass()
kusto_client = KustoClient(kusto_cluster=kusto_cluster, username = 'ropandey@microsoft.com', password = passwd)
kusto_database = 'vmadb'

def executeQuery(query):
    response = kusto_client.execute(kusto_database, query)
    data = response.fetchall();
    time.sleep(0.6)
    x = []
    y = []
    z = []
    for row in data:
        #print(row[0], row[1])
        x.append(row[0])
        y.append(row[1])
        for j in range( int(row[1])):
            z.append(row[0])
    return [np.array(x), np.array(z)]

def executeBootingQuery(query):
    response = kusto_client.execute(kusto_database, query)
    data = response.fetchall();
    time.sleep(0.6)
    gammaprime = []
    T3Prime = []
    NewState = []
    T3 = []
    for row in data:
        for j in range(int(row[3])):
            gammaprime.append(row[0])
            T3Prime.append(row[1])
            NewState.append(row[2])
            T3.append(row[4])
    return (np.array([np.array(gammaprime), np.array(T3Prime), np.array(T3)]), NewState)

def constructBootingVectors(tau3):
    counts = np.zeros(6)
    times = np.zeros(6)
    if allBootingEvents[1] is None:
        res = executeBootingQuery(allBootingEvents[0])
        allBootingEvents[1] = res
    else:
        res = allBootingEvents[1]
    for i in range(len(res[1])):
        [gammaprime, T3Prime, T3] = [res[0][0,i], res[0][1,i], res[0][2,i]]
        if res[1][i] == 'Ready':
            # If duration is less than tau3, move it to ready
            if T3 < tau3:
                counts += np.array([0,0,0,0,0,1])
                times += np.array([0,0,0,0,0,max(T3 - gammaprime, 0)]) #T3 - gammaprime = T3Prime.
            else:
                counts += np.array([0,1,0,0,0,0])
                times += np.array([0,max(tau3 - gammaprime,0),0,0,0,0])
        elif res[1][i] == 'PoweringOn':
            counts += np.array([0,1,0,0,0,0])
            times += np.array([0,max(tau3 - gammaprime,0),0,0,0,0])
        elif res[1][i] == 'Dead':
            counts += np.array([0,0,0,0,1,0])
            times += np.array([0,0,0,0,1.0*tau3,0]) ##################
        elif res[1][i] == 'HumanInvestigate':
            counts += np.array([0,0,0,1,0,0])
            times += np.array([0,0,0,T3Prime,0,0])
    counts1 = np.copy(counts)
    counts1[counts1 == 0] = 1
    return [counts / sum(counts), times / counts1]

def populateRawVecs(query, ignore = {}):
    mapping = { 'Unhealthy' : 0, 'PoweringOn': 1, 'Booting' : 2, 'HumanInvestigate' : 3, 'Dead' : 4, 'Ready' : 5 }
    powOnCounts = np.zeros(6)
    powOnTimes = np.zeros(6)
    response = kusto_client.execute(kusto_database, query)
    data = response.fetchall();
    time.sleep(0.6)
    for row in data:
        if row[0] not in ignore and row[0] in mapping:
            powOnCounts[mapping[row[0]]] += row[2]
            powOnTimes[mapping[row[0]]] = row[3]
    return [powOnCounts/sum(powOnCounts), powOnTimes]


def getHiCost(query):
    response = kusto_client.execute(kusto_database, query)
    data = response.fetchall();
    time.sleep(0.6)
    for row in data:
        return row[0]

'''
Constructs the matrices and calculates costs corresponding to a range of thresholds.
Savings matrices to view in excel is possible via - 
>> np.savetxt('p.csv', probs, delimiter=',')
'''
def constructFullMatrixForBootingThreshold(plot=False):
    [powOnProbs, powOnTimes] = populateRawVecs(powOnAllTrnstns[0], {'PoweringOn'})
    [unhlProbs, unhlTimes] = populateRawVecs(unhlAllTrnstns[0], {'Unhealthy'})
    [deadProbs, deadTimes] = populateRawVecs(deadAllTranstns[0], {'Dead'})
    hiProbs = np.array([0,0,0,0,0,1])
    hiTimes = np.array([0,0,0,0,0,getHiCost(hiCostQuery[0])*60])
    [readyProbs, readyTimes] = populateRawVecs(readyQuery[0], {'Ready'})
    bootingWithTau = []
    for tau3 in np.arange(300, 940, 10):
        [bootingProbs, bootingTimes] = constructBootingVectors(tau3)
        probs = np.matrix(np.array([unhlProbs, powOnProbs, bootingProbs, hiProbs, deadProbs, readyProbs]))
        times = np.matrix(np.array([unhlTimes, powOnTimes, bootingTimes, hiTimes, deadTimes, readyTimes]))
        timesToAbsorbing = TimeToAbsorbing(probs, times, 5)
        bootingToReady = timesToAbsorbing[2]
        bootingWithTau.append(bootingToReady)
    if plot:
        fig = plt.figure()
        plt.plot(np.arange(300, 940, 10), bootingWithTau)
        plt.savefig('..\\Plots\\BootingThresholdProfile\\' + 'global' + '.png')
        plt.close(fig)
    return np.arange(300, 940, 10)[np.argmin(bootingWithTau)]

def bootingVecs(tau3, tau1, p3 = 0.39):
    [x, z] = executeQuery(gammaquery)
    avgGamma = np.mean(z)
    [x, organic] = executeQuery(pxeBootOrganicQuery)
    [x, inorganic] = executeQuery(pxeBootCensoredQuery)
    p_T_gr_tau = (sum(organic > tau3) + sum(inorganic > 880)) / (len(organic) + sum(inorganic > 880))
    probs = [0, (1-p3)*p_T_gr_tau, 0, 0, p3*p_T_gr_tau, (1-p_T_gr_tau)]
    eTs = [0, tau3 + avgGamma - tau1, 0, 0, 3*(tau) - 180, np.mean(organic[organic < tau]) + avgGamma - tau1]
    return [np.array(probs), np.array(eTs)]

def poweringOnVecs(tau):
    [x, organic] = executeQuery(powOnOrganicQuery)
    [x, inorganic] = executeQuery(powOnCensoredQuery)
    p_T_gr_tau = (sum(organic > tau) + sum(inorganic > 880)) / (len(organic) + sum(inorganic > 880))
    probs = [0, 0, 0, p_T_gr_tau, 0, (1-p_T_gr_tau)]
    eTs = [0, 0, 0, tau, 0, np.mean(organic[organic < tau])]
    return [np.array(probs), np.array(eTs)]

################################
# Kusto queries
################################
allBootingEvents = ("NodeStateTransitions | where ContainerCount > 0 and PreciseTimeStamp > ago(20d)\n" +
"| where OldState in (\"Booting\")\n" +
"| summarize by DurationInSeconds, ContainerCount, NodeId, PreciseTimeStamp = bin(PreciseTimeStamp, 1s), Cluster, NewState\n" +
"| join kind = inner (\n"+
"cluster(\"Azurecm\").database('AzureCM').TMMgmtNodeEventsEtwTable\n" +
"| where PreciseTimeStamp >= ago(20d)\n" +
"and Message contains \"PxeInfo suggests the Node must be booting. Based on HealthTimeoutsRebooting and lastPxeReqArrivalTime\" and Tenant contains \"prdapp\"\n" +
"| project pxetime = bin(PreciseTimeStamp,1s) , Tenant , NodeId , Message\n" +
"| extend additionalt = todouble(extract(\"let's give it ([0-9]+)\", 1, Message))\n" +
"| where additionalt <= 900| extend trupxetime = pxetime - (900 - additionalt)*1s) on NodeId\n" +
"| extend delta = iff(pxetime - 20m < PreciseTimeStamp and PreciseTimeStamp < pxetime + 20m, 1, 0)\n" +
"| where (isnull(pxetime) or delta > 0)\n" +
"| project DurationInSeconds, NodeId, bin(PreciseTimeStamp,1s), bin(pxetime,1s), Cluster, ContainerCount, additionalt, \n" +
"trupxetime, pxeToNew = (PreciseTimeStamp - trupxetime)/1s , NewState\n" +
"| extend pxeToBooting = -DurationInSeconds + pxeToNew\n" +
"| where pxeToNew > 0\n" +
"| summarize argmin(pxeToNew, additionalt, pxeToBooting) by DurationInSeconds, NodeId, bin(PreciseTimeStamp,1s), ContainerCount, NewState\n" +
"| project gammaprime = min_pxeToNew_pxeToBooting, T3Prime = DurationInSeconds, NewState, ContainerCount\n" +
"| extend T3 = gammaprime + T3Prime")

allBootingEvents = ("cluster(\"Azurecm\").database(\'AzureCM\').NodeStateChangeDurationDetails | \n" +
"where PreciseTimeStamp > ago(20d)\n" +
"and oldState in (\"Booting\") and Tenant contains \"prdapp\"\n" +
"| extend NodeId = nodeId\n" +
"| extend DurationInSeconds = todouble(stateDurationMilliseconds)/1000\n" +
"| join kind=inner\n" +
"(\n" +
"    cluster(\"Azurecm\").database(\'AzureCM\').TMMgmtNodeEventsEtwTable\n" +
"    | where PreciseTimeStamp >= ago(20d)\n" +
"    and Message contains \"PxeInfo suggests the Node must be booting. Based on HealthTimeoutsRebooting and lastPxeReqArrivalTime\" and Tenant contains \"prdapp\"\n" +
"    | project pxetime = bin(PreciseTimeStamp,1s) , Tenant , NodeId , Message\n" +
"    | extend additionalt = todouble(extract(\"let's give it ([0-9]+)\", 1, Message))\n" +
"    | where additionalt <= 900 | extend trupxetime = pxetime - (900 - additionalt)*1s \n" +
")\n" +
"on NodeId\n" +
"| where trupxetime - 20m < PreciseTimeStamp and PreciseTimeStamp < trupxetime + 20m\n" +
"| summarize by bin(PreciseTimeStamp,1ms), NodeId, oldState, newState, DurationInSeconds, additionalt, trupxetime, pxeToNew = (PreciseTimeStamp - trupxetime)/1s\n" +
"| extend pxeToBooting = pxeToNew - DurationInSeconds \n" +
"| where pxeToNew > 0 and pxeToBooting > 0\n" +
"| summarize argmin(pxeToNew, additionalt, pxeToBooting) by DurationInSeconds, NodeId, bin(PreciseTimeStamp, 1ms), newState\n" +
"| project gammaprime = min_pxeToNew_pxeToBooting, T3Prime = DurationInSeconds, NewState = newState, NodeId, PreciseTimeStamp\n" +
"| extend T3 = gammaprime + T3Prime\n" +
"| join kind=inner\n" +
"(\n" +
"    VMALENS | where StartTime > ago(20d) | project NodeId, StartTime, EndTime, DurationInMin, Hardware_Model, ResourceId\n" +
")\n" +
"on NodeId\n" +
"| where StartTime < PreciseTimeStamp and PreciseTimeStamp < EndTime \n" +
"| summarize count(), dcount(ResourceId) by gammaprime, T3Prime, NewState, Hardware_Model, T3, NodeId\n" +
"| project gammaprime, T3Prime, NewState, ContainerCount = dcount_ResourceId, Hardware_Model, T3\n" +
"| project-away Hardware_Model")
allBootingEvents = [allBootingEvents, None]

gammaquery = ("NodeStateTransitions | where ContainerCount > 0 and PreciseTimeStamp > datetime(11-19-2017)\n" + 
"| where OldState == \"Unhealthy\" and NewState == \"Booting\" \n" +
"| summarize by DurationInSeconds, ContainerCount, NodeId, PreciseTimeStamp = bin(PreciseTimeStamp, 1s), Cluster \n" +
"| join kind = inner ( \n" +
    "cluster(\"Azurecm\").database('AzureCM').TMMgmtNodeEventsEtwTable \n" +
    "| where PreciseTimeStamp >= datetime(11-19-2017) \n" +
    "and Message contains \"PxeInfo suggests the Node must be booting. Based on HealthTimeoutsRebooting and lastPxeReqArrivalTime\"\n" +
    "and Tenant contains \"prdapp\" \n" +
    "| project pxetime = bin(PreciseTimeStamp,1s) , Tenant , NodeId , Message \n" +
    "| extend additionalt = todouble(extract(\"let's give it ([0-9]+)\", 1, Message)) \n" +
") on NodeId \n" +
"| extend delta = iff(pxetime - 15m < PreciseTimeStamp and PreciseTimeStamp < pxetime + 15m, 1, 0) \n" +
"| where (isnull(pxetime) or delta > 0) \n" +
"| project DurationInSeconds, NodeId, bin(PreciseTimeStamp,1s), bin(pxetime,1s), Cluster, ContainerCount, additionalt, Message \n" +
"| where additionalt <= 900 \n" +
"| project additionalt + DurationInSeconds - 900, ContainerCount")
gammaquery = [gammaquery, None]

pxeBootOrganicQuery = ("NodeStateTransitions | where ContainerCount > 0 and PreciseTimeStamp > ago(20d)\n" +
"| where OldState in (\"Booting\") and NewState == \"Ready\"\n" +
"| summarize by DurationInSeconds, ContainerCount, NodeId, PreciseTimeStamp = bin(PreciseTimeStamp, 1s), Cluster\n" +
"| join kind = inner (\n" +
    "cluster(\"Azurecm\").database('AzureCM').TMMgmtNodeEventsEtwTable \n" +
    "| where PreciseTimeStamp >= datetime(11-19-2017) \n" + 
    "and Message contains \"PxeInfo suggests the Node must be booting. Based on HealthTimeoutsRebooting and lastPxeReqArrivalTime\"" +
    "and Tenant contains \"prdapp\"\n" +
    "| project pxetime = bin(PreciseTimeStamp,1s) , Tenant , NodeId , Message\n" +
    "| extend additionalt = todouble(extract(\"let's give it ([0-9]+)\", 1, Message))\n" +
    "| where additionalt <= 900" +
    "| extend trupxetime = pxetime - (900 - additionalt)*1s" +
") on NodeId\n" + 
"| extend delta = iff(pxetime - 20m < PreciseTimeStamp and PreciseTimeStamp < pxetime + 20m, 1, 0)\n" + 
"| where (isnull(pxetime) or delta > 0)\n" +
"| project DurationInSeconds, NodeId, bin(PreciseTimeStamp,1s), bin(pxetime,1s), Cluster, ContainerCount, additionalt, trupxetime, pxeToReady = (PreciseTimeStamp - trupxetime)/1s\n" +
"| where pxeToReady > 0\n" +
"| summarize min(pxeToReady) by DurationInSeconds, NodeId, PreciseTimeStamp, ContainerCount\n" +
"| project min_pxeToReady, ContainerCount")
pxeBootOrganicQuery = [pxeBootOrganicQuery, None]

## Note - this is not PXE to powering on durations. But it doesn't matter since we only use the count.
pxeBootCensoredQuery = ("NodeStateTransitions | where ContainerCount > 0 and PreciseTimeStamp > ago(20d)\n" + 
"| where OldState == \"Booting\" and NewState in (\"PoweringOn\", \"Dead\") \n" +
"| summarize by DurationInSeconds, ContainerCount, NodeId, PreciseTimeStamp = bin(PreciseTimeStamp, 1s), Cluster \n" +
"| join kind = inner ( \n" +
    "cluster(\"Azurecm\").database('AzureCM').TMMgmtNodeEventsEtwTable \n" +
    "| where PreciseTimeStamp >= datetime(11-19-2017) \n" +
    "and Message contains \"PxeInfo suggests the Node must be booting. Based on HealthTimeoutsRebooting and lastPxeReqArrivalTime\"\n" +
    "and Tenant contains \"prdapp\" \n" +
    "| project pxetime = bin(PreciseTimeStamp,1s) , Tenant , NodeId , Message \n" +
    "| extend additionalt = todouble(extract(\"let's give it ([0-9]+)\", 1, Message)) \n" +
") on NodeId \n" +
"| extend delta = iff(pxetime - 15m < PreciseTimeStamp and PreciseTimeStamp < pxetime + 15m, 1, 0) \n" +
"| where (isnull(pxetime) or delta > 0) \n" +
"| project DurationInSeconds, NodeId, bin(PreciseTimeStamp,1s), bin(pxetime,1s), Cluster, ContainerCount, additionalt, Message \n" +
"| project DurationInSeconds, ContainerCount")
pxeBootCensoredQuery = [pxeBootCensoredQuery, None]

powOnOrganicQuery = ("NodeStateTransitions | where ContainerCount > 0 and PreciseTimeStamp > ago(20d)\n" +
"| where OldState in (\"PoweringOn\", \"Recovering\") and NewState == \"Ready\"\n" +
"| summarize by DurationInSeconds, ContainerCount, NodeId, PreciseTimeStamp = bin(PreciseTimeStamp, 1s), Cluster\n" +
"| project DurationInSeconds, ContainerCount")
powOnOrganicQuery = [powOnOrganicQuery, None]

powOnCensoredQuery = ("NodeStateTransitions | where ContainerCount > 0 and PreciseTimeStamp > datetime(11-19-2017)\n" +
"| where OldState in (\"PoweringOn\", \"Recovering\") and NewState == \"HumanInvestigate\"\n" +
"| summarize by DurationInSeconds, ContainerCount, NodeId, PreciseTimeStamp = bin(PreciseTimeStamp, 1s), Cluster\n" +
"| project DurationInSeconds, ContainerCount")
powOnCensoredQuery = [powOnCensoredQuery, None]

powOnAllTrnstns = ("cluster(\"Azurecm\").database('AzureCM').NodeStateChangeDurationDetails | \n" +
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
"| summarize sumdurtn = sum(durtnMultVMs), sumvms = sum(min_delta_VMCount) by newState\n" +
"| extend avgt = sumdurtn / sumvms * 60\n" +
"| where newState in (\"Ready\", \"Booting\", \"HumanInvestigate\", \"Dead\", \"Unhealthy\")")
powOnAllTrnstns = [powOnAllTrnstns, None]

unhlAllTrnstns = ("cluster(\"Azurecm\").database('AzureCM').NodeStateChangeDurationDetails | \n" +
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
"| summarize sumdurtn = sum(durtnMultVMs), sumvms = sum(min_delta_VMCount) by newState\n" +
"| extend avgt = sumdurtn / sumvms * 60\n" +
"| where newState in (\"Ready\", \"Booting\", \"HumanInvestigate\", \"Dead\", \"Unhealthy\", \"PoweringOn\", \"Recovering\")")
unhlAllTrnstns = [unhlAllTrnstns, None]

deadAllTranstns = ("cluster(\"Azurecm\").database('AzureCM').NodeStateChangeDurationDetails | \n" +
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
"| summarize sumdurtn = sum(durtnMultVMs), sumvms = sum(min_delta_VMCount) by newState\n" +
"| extend avgt = sumdurtn / sumvms * 60\n" +
"| where newState in (\"Ready\", \"Booting\", \"HumanInvestigate\", \"Dead\", \"Unhealthy\", \"PoweringOn\")")
deadAllTranstns = [deadAllTranstns, None]

hiCostQuery = ("cluster(\"Azurecm\").database('AzureCM').ServiceHealingTriggerEtwTable | where Tenant contains \"prdapp\" and TriggerType == \"Node\"\n" +
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
"| summarize avg(shdurtn), count(), stdev(shdurtn)")
hiCostQuery = [hiCostQuery, None]

readyQuery = ("cluster(\"Azurecm\").database('AzureCM').NodeStateChangeDurationDetails | \n" +
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
"| summarize sumdurtn = sum(durtnMultVMs), sumvms = sum(min_delta_VMCount) by newState\n" +
"| extend avgt = sumdurtn / sumvms * 60")
readyQuery = [readyQuery, None]

