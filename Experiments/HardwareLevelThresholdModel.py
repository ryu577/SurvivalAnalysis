import matplotlib.pyplot as plt
import datetime
import numpy as np
import time
import pandas as pd
from MarkovChains import *
from GlobalThresholdModel import *
import uuid
from datetime import *
import time

'''
Constructs the matrices and calculates costs corresponding to a range of thresholds.
>> np.savetxt('p.csv', probs, delimiter=',')
'''
def constructFullMatricesByHW(level = 'Wiwynn Azure Gen5 1 Compute C1042H', plot=False):
  ## First get global vectors.
  [powOnProbsBase, powOnTimesBase] = populateRawVecs(powOnAllTrnstns[0], {'PoweringOn'})
  [powOnProbs, powOnTimes] = populateRawVecsForFeature(powOnAllTrnstnsWHWQry[0], {'PoweringOn'}, level, powOnProbsBase * 1.0, powOnTimesBase)
  [unhlProbsBase, unhlTimesBase] = populateRawVecs(unhlAllTrnstns[0], {'Unhealthy'})
  [unhlProbs, unhlTimes] = populateRawVecsForFeature(unhlAllTrnstnsWHWQry[0], {'Unhealthy'}, level, unhlProbsBase * 1.0, unhlTimesBase)
  [deadProbsBase, deadTimesBase] = populateRawVecs(deadAllTranstns[0], {'Dead'})
  [deadProbs, deadTimes] = populateRawVecsForFeature(deadAllTrnstnsWHWQry[0], {'Dead'}, level, deadProbsBase * 1.0, deadTimesBase)
  hiProbs = np.array([0,0,0,0,0,1])
  hiTimes = np.array([0,0,0,0,0,getHiCost(hiCostQuery[0])*60]) #####TODO!!
  [readyProbsBase, readyTimesBase] = populateRawVecs(readyQuery[0], {'Ready'})
  [readyProbs, readyTimes] = populateRawVecsForFeature(readyTransitionsWHWQry[0], {'Dead'}, level, readyProbsBase * 1.0, deadTimesBase)
  bootingWithTau = []
  for tau3 in np.arange(300, 940, 10):
      [bootingProbsBase, bootingTimesBase] = constructBootingVectors(tau3)
      [bootingProbs, bootingTimes] = constructBootingVectorsForFeature(tau3, level, bootingProbsBase * 1.0, bootingTimesBase)
      probs = np.matrix(np.array([unhlProbs, powOnProbs, bootingProbs, hiProbs, deadProbs, readyProbs]))
      times = np.matrix(np.array([unhlTimes, powOnTimes, bootingTimes, hiTimes, deadTimes, readyTimes]))
      timesToAbsorbing = TimeToAbsorbing(probs, times, 5)
      bootingToReady = timesToAbsorbing[2]
      bootingWithTau.append(bootingToReady)
  if plot:
    fig = plt.figure()
    plt.plot(np.arange(300, 940, 10), bootingWithTau)
    plt.savefig('..\\Plots\\BootingThresholdProfile\\' + level + '.png')
    plt.close(fig)
  return np.arange(300, 940, 10)[np.argmin(bootingWithTau)]


def executeBootingQueryByFeture(query):
    response = kusto_client.execute(kusto_database, query)
    data = response.fetchall();
    time.sleep(0.6)
    gammaprime = []
    T3Prime = []
    NewState = []
    T3 = []
    feature = []
    for row in data:
        for j in range(int(row[3])):
            gammaprime.append(row[0])
            T3Prime.append(row[1])
            NewState.append(row[2])
            feature.append(row[4])
            T3.append(row[5])
    return (np.array([np.array(gammaprime), np.array(T3Prime), np.array(T3)]), NewState, feature)

def constructBootingVectorsForFeature(tau3, level = 'Wiwynn Gen6 Optimized', baseCounts = np.zeros(6), baseTimes = np.zeros(6)):
    counts = np.zeros(6)
    times = np.zeros(6)
    if allBootingEventsWHWQry[1] is None:
        res = executeBootingQueryByFeture(allBootingEventsWHWQry[0])
        allBootingEventsWHWQry[1] = res
    else:
        res = allBootingEventsWHWQry[1]
    for i in range(len(res[1])):
        if res[2][i] == level:
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
              times += np.array([0,0,0,0,1.0*tau3,0])
          elif res[1][i] == 'HumanInvestigate':
              counts += np.array([0,0,0,1,0,0])
              times += np.array([0,0,0,T3Prime,0,0])
    times = (times + baseTimes * baseCounts) / (counts + baseCounts + 1e-9)
    counts += baseCounts
    return [counts / sum(counts), times]

def populateRawVecsForFeature(query, ignore = {}, level = 'Wiwynn Gen6 Optimized', baseCounts = np.zeros(6), baseTimes = np.zeros(6)):
    mapping = { 'Unhealthy' : 0, 'PoweringOn': 1, 'Booting' : 2, 'HumanInvestigate' : 3, 'Dead' : 4, 'Ready' : 5 }
    powOnCounts = np.zeros(6)
    powOnTimes = np.zeros(6)
    response = kusto_client.execute(kusto_database, query)
    data = response.fetchall()
    time.sleep(0.6)
    for row in data:
        if row[0] not in ignore and row[0] in mapping and row[1] == level:
            powOnCounts[mapping[row[0]]] = row[3]
            powOnTimes[mapping[row[0]]] = row[4]
    powOnTimes = (powOnTimes * powOnCounts + baseTimes * baseCounts) / (powOnCounts + baseCounts + 1e-9) # Add by a small number to avoid division by zero.
    powOnCounts += baseCounts
    return [powOnCounts/sum(powOnCounts), powOnTimes]


def thresholdsByHW():
  response = kusto_client.execute(kusto_database, allHWQuery)
  data = response.fetchall();
  time.sleep(0.6)
  i = 0
  for row in data:
    if i < 1e6:
      thresh = constructFullMatricesByHW(row['Hardware_Model'], True)
      print(row['Hardware_Model'] + "," + str(thresh))
      i += 1


def thresholdsByHWFinalTables():
  guid = str(uuid.uuid4())
  ver = datetime.now().year*1e10+datetime.now().month*1e8+datetime.now().day*1e6 + datetime.now().hour*3600 + datetime.now().minute*60 + datetime.now().second
  ver = str(int(ver))
  date = str(datetime.now())
  metadata = "PxeBootingTimeout,How much to wait from PXE for node to be Ready," + ver + ",Prod," + date + ",HardwareModel,OptimumThreshold," + guid

  response = kusto_client.execute(kusto_database, allHWQuery)
  data = response.fetchall()
  time.sleep(0.6)
  i = 0
  modeldata = ""
  for row in data:
    if i < 1e6:
      thresh = constructFullMatricesByHW(row['Hardware_Model'], True)
      modeldata = modeldata + guid + ",PxeBootingTimeout," + ver + ",Booting,HardwareModel," + row['Hardware_Model'] + ",OptimumThreshold," + str(max(600,thresh)) + "," + date + "\n"
      print(row['Hardware_Model'] + "," + str(thresh))
      i += 1
  metacsv = open("out/metadata.csv", "w")
  metacsv.write(metadata)
  metacsv.close()
  datcsv = open("out/modeldata.csv", "w")
  datcsv.write(modeldata)
  datcsv.close()


def tst():
  response = kusto_client.execute(kusto_database, allBootingEventsWHWQry[0])
  data = response.fetchall();
  time.sleep(0.6)
  for row in data:
    #print(row['Hardware_Model'])
    print(row[4])



################################
# Kusto queries
################################
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

allBootingEventsWHWQry = ("cluster(\"Azurecm\").database(\'AzureCM\').NodeStateChangeDurationDetails | \n" +
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
"| project gammaprime, T3Prime, NewState, ContainerCount = dcount_ResourceId, Hardware_Model, T3")

allBootingEventsWHWQry = [allBootingEventsWHWQry, None]

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
powOnAllTrnstnsWHWQry = [powOnAllTrnstnsWHWQry, None]

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
unhlAllTrnstnsWHWQry = [unhlAllTrnstnsWHWQry, None]

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
deadAllTrnstnsWHWQry = [deadAllTrnstnsWHWQry, None]

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
readyTransitionsWHWQry = [readyTransitionsWHWQry, None]
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
hiCostWHWQry = [hiCostWHWQry, None]


allHWQuery = ("CADDAILY | where StartTime > ago(20d) and isnotnull(Hardware_Model)\n" +
            "| summarize by Hardware_Model")


