
allNSTQuery = ("cluster('Vmainsight').database('vmadbexp').NSTwVMA \n" +
	"| join kind = leftouter\n" +
	"(\n" +
	"    cluster('Vmainsight').database('vmadbexp').NSTwVMA | extend rownum = toint(rownum - 1)\n" +
	")\n" +
	"on rownum and NodeId and StartTime\n" +
	"| extend delta = (PreciseTimeStamp1 - PreciseTimeStamp)/1m\n" +
	"| project NewTime = PreciseTimeStamp, OldTime = PreciseTimeStamp1, \n" +
	"OldState = NewState, NewState = NewState1, delta, NodeId, vms, Hardware_Model, VMADurtn\n" +
	"| order by NodeId, NewTime asc\n" +
	"| where NewState == \"Unhealthy\" | summarize count() by OldState"
	"| where OldState in (\"PoweringOn\", \"Recovering\") and RCA !contains \"roothe\"\n" +
	"and isnotnull(OldTime) and NewState != \"Raw\"\n" +
	"| project NewState, delta, vms, Hardware_Model\n" +
	"| where delta < 50"
	)


###############################################################
## Other Kusto queries
###############################################################
## Nodes that have the new type of UD PXE.
"""
cluster('AzureDCM').database('AzureDCMDb').dcmInventoryComponentSystem
| extend NodeIdL = tolower(NodeId) 
| where BIOSVersion == "C1042.BS.1C10.AI1"  
| project DataCollectedOn, ClusterId, NodeIdL, BIOSVersion  
| join (cluster('AzureDCM').database('AzureDCMDb').ResourceSnapshotV1    
| where LifecycleState == "Production" 
| project ResourceId, LifecycleState, PfState , HealthGrade, FaultDescription, HealthSummary , PfRepairState 
) on $left.NodeIdL == $right.ResourceId
| project ClusterId, NodeIdL, LifecycleState, BIOSVersion 
| order by ClusterId asc  
| project NodeIdL
"""

