# Penumonia in ICU data

## Result
* **Scenario1**

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.5664|0.0068|0.5657|0.0137|
|Clinical|0.6448|0.0410|**0.5853**|0.0139|
|Ensemble(Cox)|0.6168|0.0172|0.5848|0.0141|
|Ensemble(AFT)|0.6190|0.0175|**0.5853**|0.0140|

* **Scenario2(Cox)**

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.5664|0.0068|0.5657|0.0137|
|Clinical|0.6409|0.0425|**0.5844**|0.0138|
|Ensemble(Cox)|0.6121|0.0172|**0.5844**|0.0141|
|Ensemble(AFT)|0.6123|0.0174|0.5842|0.0139|

|**Remaining Variables**|adm.cens.exit(100), pneu(97), event_3(38), Age(100), Sex(100)|
|---|---|

* Scenario3(AFT)

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.5664|0.0068|0.5657|0.0137|
|Clinical|0.6439|0.0430|**0.5847**|0.0145|
|Ensemble(Cox)|0.6110|0.0173|0.5844|0.0151|
|Ensemble(AFT)|0.6120|0.0169|0.5844|0.0141|

|**Remaining Variables**|adm.cens.exit(100), pneu(97), event_3(44), Age(100), Sex(100)|
|---|---|

## Source
The data were obtained from

Beyersmann J. and Schumacher M. (2008) Time-dependent covariates in the proportional hazards model for competing risks. Biostatistics, 9:765--776.