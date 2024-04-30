# Pancreatic data

## Result
* **Scenario1**

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.6539|0.0782|0.6206|0.0655|
|Clinical|0.7648|0.0369|**0.6909**|0.0566|
|Ensemble(Cox)|0.7525|0.0392|0.6814|0.0556|
|Ensemble(AFT)|0.7542|0.0376|0.6794|0.0559|

* **Scenario2(Cox)**

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.6628|0.086|0.6305|0.0705|
|Clinical|0.7686|0.0425|**0.6775**|0.0604|
|Ensemble(Cox)|0.7635|0.0472|0.6653|0.0616|
|Ensemble(AFT)|0.7604|0.0467|0.6678|0.0625|

|**Remaining Variables**|T_Stage(43), alcohol_history_documented_YES(36), history_of_chronic_pancreatitis_YES(41), history_of_diabetes_YES(45), person_neoplasm_cancer_status_WITH TUMOR(87), postoperative_rx_tx_YES(95), N_Stage(50), tobacco_smoking_history(54), maximum_tumor_dimension(54), radiation_therapy_YES(52), residual_tumor_R1(24), Sex(100), Age(100)|

* Scenario3(AFT)

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.6560|0.0841|0.6220|0.0650|
|Clinical|0.7685|0.0366|**0.6811**|0.0541|
|Ensemble(Cox)|0.7603|0.0366|0.6709|0.0557|
|Ensemble(AFT)|0.7604|0.0344|0.6711|0.0573|

|**Remaining Variables**|T_Stage(51), alcohol_history_documented_YES(42), history_of_chronic_pancreatitis_YES(42), history_of_diabetes_YES(34), person_neoplasm_cancer_status_WITH TUMOR(87), postoperative_rx_tx_YES(99), N_Stage(49), tobacco_smoking_history(58), maximum_tumor_dimension(53), radiation_therapy_YES(57), residual_tumor_R1(26), Sex(100), Age(100)|

## Source
The data were obtained from

Mok L, Kim Y, Lee S et al. HisCoM-PAGE: Hierarchical structureal component models for pathway analysis of gene expression data. 2019. Genes, 10, 931.