# Pancreatic data

## Source
The data were obtained from

Mok L, Kim Y, Lee S et al. HisCoM-PAGE: Hierarchical structureal component models for pathway analysis of gene expression data. 2019. Genes, 10, 931.

## Result
*Scenario1

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.6539|0.0782|0.6206|0.0655|
|Clinical|0.7648|0.0369|**0.6909**|0.0566|
|Ensemble(Cox)|0.7525|0.0392|0.6814|0.0556|
|Ensemble(AFT)|0.7542|0.0376|0.6794|0.0559|