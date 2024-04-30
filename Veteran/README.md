# Veteran lung cancer data

## Result
* **Scenario1**

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.7154|0.0170|**0.6998**|0.0344|
|Clinical|0.7909|0.0337|0.6967|0.0397|
|Ensemble(Cox)|0.7741|0.0318|0.6967|0.0383|
|Ensemble(AFT)|0.7760|0.0338|0.6960|0.0388|

* **Scenario2(Cox)**

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.7149|0.0169|**0.7008**|0.0350|
|Clinical|0.7605|0.0284|0.6942|0.0371|
|Ensemble(Cox)|0.7588|0.0255|0.6949|0.0371|
|Ensemble(AFT)|0.7582|0.0254|0.6945|0.0368|

|**Remaining Variables**|karno(100), celltype_0(59), celltype_1(69), celltype_2(43), diagtime(43), trt_2(35), prior_10(32), Age(100)|
|---|---|

* Scenario3(AFT)

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.7146|0.0173|**0.7002**|0.0342|
|Clinical|0.7607|0.0245|0.6965|0.0370|
|Ensemble(Cox)|0.7599|0.0228|0.6961|0.0392|
|Ensemble(AFT)|0.7592|0.0250|0.6954|0.0388|

|**Remaining Variables**|karno(100), celltype_0(62), celltype_1(78), celltype_2(52), diagtime(44), trt_2(46), prior_10(39), Age(100)|
|---|---|

## Source
The data were obtained from

Kalbfleisch D, Prentice R. (1980), The Statistical Analysis of Failure Time Data. Wiley, New York.