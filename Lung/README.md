# Lung cancer data

## Result
* **Scenario1**

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.5325|0.0168|0.5255|0.0389|
|Clinical|0.6678|0.0168|**0.6407**|0.0325|
|Ensemble(Cox)|0.6611|0.0201|0.6348|0.0339|
|Ensemble(AFT)|0.6619|0.0207|0.6347|0.0338|

* **Scenario2(Cox)**

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.5653|0.0415|0.5559|0.0455|
|Clinical|0.6670|0.0208|**0.6316**|0.0341|
|Ensemble(Cox)|0.6612|0.0197|0.6292|0.0348|
|Ensemble(AFT)|0.6622|0.0210|0.6282|0.0347|

|**Remaining Variables**|ph.ecog(74), pat.karno(81), ph.karno(58), wt.loss(54), meal.cal(51), Age(100), Sex(100)|
|---|---|

* Scenario3(AFT)

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.5616|0.0390|0.5598|0.0486|
|Clinical|0.6690|0.0229|**0.6304**|0.0369|
|Ensemble(Cox)|0.6623|0.0233|0.6268|0.0375|
|Ensemble(AFT)|0.6638|0.0230|0.6264|0.0362|

|**Remaining Variables**|ph.ecog(76), ph.karno(46), pat.karno(76), wt.loss(46), meal.cal(52), Age(100), Sex(100)|
|---|---|

## Source
The data were obtained from

Loprinzi C, Laurie J, Wieand H et al. (1994) Prospective evaluation of prognostic variables from patient-completed questionnaires. North Central Cancer Treatment Group. Journal of Clinical Oncology. 12(3):601-7