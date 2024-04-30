# Malignant Melanoma data

## Result
* **Scenario1**

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.8775|0.0154|**0.8802**|0.0314|
|Clinical|0.8984|0.0251|0.8532|0.0258|
|Ensemble(Cox)|0.8795|0.0161|0.8750|0.0311|
|Ensemble(AFT)|0.8792|0.0161|0.8731|0.0315|

* **Scenario2(Cox)**

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.8776|0.0154|0.8804|0.3130|
|Clinical|0.8965|0.0284|0.8576|0.0264|
|Ensemble(Cox)|0.8843|0.0171|0.8821|0.0332|
|Ensemble(AFT)|0.8842|0.0159|**0.8824**|0.0323|

|**Remaining Variables**|year(100), tickness(50), ulcer(17), Age(100), Sex(100)|
|---|---|

* Scenario3(AFT)

|Kernel Function|<span style="color:red">mean(train C-index)</span>|<span style="color:red">sd(train C-index)</span>|<span style="color:green">mean(test C-index)</span>|<span style="color:green">sd(test C-index)</span>|
|:---|---:|---:|---:|---:|
|Linear|0.8775|0.0154|**0.8802**|0.0314|
|Clinical|0.8987|0.0315|0.8526|0.0271|
|Ensemble(Cox)|0.8803|0.0167|0.8780|0.0313|
|Ensemble(AFT)|0.8803|0.0162|0.8775|0.0320|

|**Remaining Variables**|year(100), tickness(47), ulcer(42), Age(100), Sex(100)|
|---|---|

## Source
The data were obtained from

Andersen P, Borgan O et al. (1993) Statistical Models Based on Counting Processes. Springer-Verlag.