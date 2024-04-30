# Ensemble-kernel

## Introduction
*When implementing SVM for a predictive survival model, Daemen et al. proposed a clinical kernel to equalize the influence of clinical variables and take account of the range of these variables.

*The clinical kernel uses the same weight for all clinical variables without considering the different effect of those variables.

*In this study, we proposed a simple ensemble kernel by combining a clinical kernel with model fitting.

*The ensemble kernel is formed through two steps:
    In the first step, we fit a Cox model or an AFT model and in the second step, we modify a clinical kernel by taking the weighted average of the effect of clinical variables from the fitted model.

*The performance of ensemble kernel is compared with those of linear kernel and clinical kernel by  the concordance index (C-index) of survival models for the four different datasets.

## References
Daemen A, Moor De B. (2009) Development of a kernel function for clinical data. In: Proc of conf of IEEE Engineering in Medicine and Biology Society (EMBC). p. 5913–7.
Daemen A, Timmerman D, Bosch T, Bottomley C, Kirk E, Holsbeke C et al. (2012) Improved modeling of clinical data with kernel methods. Artificial Intelligence in Medicine. 54, p.103-114.
Harrel F, Califf R, Pryor D et al. (1982) Evaluating the yield of medical tests. Journal of American Medical Association. 247, p. 2543-2546.

## Source
The data vere obtained from

(pancreatic) Mok L, Kim Y, Lee S et al. HisCoM-PAGE: Hierarchical structureal component models for pathway analysis of gene expression data. 2019. Genes, 10, 931.

(veteran) Kalbfleisch D, Prentice R. (1980), The Statistical Analysis of Failure Time Data. Wiley, New York.

(lung) Loprinzi C, Laurie J, Wieand H et al. (1994) Prospective evaluation of prognostic variables from patient-completed questionnaires. North Central Cancer Treatment Group. Journal of Clinical Oncology. 12(3):601-7

(colon) Laurie J, Moertel C, Fleming T et al. (1989) Surgical adjuvant therapy of large-bowel carcinoma: An evaluation of levamisole and the combination of levamisole and fluorouracil. J Clinical Oncology, 7:1447-1456.

(icu.pneu) Beyersmann J. and Schumacher M. (2008) Time-dependent covariates in the proportional hazards model for competing risks. Biostatistics, 9:765--776.

(melanoma) Andersen P, Borgan O et al. (1993) Statistical Models Based on Counting Processes. Springer-Verlag.