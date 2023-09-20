# Ensemble-kernel

## Introduction
*When implementing SVM for a predictive survival model, Daemen et al. proposed a clinical kernel to equalize the influence of clinical variables and take account of the range of these variables.

*The clinical kernel uses the same weight for all clinical variables without considering the different effect of those variables.

*In this study, we proposed a simple ensemble kernel by combining a clinical kernel with model fitting.

*The ensemble kernel is formed through two steps:
    In the first step, we fit a Cox model or an AFT model and in the second step, we modify a clinical kernel by taking the weighted average of the effect of clinical variables from the fitted model.

*The performance of ensemble kernel is compared with those of linear kernel and clinical kernel by  the concordance index (C-index) of survival models for the four different datasets.