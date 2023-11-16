import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines import LogNormalAFTFitter, WeibullAFTFitter, LogLogisticAFTFitter
from lifelines import WeibullFitter, LogNormalFitter, LogLogisticFitter

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import set_config
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sksurv.column import encode_categorical
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
from sksurv.kernels import clinical_kernel
from sklearn import metrics

from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from statsmodels.stats.outliers_influence import variance_inflation_factor

import copy
import time
import json
from skopt import BayesSearchCV

import sys
sys.path.append('G:\내 드라이브\대학\대외\2023\연구원\연구과제\Ensemble-Kernel')

from Ensemble_Kernel_New import split_data,prepare_response_variable, c_o, nom, new_kernel, train_predict_fastKernelSurvivalSVM_bayesian, train_predict_fastKernelSurvivalSVM_clinical_bayesian, one_hot_encode_columns, remove_high_vif_columns, convert_categorical_to_boolean, compare_kernels_with_abs_and_no_abs, train_predict_fastKernelSurvivalSVM_bayesian_no_abs, train_predict_fastKernelSurvivalSVM, train_predict_fastKernelSurvivalSVM_AFT, new_kernel_AFT, train_predict_fastKernelSurvivalSVM_bayesian_AFT, train_predict_fastKernelSurvivalSVM_clinical, train_predict_fastKernelSurvivalSVM_clinical_bayesian, C_index_fastKernelSurvivalSVM

def ensemble_cox_kernel(x1, x2=None, coef=None, keywords = ['Age', 'Sex']):
    if x2 is None:
        x2 = x1
        x = x1
    else:
        x = pd.concat([x1, x2], axis=0, join='inner')

    if coef is None:
        
        coxph = CoxPHFitter()
        coxph.fit(x, duration_col='OS', event_col='Status')
        coef = np.abs(np.log(coxph.hazard_ratios_))

        # Normalize coef to have a sum of 1
        coef = coef/(coef.sum())

    x_drop = x.drop(['Status', 'OS'], axis=1)

    remaining_variables = x_drop.columns.tolist()

    nominal_columns = x_drop.select_dtypes(include=['object', 'category', 'bool']).columns
    continuous_columns = x_drop.drop(nominal_columns, axis=1).columns

    sum_matrix = sum(coef[i] * (c_o(x1[i], x2[i]) if i in continuous_columns else nom(x1[i], x2[i])) for i in x_drop)
    mat = sum_matrix / sum(coef)

    return mat, coef, remaining_variables

def ensemble_AFT_kernel(x1, x2=None, coef=None, keywords = ['Age', 'Sex'], aft_model=None):
    if x2 is None:
        x2 = x1
        x = x1
    else:
        x = pd.concat([x1, x2], axis=0, join='inner')

    if coef is None:
        
        if aft_model == 'lognormal':
            lognormal_aft = LogNormalAFTFitter()
            aft_fit = lognormal_aft.fit(x, duration_col="OS", event_col="Status")
            coef = (((np.abs(aft_fit.summary['coef'])).drop(['sigma_'], axis=0, level=0)).drop(['Intercept'],axis=0,level=1)).droplevel(axis=0,level=0)
        elif aft_model == 'weibull':
            weibull_aft = WeibullAFTFitter()
            aft_fit = weibull_aft.fit(x, duration_col="OS", event_col="Status")
            coef = (((np.abs(aft_fit.summary['coef'])).drop(['rho_'], axis=0, level=0)).drop(['Intercept'],axis=0,level=1)).droplevel(axis=0,level=0)
        else:
            loglogistic_aft = LogLogisticAFTFitter()
            aft_fit = loglogistic_aft.fit(x, duration_col="OS", event_col="Status")
            coef = (((np.abs(aft_fit.summary['coef'])).drop(['beta_'], axis=0, level=0)).drop(['Intercept'],axis=0,level=1)).droplevel(axis=0,level=0)

        # Normalize coef to have a sum of 1
        coef = coef/sum(coef)

    x_drop = x.drop(['Status', 'OS'], axis=1)

    remaining_variables = x_drop.columns.tolist()

    nominal_columns = x_drop.select_dtypes(include=['object', 'category', 'bool']).columns
    continuous_columns = x_drop.drop(nominal_columns, axis=1).columns

    sum_matrix = sum(coef[i] * (c_o(x1[i], x2[i]) if i in continuous_columns else nom(x1[i], x2[i])) for i in x_drop)

    mat = sum_matrix / sum(coef)

    return mat, coef, remaining_variables


def c_index_kernel_type(x_train, y_train, x_test, y_test, param_grid, param_space, cv, keywords = ['Age', 'Sex'], type = 'ensemble_cox'):

    x_train_drop = x_train.drop(['OS','Status'],axis=1)
    x_test_drop = x_test.drop(['OS','Status'],axis=1)

    if type == 'ensemble_cox':
        # Create kernel using x_train
        train_kernel, coef, remaining_variables = ensemble_cox_kernel(x_train, coef=None, keywords = ['Age', 'Sex'])

        # Grid Optimization
        kssvm = FastKernelSurvivalSVM(optimizer = "rbtree", kernel = 'precomputed', max_iter = 1000, tol = 1e-6, random_state = 36)
        kgcv = GridSearchCV(kssvm, param_grid, n_jobs=-1, refit=True, cv=cv)

        # Fit model
        kgcv.fit(train_kernel, y_train)

        # Predict on x_train and calculate c_index
        train_pred = kgcv.predict(train_kernel)
        train_c_index = concordance_index_censored(y_train["Status"], y_train["OS"], train_pred)

        # Predict on x_test and calculate c_index
        test_kernel, _, _ = ensemble_cox_kernel(x_train, x_test, coef=coef, keywords=['Age', 'Sex'])
        test_pred = kgcv.predict(test_kernel)
        test_c_index = concordance_index_censored(y_test["Status"], y_test["OS"], test_pred)

    elif type == 'ensemble_aft':
        # 각 피팅 인스턴스화 
        wb = WeibullFitter() 
        log = LogNormalFitter() 
        loglogis = LogLogisticFitter()

        min_AIC = []

        # AIC를 최소로하는 모형 선택
        for model in [wb, log, loglogis]:
            model.fit(durations = x_train["OS"], event_observed = x_train["Status"])
            min_AIC.append(model.AIC_)

        if min_AIC.index(min(min_AIC)) == 0:
            model_type = 'weibull'

        elif min_AIC.index(min(min_AIC)) == 1:
            model_type = 'lognormal'

        else:
            model_type = 'loglogistic'

        # Create kernel using x_train
        train_kernel, coef, remaining_variables = ensemble_AFT_kernel(x_train, keywords = keywords, aft_model = model_type)

        # Grid Optimization
        kssvm = FastKernelSurvivalSVM(optimizer = "rbtree", kernel = 'precomputed', max_iter = 1000, tol = 1e-6, random_state = 36)
        kgcv = GridSearchCV(kssvm, param_grid, n_jobs=-1, refit=True, cv=cv)

        # Fit model
        kgcv.fit(train_kernel, y_train)

        # Predict on x_train and calculate c_index
        train_pred = kgcv.predict(train_kernel)
        train_c_index = concordance_index_censored(y_train["Status"], y_train["OS"], train_pred)

        # Predict on x_test and calculate c_index
        test_kernel, _, _= ensemble_AFT_kernel(x_train, x_test, coef=coef, keywords=keywords, aft_model=model_type)
        test_pred = kgcv.predict(test_kernel)
        test_c_index = concordance_index_censored(y_test["Status"], y_test["OS"], test_pred)

    elif type == 'clinical':
        # Create kernel using x_train_clinical
        train_kernel = clinical_kernel(x_train_drop)

        # Bayesian Optimization
        kssvm = FastKernelSurvivalSVM(optimizer = "rbtree", kernel = 'precomputed', max_iter = 1000, tol = 1e-6, random_state = 36)
        kgcv = GridSearchCV(kssvm, param_grid, n_jobs=-1, refit=True, cv=cv)

        # Fit model
        kgcv.fit(train_kernel, y_train)

        # Predict on x_train and calculate c_index
        train_pred = kgcv.predict(train_kernel)
        train_c_index = concordance_index_censored(y_train["Status"], y_train["OS"], train_pred)

        # Predict on x_test and calculate c_index
        test_kernel = clinical_kernel(x_test_drop[x_train_drop.columns], x_train_drop)
        test_pred = kgcv.predict(test_kernel)
        test_c_index = concordance_index_censored(y_test["Status"], y_test["OS"], test_pred)

        remaining_variables = x_train_drop.columns

    elif type == 'linear':
        # Grid Optimization
        kssvm = FastKernelSurvivalSVM(optimizer = "rbtree", kernel = 'linear', max_iter = 1000, tol = 1e-6, random_state=36)
        kgcv = GridSearchCV(kssvm, param_grid, n_jobs=-1, refit=True, cv=cv)

        # Fit model
        kgcv.fit(x_train_drop, y_train)

        # Optimal hyperparameters and c_index
        best_params = kgcv.best_params_
        best_c_index = kgcv.best_score_

        # Predict on x_train and calculate c_index
        train_pred = kgcv.predict(x_train_drop)
        train_c_index = concordance_index_censored(y_train["Status"], y_train["OS"], train_pred)

        # Predict on x_test and calculate c_index
        test_pred = kgcv.predict(x_test_drop)
        test_c_index = concordance_index_censored(y_test["Status"], y_test["OS"], test_pred)
        remaining_variables = x_train_drop.columns

    else:
        pass

    train_C_index = train_c_index[0]
    test_C_index = test_c_index[0]
    remaining_variables = remaining_variables

    return train_C_index, test_C_index, remaining_variables