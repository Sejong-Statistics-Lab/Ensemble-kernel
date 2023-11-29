import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, LogNormalAFTFitter, WeibullAFTFitter, LogLogisticAFTFitter
from lifelines import LogNormalFitter, WeibullFitter, LogLogisticFitter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.kernels import clinical_kernel

def split_data(df, testSize=0.33, randomState=None):
    data_0 = df[df.Status==0]
    data_1 = df[df.Status==1]

    x_train_0, x_test_0, target_train_0, target_test_0 = train_test_split(data_0.drop(['OS','Status'], axis = 1), data_0[['OS','Status']], test_size=testSize, random_state=randomState)
    x_train_1, x_test_1, target_train_1, target_test_1 = train_test_split(data_1.drop(['OS','Status'], axis = 1), data_1[['OS','Status']], test_size=testSize, random_state=randomState)

    x_train = pd.concat([x_train_0, x_train_1])
    x_test = pd.concat([x_test_0, x_test_1])
    target_train = pd.concat([target_train_0, target_train_1])
    target_test = pd.concat([target_test_0, target_test_1])

    x_train['OS'] = target_train['OS']
    x_train['Status'] = target_train['Status']
    x_test['OS'] = target_test['OS']
    x_test['Status'] = target_test['Status']
    
    return x_train, x_test, target_train, target_test

def prepare_response_variable(target):
    dt = np.dtype([('Status', np.bool_), ('OS', np.int64)])
    result = np.empty(shape=(len(target),), dtype=dt)

    for i in range(len(target)):
        result[i] = bool(target.iloc[i]['Status']), target.iloc[i]['OS']

    return result

def c_o(x1, x2):
    x_matrix = np.eye(N=len(x2), M=len(x1))
    x = pd.concat([x1,x2],axis=0, join='inner')
    d = np.max(x) - np.min(x)

    for i in range(len(x2)):
        for j in range(len(x1)):
            x_matrix[i,j] = (d-np.abs(x2.iloc[i]-x1.iloc[j]))/d
    return x_matrix

def nom(x1, x2):
    x_matrix = np.eye(N=len(x2), M=len(x1))
    for i in range(len(x2)):
        for j in range(len(x1)):
            if x2.iloc[i] == x1.iloc[j]:
                x_matrix[i,j] = 1
            else:
                x_matrix[i,j] = 0
    return x_matrix

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