import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines import LogNormalAFTFitter
from lifelines import WeibullAFTFitter
from lifelines import LogLogisticAFTFitter
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

def one_hot_encode_columns(df, drop_first=True):
    df_encoded = copy.deepcopy(df)

    categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns

    # Ignore 'OS' column
    categorical_columns = categorical_columns.difference(['OS','Status'])

    # Apply one-hot encoding and convert data type to boolean
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, drop_first=drop_first)
    
    return df_encoded, categorical_columns

def convert_categorical_to_boolean(df_encoded_reduced, categorical_columns):
    for col in df_encoded_reduced.columns:
        if col.startswith(tuple(categorical_columns)) or col == 'Status':
            df_encoded_reduced[col] = df_encoded_reduced[col].astype(bool)

    return df_encoded_reduced   

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

def new_kernel(x1, x2=None, coef=None, drop=False, coef_drop=None, keywords = ['Age', 'Sex']):
    if x2 is None:
        x2 = x1
        x = x1
    else:
        x = pd.concat([x1, x2], axis=0, join='inner')

    if coef is None:
        
        coxph = CoxPHFitter()
        coxph.fit(x, duration_col='OS', event_col='Status')
        coef = np.abs(np.log(coxph.hazard_ratios_))

        # Calculate the sum of coef
        coef_sum = coef.sum()

        # Normalize coef to have a sum of 1
        coef = coef/coef_sum

    if drop:
        if coef_drop is None:
            coef_drop_by_pvalue = coxph.summary['p'][coxph.summary['p'] > 0.05].index

            keywords = keywords
            coef_drop = coef_drop_by_pvalue.drop([item for item in coef_drop_by_pvalue if any(keyword in item for keyword in keywords)])

        x = x.drop(coef_drop, axis=1)
        
        coxph_drop = CoxPHFitter()
        coxph_drop.fit(x, duration_col='OS', event_col='Status')
        coef = np.abs(np.log(coxph_drop.hazard_ratios_))

        coef_sum = coef.sum()
        coef = coef/coef_sum

    x_1 = x.drop(['Status', 'OS'], axis=1)

    remaining_variables = x_1.columns.tolist()

    if coef_drop is not None:
        remaining_variables = [var for var in remaining_variables if var not in coef_drop]
    
    x_1 = x_1[remaining_variables]

    nominal_columns = x_1.select_dtypes(include=['object', 'category', 'bool']).columns
    continuous_columns = x_1.drop(nominal_columns, axis=1).columns

    sum_matrix = sum(coef[i] * (c_o(x1[i], x2[i]) if i in continuous_columns else nom(x1[i], x2[i])) for i in x_1)

    mat = sum_matrix / sum(coef)

    return mat, coef, coef_drop, remaining_variables

def new_kernel_AFT(x1, x2=None, coef=None, coef_drop=None, drop=False, keywords = ['Age', 'Sex'], aft_model=None):
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
        weight = coef/sum(coef)

    if drop:
        if coef_drop is None:
            if aft_model == 'lognormal':
                lognormal_aft = LogNormalAFTFitter()
                aft_fit = lognormal_aft.fit(x, duration_col="OS", event_col="Status")
                p_value = (((aft_fit.summary['p']).drop(['sigma_'], axis=0, level=0)).drop(['Intercept'],axis=0,level=1)).droplevel(axis=0,level=0)
            elif aft_model == 'weibull':
                weibull_aft = WeibullAFTFitter()
                aft_fit = weibull_aft.fit(x, duration_col="OS", event_col="Status")
                p_value = (((aft_fit.summary['p']).drop(['rho_'], axis=0, level=0)).drop(['Intercept'],axis=0,level=1)).droplevel(axis=0,level=0)
            else:
                loglogistic_aft = LogLogisticAFTFitter()
                aft_fit = loglogistic_aft.fit(x, duration_col="OS", event_col="Status")
                p_value = (((aft_fit.summary['p']).drop(['beta_'], axis=0, level=0)).drop(['Intercept'],axis=0,level=1)).droplevel(axis=0,level=0)
        
            coef_drop_by_pvalue = p_value[p_value > 0.05].index

            keywords = keywords
            coef_drop = coef_drop_by_pvalue.drop([item for item in coef_drop_by_pvalue if any(keyword in item for keyword in keywords)])

        x = x.drop(coef_drop, axis=1)
        
        if aft_model is 'lognormal':
            lognormal_drop = LogNormalAFTFitter()
            aft_fit = lognormal_drop.fit(x, duration_col='OS', event_col='Status')
            coef = (((np.abs(aft_fit.summary['coef'])).drop(['sigma_'], axis=0, level=0)).drop(['Intercept'],axis=0,level=1)).droplevel(axis=0,level=0)
        elif aft_model is 'weibull':
            weibull_aft = WeibullAFTFitter()
            aft_fit = weibull_aft.fit(x, duration_col="OS", event_col="Status")
            coef = (((np.abs(aft_fit.summary['coef'])).drop(['rho_'], axis=0, level=0)).drop(['Intercept'],axis=0,level=1)).droplevel(axis=0,level=0)
        else:
            loglogistic_aft = LogLogisticAFTFitter()
            aft_fit = loglogistic_aft.fit(x, duration_col="OS", event_col="Status")
            coef = (((np.abs(aft_fit.summary['coef'])).drop(['beta_'], axis=0, level=0)).drop(['Intercept'],axis=0,level=1)).droplevel(axis=0,level=0)

        # Normalize coef to have a sum of 1
        weight = coef/sum(coef)

    x_1 = x.drop(['Status', 'OS'], axis=1)

    remaining_variables = x_1.columns.tolist()

    if coef_drop is not None:
        remaining_variables = [var for var in remaining_variables if var not in coef_drop]
    
    x_1 = x_1[remaining_variables]

    nominal_columns = x_1.select_dtypes(include=['object', 'category', 'bool']).columns
    continuous_columns = x_1.drop(nominal_columns, axis=1).columns

    sum_matrix = sum(coef[i] * (c_o(x1[i], x2[i]) if i in continuous_columns else nom(x1[i], x2[i])) for i in x_1)

    mat = sum_matrix / sum(coef)

    return mat, coef, coef_drop, remaining_variables

def compare_kernels_with_abs_and_no_abs(df, keywords = ['Age', 'Sex'], random_state=42):
    # 랜덤하게 n_samples명의 환자를 선택
    selected_df = df.sample(n=df.shape[0], random_state=random_state)

    # 선택된 환자 데이터 전치시키기
    transposed_df = selected_df.transpose()
    transposed_df = transposed_df.iloc[:, :3]
    def convert_bool_to_int(value):
        if isinstance(value, bool):
            return int(value)
        else:
            return value

    # 데이터프레임에 함수 적용
    transposed_df = transposed_df.applymap(convert_bool_to_int)
    pd.options.display.float_format = '{:.3f}'.format
    print("Transposed selected data:")
    print(pd.DataFrame(transposed_df))

    kernel_with_abs, _, _, _ = new_kernel(selected_df, keywords=keywords)
    kernel_with_abs = kernel_with_abs[:5, :5]
    print("\nKernel with absolute values:")
    print(pd.DataFrame(kernel_with_abs))

def train_predict_fastKernelSurvivalSVM(x_train, y_train, x_test, y_test, param_grid, keywords = ['Age', 'Sex'], drop=False, coef_drop=None,
                                            cv = KFold(n_splits = 5, shuffle=True, random_state=36)):
    # Create kernel using x_train
    train_kernel, coef, coef_drop, remaining_variables = new_kernel(x_train, drop=drop, keywords = keywords)

    # Grid Optimization
    kssvm = FastKernelSurvivalSVM(optimizer = "rbtree", kernel = 'precomputed', max_iter = 1000, tol = 1e-6, random_state = 36)
    kgcv = GridSearchCV(kssvm, param_grid, n_jobs=-1, refit=True, cv=KFold(n_splits = 5, shuffle=True, random_state=36))

    # Fit model
    kgcv.fit(train_kernel, y_train)

    # Optimal hyperparameters and c_index
    best_params = kgcv.best_params_
    best_c_index = kgcv.best_score_

    # Predict on x_train and calculate c_index
    train_pred = kgcv.predict(train_kernel)
    train_c_index = concordance_index_censored(y_train["Status"], y_train["OS"], train_pred)

    # Predict on x_test and calculate c_index
    test_kernel, _, _, _ = new_kernel(x_train, x_test, coef=coef, drop=drop, coef_drop=coef_drop, keywords=keywords)
    test_pred = kgcv.predict(test_kernel)
    test_c_index = concordance_index_censored(y_test["Status"], y_test["OS"], test_pred)

    return {
        "remaining variables" : remaining_variables, 
        "best_params": best_params,
        "best_c_index": best_c_index,
        "train_c_index": train_c_index[0],
        "test_c_index": test_c_index[0],
    }, coef_drop

def train_predict_fastKernelSurvivalSVM_clinical(x_train, y_train, x_test, y_test, param_grid, drop=False, coef_drop=None,
                                                        cv=KFold(n_splits = 5, shuffle=True, random_state=36)):
    # Remove 'OS' and 'Status' columns from x_train and x_test
    x_train_clinical = x_train.drop(columns=['OS', 'Status'])
    x_test_clinical = x_test.drop(columns=['OS', 'Status'])

    # Remove 'coef_drop' column if drop is True
    if drop and coef_drop is not None:
        x_train_clinical = x_train_clinical.drop(columns=coef_drop)
        x_test_clinical = x_test_clinical.drop(columns=coef_drop)

    remaining_variables = x_train_clinical.columns.tolist()

    # Create kernel using x_train_clinical
    train_kernel = clinical_kernel(x_train_clinical)
    test_kernel = clinical_kernel(x_test_clinical[x_train_clinical.columns], x_train_clinical)

    # Bayesian Optimization
    kssvm = FastKernelSurvivalSVM(optimizer = "rbtree", kernel = 'precomputed', max_iter = 1000, tol = 1e-6, random_state = 36)
    kgcv = GridSearchCV(kssvm, param_grid, n_jobs=-1, refit=True, cv=cv)

    # Fit model
    kgcv.fit(train_kernel, y_train)

    # Optimal hyperparameters and c_index
    best_params = kgcv.best_params_
    best_c_index = kgcv.best_score_

    # Predict on x_train and calculate c_index
    train_pred = kgcv.predict(train_kernel)
    train_c_index = concordance_index_censored(y_train["Status"], y_train["OS"], train_pred)

    # Predict on x_test and calculate c_index
    test_pred = kgcv.predict(test_kernel)
    test_c_index = concordance_index_censored(y_test["Status"], y_test["OS"], test_pred)

    return {
        "remaining variables" : remaining_variables,
        "best_params": best_params,
        "best_c_index": best_c_index,
        "train_c_index": train_c_index[0],
        "test_c_index": test_c_index[0],
    }

def train_predict_fastKernelSurvivalSVM_AFT(x_train, y_train, x_test, y_test, param_grid, aft_model=None, keywords = ['Age', 'Sex'], drop=False, coef_drop=None, cv = KFold(n_splits=5, shuffle=True, random_state=42)):
    # Create kernel using x_train
    train_kernel, coef, coef_drop, remaining_variables = new_kernel_AFT(x_train, drop=drop, keywords = keywords, aft_model=aft_model)

    # Grid Optimization
    kssvm = FastKernelSurvivalSVM(optimizer = "rbtree", kernel = 'precomputed', max_iter = 1000, tol = 1e-6, random_state = 36)
    kgcv = GridSearchCV(kssvm, param_grid, n_jobs=-1, refit=True, cv=cv)

    # Fit model
    kgcv.fit(train_kernel, y_train)

    # Optimal hyperparameters and c_index
    best_params = kgcv.best_params_
    best_c_index = kgcv.best_score_

    # Predict on x_train and calculate c_index
    train_pred = kgcv.predict(train_kernel)
    train_c_index = concordance_index_censored(y_train["Status"], y_train["OS"], train_pred)

    # Predict on x_test and calculate c_index
    test_kernel, _, _, _ = new_kernel_AFT(x_train, x_test, coef=coef, drop=drop, coef_drop=coef_drop, keywords=keywords, aft_model=aft_model)
    test_pred = kgcv.predict(test_kernel)
    test_c_index = concordance_index_censored(y_test["Status"], y_test["OS"], test_pred)

    return {
        "remaining variables" : remaining_variables, 
        "best_params": best_params,
        "best_c_index": best_c_index,
        "train_c_index": train_c_index[0],
        "test_c_index": test_c_index[0],
    }, coef_drop
    
def train_predict_fastKernelSurvivalSVM_random(x_train, y_train, x_test, y_test, param_grid, keywords = ['Age', 'Sex'], drop=False, coef_drop=None, random_state=None):
    
    results=pd.DataFrame()
    results['train_C_index']=[]
    results['test_C_index']=[]
    results['remaining_variables']=[]
    
    for i in range(0,100):
        # Create kernel using x_train
        train_kernel, coef, coef_drop, remaining_variables = new_kernel(x_train, drop=drop, keywords = keywords)

        # Grid Optimization
        kssvm = FastKernelSurvivalSVM(optimizer = "rbtree", kernel = 'precomputed', max_iter = 1000, tol = 1e-6, random_state = random_state[i])
        kgcv = GridSearchCV(kssvm, param_grid, n_jobs=-1, refit=True, cv=KFold(n_splits = 5, shuffle=True, random_state=random_state[i]))

        # Fit model
        kgcv.fit(train_kernel, y_train)

        # Optimal hyperparameters and c_index
        best_params = kgcv.best_params_
        best_c_index = kgcv.best_score_

        # Predict on x_train and calculate c_index
        train_pred = kgcv.predict(train_kernel)
        train_c_index = concordance_index_censored(y_train["Status"], y_train["OS"], train_pred)

        # Predict on x_test and calculate c_index
        test_kernel, _, _, _ = new_kernel(x_train, x_test, coef=coef, drop=drop, coef_drop=coef_drop, keywords=keywords)
        test_pred = kgcv.predict(test_kernel)
        test_c_index = concordance_index_censored(y_test["Status"], y_test["OS"], test_pred)

        results=results.append({"train_C_index":train_c_index[0],"test_C_index":test_c_index[0],"remaining_variables":remaining_variables},
                           ignore_index=True)
        
    return results   

def train_predict_fastKernelSurvivalSVM_random_AFT(x_train, y_train, x_test, y_test, param_grid, aft_model=None, keywords = ['Age', 'Sex'], drop=False, coef_drop=None, random_state=None):
    
    results=pd.DataFrame()
    results['train_C_index']=[]
    results['test_C_index']=[]
    results['remaining_variables']=[]
    
    for i in range(0,100):
        # Create kernel using x_train
        train_kernel, coef, coef_drop, remaining_variables = new_kernel_AFT(x_train, drop=drop, keywords = keywords, aft_model=aft_model)

        # Grid Optimization
        kssvm = FastKernelSurvivalSVM(optimizer = "rbtree", kernel = 'precomputed', max_iter = 1000, tol = 1e-6, random_state = random_state[i])
        kgcv = GridSearchCV(kssvm, param_grid, n_jobs=-1, refit=True, cv=KFold(n_splits = 5, shuffle=True, random_state=random_state[i]))

        # Fit model
        kgcv.fit(train_kernel, y_train)

        # Optimal hyperparameters and c_index
        best_params = kgcv.best_params_
        best_c_index = kgcv.best_score_

        # Predict on x_train and calculate c_index
        train_pred = kgcv.predict(train_kernel)
        train_c_index = concordance_index_censored(y_train["Status"], y_train["OS"], train_pred)

        # Predict on x_test and calculate c_index
        test_kernel, _, _, _ = new_kernel_AFT(x_train, x_test, coef=coef, drop=drop, coef_drop=coef_drop, keywords=keywords, aft_model=aft_model)
        test_pred = kgcv.predict(test_kernel)
        test_c_index = concordance_index_censored(y_test["Status"], y_test["OS"], test_pred)

        results=results.append({"train_C_index":train_c_index[0],"test_C_index":test_c_index[0],"remaining_variables":remaining_variables},
                           ignore_index=True)
        
    return results

def C_index_fastKernelSurvivalSVM(x_train, y_train, x_test, y_test, param_grid, keywords = ['Age', 'Sex'], drop=False, coef_drop=None,
                                            cv = KFold(n_splits = 5, shuffle=True, random_state=36)):
    # Create kernel using x_train
    train_kernel, coef, coef_drop, remaining_variables = new_kernel(x_train, drop=drop, keywords = keywords)

    # Grid Optimization
    kssvm = FastKernelSurvivalSVM(optimizer = "rbtree", kernel = 'precomputed', max_iter = 1000, tol = 1e-6, random_state = 36)
    kgcv = GridSearchCV(kssvm, param_grid, n_jobs=-1, refit=True, cv=KFold(n_splits = 5, shuffle=True, random_state=36))

    # Fit model
    kgcv.fit(train_kernel, y_train)

    # Optimal hyperparameters and c_index
    best_params = kgcv.best_params_
    best_c_index = kgcv.best_score_

    # Predict on x_train and calculate c_index
    train_pred = kgcv.predict(train_kernel)
    train_c_index = concordance_index_censored(y_train["Status"], y_train["OS"], train_pred)

    # Predict on x_test and calculate c_index
    test_kernel, _, _, _ = new_kernel(x_train, x_test, coef=coef, drop=drop, coef_drop=coef_drop, keywords=keywords)
    test_pred = kgcv.predict(test_kernel)
    test_c_index = concordance_index_censored(y_test["Status"], y_test["OS"], test_pred)

    train_C_index = train_c_index[0]
    test_C_index = test_c_index[0]
    remaining_variables = remaining_variables

    return train_C_index, test_C_index, remaining_variables