from pipeline_utils import *
import pandas as pd
import numpy as np
import datetime
import re
import collections
import os
import seaborn as sns
import graphviz
import scikitplot as skplt
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import svm
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from sklearn import ensemble 
from sklearn import neighbors
import functools
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import matplotlib.lines as lines
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
get_ipython().magic('matplotlib inline')

# ============================ #
# READING AND EXPLORING DATA   #
# ============================ #

def read_data(filename):
    '''
    PURPOSE: To read a csv into a dataframe
    Input: filename
    Returns: pd dataframe
    '''
    assert os.path.isfile(filename), "File does not exist"
    df = pd.read_csv(filename)
    return df

def data_summary_stats(df, zparam=None, outlier_threshold=None, hist_draw=False, ptitle=None):
    '''
    PURPOSE: To produce target summary stats from the dataframe. 
    INPUT:
        df: dataframe
        zparam (float): zscore threshold to label outliers
        outlier_threshsold (int): how many places a feature must be an outlier in order
            to be considered an outlier overall
        hist_draw (bool): flag that dictates whether histograms for features are drawn
        
    RETURNS:
        outlier_dict (dict)
    '''
    # assume the first column is the identifying column for all dataframes
    print("Warning: assumes first column is a unique identifier for the df, " +
          "like an ID. Does not calculate all stats for the first column.\n\n")
    colnames = df.columns[1:]
    outlier_dict = {}
    ## updated here
    get_corr(df, ptitle=None)
    for col in colnames:
        ## UPDATED HERE
        get_column_dist(df, col)
        if df[col].dtype is (np.int64 or np.float64):
            newcol, outliers = get_zscore(df, col, zparam=zparam)
            if hist_draw:
                df.hist(column=newcol, bins=30)
            outlier_dict[newcol] = outliers
        
    return outlier_dict

def get_column_dist(df, col):
    '''
    PURPOSE: Provide summary descriptions columnwise for a dataframe
    Input: df (pd dataframe), col (str, column name of interest)
    '''
    print("Distributions for ", col)
    print(df[col].describe())
    print("\n\n")
    
def get_zscore(df, col, zparam=1.96):
    '''
    PURPOSE: Calculate z-score for data and then store as outliers for those with 
    more than a given z-score from mean
    
    INPUTS:
        df (pd dataframe)
        col (str) the column name
        zparam (float) the z-score to consider an outlier. Defaults to 1.96 (p=0.05)
        
    RETURNS:
        df (pd dataframe) that has been updated
        newcol (str) name of the new column
        outliers (list) list of indices pertaining to outlier set
    '''
    newcol = str(col) + "_zscore"
    currz = df[col].apply(lambda x: (x - df[col].mean()) / df[col].std())
    outliers = df.index[abs(currz) >= zparam ].tolist()
    return newcol, outliers
    
def get_corr(df, ptitle=None):
    '''
    PURPOSE: Generate correlation plot so user can easily visualize which columns 
        are most important
    INPUT: df (pd dataframe)
    '''
    ax = plt.axes()
    cols = [col for col in df.columns.tolist() if 'zscore' not in col]
    currcorr = df[cols].corr()
    sns.heatmap(currcorr, 
        xticklabels = currcorr.columns,
        yticklabels = currcorr.columns,
        vmin = -1.0, 
        vmax = 1.0,
        ax = ax,
        cmap = "RdBu")
    ax.set_title(ptitle)

# ============================ #
# PREPROCESSING DATA           #
# ============================ #

def deal_with_outliers(df, outlier_dict, deletion_option='none', delete_list=None):
    '''
    PURPOSE: Systematically handle outliers in the dataframe
    
    INPUTS:
        df (pd dataframe)
        outlier_dict (dictionary) as above, of column names to indices 
            that are outliers per coluMN
        deletion_option (str from set described below):
            all: delete all outliers contained in outlier_dict
            none: do not delete any outliers
            manual: delete outliers contained in 'delete_list'
            TO ADD: cap: cap all outliers at 2sd away from the mean
        delete_list (list): provide a list alongside "manual" in order to delete by index
    
    RETURNS:
        updated_df (pd dataframe)
    '''
    assert deletion_option in ["none", "all", "all_extreme", "manual"], "You've entered an invalid deletion option."
    if deletion_option == "none":
        return df
    
    elif deletion_option == "all":
        outlier_idx_list = get_all_outliers(outlier_dict)
    elif deletion_option == "manual":
        assert type(delete_list) == list, "You've entered an invalid idx list"
        outlier_idx_list = delete_list
        
    updated_df = df.drop(df.index[outlier_idx_list])
    
    return updated_df

def get_all_outliers(outlier_dict):
    '''
    PURPOSE: read in the outlier dict, formatted as below, in order to produce a list of
        outlier indices. 
    INPUT: outlier dict (dictionary) is 'feature': [idx1, idx2, idx3]
    RETURN: list of outlier idx
    '''
    outlier_idx_set = set()
    for feature, idx_list in outlier_dict.items():
        for i in idx_list:
            outlier_idx_set.add(i)
    return list(outlier_idx_set)

def fill_values(df, fill_missing_method='mean',  cols=None):
    '''
    PURPOSE: To fill in missing values while the dataframe is being preprocessed
    INPUTS:
        df (pd dataframe): dataframe to processs
        fill_missing_method (string): of options below, describes how to fill values
            mean: fill missing values with the mean per feature
            median: fill missing values with median
        columns (list of colnames): optional parameter to say which columns to fill
            
    RETURNS: new_df (pd dataframe) with filled values
    '''
    assert fill_missing_method in ['mean', 'median'], "You've entered an invalid method to fill missing values"
    
    cols = df.columns.tolist() if cols is None else cols
    new_df = df.copy()
    
    if fill_missing_method=="mean":
        new_df[cols] = df[cols].fillna(df[cols].mean())
        
    if fill_missing_method=="median":
        new_df[cols] = df[cols].fillna(df[cols].median())
        
    return new_df

def get_rid_of_no_index_col(df):
    '''
    PURPOSE: This function assumes that the first column is the index column and removes 
    all rows of a df that do not have that index. 
    INPUT: df (pandas dataframe)
    RETURN: new_df (pandas dataframe)
    '''
    new_df = df[df[df.columns[0]].notnull()]
    return new_df


# ============================ #
# UPDATE VALUES FOR FEATURES   #
# ============================ #

def make_discretized(df, var_of_interest, num_buckets=4):
    '''
    PURPOSE: Create a discretized variable from a continuous variable
    
    INPUTS: 
        df (pd dataframe)
        var_of_interest (str) colname for the feature you want to discretize
        num_buckets (int) the number of categories you want to create discretized 
            variables for
    
    RETURNS: df (pd dataframe) with appended categorical variable columns
    '''
    working_series = df[var_of_interest]
    new_col_name = str(var_of_interest) + '_category'
    df[new_col_name] = pd.cut(working_series, num_buckets)
    df_append = pd.get_dummies(data=df[new_col_name])
    df = pd.concat([df, df_append], axis=1)
    return df

def get_dummy(df, var_of_interest, threshold=None, thresh_type='>'):
    '''
    PURPOSE: Create a dummy variable from a continuous variable
    
    INPUTS: 
        df (pd dataframe)
        var_of_interest (str) colname for the feature you want to create a dummy for
        threshold (float) the value you want to use to create a dummy on
            !! Will default to the 75% percentile !!
        thresh_type (str from >, >=, <, <=) the type of comparison to your 
            threshold that you want
        
    RETURNS: df (pd dataframe) with appended dummy variable columns 
    '''
    assert thresh_type in ['>', '>=', '<', '<='], "Theshold type is invalid"
    working_series = df[var_of_interest]
    new_col_name = str(var_of_interest) + '_dummy'

    if threshold is None:
        threshold = np.percentile(working_series, 75)
    if thresh_type is '>':
        new_col = np.where(working_series > threshold, 1, 0)
    elif thresh_type is '>=':
        new_col = np.where(working_series >= threshold, 1, 0)
    elif thresh_type is '<':
        new_col = np.where(working_series < threshold, 1, 0)
    elif thresh_type is '<=':
        new_col = np.where(working_series <= threshold, 1, 0)
        
    df[new_col_name] = new_col
    
    return df

def make_categorical_dummy(df=None, cat=None):
    '''
    PURPOSE: transform a categorical variable into a series of dummies
    INPUTS: df (pd dataframe), cat (str colname of the categorical variable)
    RETURNS: df with appended categorical dummies columns
    '''
    df_append = pd.get_dummies(data=df[cat])
    df_append.columns = [str(cat) + str(col) for col in df_append.columns]
    df = pd.concat([df, df_append], axis=1)
    return df

def make_cols_strings(df):
    '''
    PURPOSE: ensure all column names are strings (not interval types)
    INPUTS and RETURNS: df (pd dataframe)
    '''
    df.columns = [str(col) for col in df.columns]
    return df

# ============================ #
# FEATURE SELECTION            #
# ============================ #

def sort_zip(t0, t1):
    '''
    CUSTOM SORT FUNCTION
    '''
    return t1[1] - t0[1]


def rf_features(df=None, var_excl=None, y_pred=None, n_jobs=10, random_state=0):
    '''
    PURPOSE: runs a random forest classifier for the sake of producing a list of features
        to include in a model 
        
    INPUTS: 
        df (pd dataframe): the dataframe to use
        var_excl (list of strings): a list of types of data to exclude
        y_pred (str): colname for the predicted variable
        n_jobs (int): sklearn default
        random_state (int): sklearn default
        
    RETURNS:
        sorted_coef_list (list): list of features to include in ranked order
    '''
    if var_excl is None:
        var_excl = ['object']
    model = Pipeline([
        ('scaler',StandardScaler()),
        ('clf',ensemble.RandomForestClassifier(n_jobs=n_jobs, random_state=random_state))
    ])  
    model.fit(df.select_dtypes(exclude=var_excl), df[y_pred])
    coef_list = list(zip(df.select_dtypes(exclude=var_excl), model.named_steps['clf'].feature_importances_))
    
    return elim_zero_coef(coef_list = coef_list)


def elim_zero_coef(coef_list=None):
    '''
    PURPOSE: eliminate all coefficients in a list with 0 influence
    INPUT: coef_list (list of coeffs)
    RETURNS: sorted feature list with only important coeffs (important means != 0)
    '''
    rl = []
    for (x, y) in coef_list:
        if y != 0:
            rl.append((x,y))
            
    return sorted(rl, key=functools.cmp_to_key(sort_zip))
    

def lcv_features(df=None, y_pred=None, var_excl=None, features=None):
    '''
    PURPOSE: runs a lasso  regression for the sake of producing a list of features
        to include in a model 
        
    INPUTS: 
        df (pd dataframe): the dataframe to use
        y_pred (str): colname for the predicted variable
        var_excl (list): datatypes to exclude
        features (list, optional): list of features to include
        
    RETURNS:
        sorted_coef_list (list): list of features to include in ranked order
    '''
    df_use = df.select_dtypes(exclude=var_excl)
    if features is None:
        features = []
        for x in df_use.columns:
            if x != y_pred:
                features.append(x)
    lcv = Pipeline([
        ('scaler',StandardScaler()),
        ('clf', LassoCV(tol=0.001))
    ])
    lcv.fit(df_use[features], df_use[y_pred])
    coef_list = list(zip(features, abs(lcv.named_steps['clf'].coef_)))
    return elim_zero_coef(coef_list = coef_list)


def rcv_features(df=None, y_pred=None, var_excl=None, features=None):
    '''
    PURPOSE: runs a ridge regression for the sake of producing a list of features
        to include in a model 
        
    INPUTS: 
        df (pd dataframe): the dataframe to use
        y_pred (str): colname for the predicted variable
        var_excl (list): datatypes to exclude
        features (list, optional): list of features to include
        
    RETURNS:
        sorted_coef_list (list): list of features to include in ranked order
    '''

    df_use = df.select_dtypes(exclude=var_excl)
    if features is None:
        features = []
        for x in df_use.columns:
            if x != y_pred:
                features.append(x)
    rcv = Pipeline([
        ('scaler',StandardScaler()),
        ('clf', RidgeCV())
    ])
    rcv.fit(df_use[features], df_use[y_pred])
    coef_list = list(zip(features, abs(rcv.named_steps['clf'].coef_)))
    return elim_zero_coef(coef_list = coef_list)

# ============================= #
# GENERATE TIME VALIDATION SETS #
# ============================= #

def generate_time_points(series_start, series_end, period):
    '''
    PURPOSE: Create a list of time intervals based on period.
    
    INPUTS:
        series_start (timestamp): the start of the time series
        series_end (timestamp): the end of the time series
        period (string e.g., '6M', '12M', '2M'): the time offset to use
        
    RETURNS:
        time_starts (date range): iterable list of timestamps
    '''
    time_starts=pd.date_range(start=series_start, end=series_end, freq=period)
    return time_starts


def generate_single_split(df, time_col, train_start, train_end, test_start, test_end):
    '''
    PURPOSE: Generate a single set of testing and training datasets (pair matched)
    
    INPUTS: 
        df (dataframe): the dataframe to use
        timecol (str): column name of the time column
        train_start (timestamp): time to start the training set (usually dataset start)
        train_end (timestamp): time to end the training set
        test_start (timestamp): time to start the testing set 
        test_end (timestamp): time to end the testing set
    
    RETURNS: test and train, two pair-matched dataframes
    '''
    test_mask = (df[time_col] >= test_start) & (df[time_col] < test_end)
    train_mask = (df[time_col] >= train_start) & (df[time_col] < train_end)
    test = df.loc[test_mask]
    train = df.loc[train_mask]
        
    return test, train

def wrap_single_split(df, time_col, times):
    '''
    PURPOSE: Create test / train time splits for an entire dataframe
    -- to add: ability to add a buffer time zone -- 
    
    INPUTS: 
        df (pd dataframe)
        time_col (str): colname of the time column
        times (date range iterable): list of time end points
        
    RETURNS:
        rv (dict) of the form { index: {'train_start': train_start, 'train_end': train_end, 
                                        'test_start': test_start, 'test_end': test_end, 
                                        'df_train': final training df, 'df_test': final testing df
                                        }
                               }
    '''
    rv = {}
    cycle_times = times[1:]
    train_start_temp = times[0]
    for i, train_end_temp in enumerate(cycle_times[:-1]):
        test_start_temp = train_end_temp
        test_end_temp = cycle_times[i+1]
        test, train = generate_single_split(df=df, time_col=time_col, 
                              train_start=train_start_temp, train_end=train_end_temp, 
                              test_start=train_end_temp, test_end=test_end_temp)
        rv[i] = {'train_start': train_start_temp, 'train_end': train_end_temp,
                'test_start' : test_start_temp, 'test_end': test_end_temp,
                'df_train': train, 'df_test': test}
    return rv

# ============================ #
# HYPERPARAMETER TUNING        #
# ============================ #

def get_param_tuning_splits(df=None, feature_list=None, y_column=None, random_state=0):
    '''
    PURPOSE: Create test / train splits for parameter tuning 
    
    INPUTS:
        df (pd dataframe)
        feature_list (list): list of colnames for features
        y_column (str): name of y_column to predict 
        random_state (int): seed, used for replication, defaults to 0
        
    RETURNS:
        X_train (df) scaled independent dataframe
        y_train (df) label df 
        X_test (df) scaled independent dataframe
        y_test (df) y test label df 
    '''
    preX_train, preX_test, y_train, y_test = train_test_split(df[feature_list],
                                                    df[y_column],random_state=random_state)

    scaler = StandardScaler().fit(preX_train)
    X_train = scaler.transform(preX_train)
    X_test = scaler.transform(preX_test)

    return X_train, y_train, X_test, y_test

def get_hyper_params(df=None, feature_list=None, y_column=None, base_models_dict=None, cv=5, n_iter=2, verbose=0,
                     random_state=0, model_list=None, scoring_mech='accuracy', hyperparam_tuning_dict=None):
    '''
    PURPOSE: Find hyper parameters through random search
    
    INPUTS:
        df (df)
        feature list (list)
        y_column (string)
        base_models_dict (dict): dict of the form {'model_type': model_call}
        cv (int): number of cv folds to use
        n_iter (int): number of iterations to try
        verbose (int): verbose level (0 is small, moves upwards)
        random_state (int): seed for duplication
        model_list (list): list of models to run hyperparameter tuning on
        scoring mechanism (str): type of score -- can be 'roc_auc', 'f1', etc. 
        hyperparam_tuning_dict (dict): dict that specifies parameters to tune per model
        
    RETURNS:
        final_hyperparams (dict): a dictionary of the optimal parameters per model type
    '''
    X_train, y_train, X_test, y_test = get_param_tuning_splits(
        df=df, feature_list=feature_list, y_column=y_column, random_state=random_state)
    
    final_hyperparams = {}
    for mod in model_list:
        temp = RandomizedSearchCV(estimator=base_models_dict[mod],
                               param_distributions = hyperparam_tuning_dict[mod], n_iter = n_iter, cv = cv,
                               verbose=2, random_state=random_state, scoring=scoring_mech)
        temp.fit(X_train, y_train)
        final_hyperparams[mod] = temp.best_params_
    
    return final_hyperparams

# ============================ #
# BUILD AND EVALUATE MODELS    #
# ============================ #

def cycle_through(time_dfs, clf_list, r, param_dict, features, y_column, threshold_list, type_list):
    '''
    PURPOSE: wrapper function to cycle through time splits
    
    INPUT:
        time_dfs (dict with dfs): the dictionary produced by the time splitter that has pairwise df matches
        clf_list (list): list of classifiers to include
        r (dict): the dictionary to populate with results
        param_dict (dict): parameter dictionary from hyper parameter tuning
        features (list): list of features to include
        y_column (str): predicted variable
        threshold_list (list): list of thresholds to predict 
        
    RETURNS:
        models (dict): dict of models, indexed by time split
        r (as a dataframe): table of results
    '''
    models = {}
    for i in time_dfs:
        temp = {}
        count = 0
        train_start, train_end = time_dfs[i]['train_start'], time_dfs[i]['train_end']
        test_start, test_end = time_dfs[i]['test_start'], time_dfs[i]['test_end']
        train, test = time_dfs[i]['df_train'], time_dfs[i]['df_test']
        
        if 'svm' in clf_list:
            print("running svm model...")
            r, yt, yp, model = add_svm_eval(train, test, features, y_column, param_dict, r)
            r = add_metrics_to_dict(r, yt, yp, type_list, threshold_list, True)
            r = update_dict_row(train_start, train_end, test_start, test_end, yt, r)
            temp['svm'] = {'model': model, 'y_pred': yp, 'y_test': yt}
            count=count+1
        
        if 'bagging' in clf_list:
            print("running bagging model...")
            r, yt, yp, model = add_bagging_eval(train, test, features, y_column, param_dict, r)
            r = add_metrics_to_dict(r, yt, yp, type_list, threshold_list)
            r = update_dict_row(train_start, train_end, test_start, test_end, yt, r)
            temp['bagging'] = {'model': model, 'y_pred': yp, 'y_test': yt}
            count=count+1   
        
        if 'boosting' in clf_list:
            print("running boosting model...")
            r, yt, yp, model = add_bagging_eval(train, test, features, y_column, param_dict, r)
            r = add_metrics_to_dict(r, yt, yp, type_list, threshold_list)
            r = update_dict_row(train_start, train_end, test_start, test_end, yt, r)
            temp['boosting'] = {'model': model, 'y_pred': yp, 'y_test': yt}
            count=count+1
            
        if 'knn' in clf_list:
            print("running knn model...")
            r, yt, yp, model = add_knn_eval(train, test, features, y_column, param_dict, r)
            r = add_metrics_to_dict(r, yt, yp, type_list, threshold_list)
            r = update_dict_row(train_start, train_end, test_start, test_end, yt, r)
            temp['knn'] = {'model': model, 'y_pred': yp, 'y_test': yt}
            count=count+1
            
        if 'logistic_regression' in clf_list:
            print("running logistic regression model...")
            r, yt, yp, model = add_logistic_eval(train, test, features, y_column, r)
            r = add_metrics_to_dict(r, yt, yp, type_list, threshold_list)
            r = update_dict_row(train_start, train_end, test_start, test_end, yt, r)
            temp['logistic_regression'] = {'model': model, 'y_pred': yp, 'y_test': yt}
            count=count+1
            
        if 'decision_tree' in clf_list:
            print("running decision tree model...")
            r, yt, yp, model = add_dt_eval(train, test, features, y_column, param_dict, r)
            r = add_metrics_to_dict(r, yt, yp, type_list, threshold_list)
            r = update_dict_row(train_start, train_end, test_start, test_end, yt, r)
            temp['decision_tree'] = {'model': model, 'y_pred': yp, 'y_test': yt}

        if 'random_forest' in clf_list:
            print("running random forest model...")
            r, yt, yp, model = add_rf_eval(train, test, features, y_column, param_dict, r)
            r = add_metrics_to_dict(r, yt, yp, type_list, threshold_list)
            r = update_dict_row(train_start, train_end, test_start, test_end, yt, r)
            temp['random_forest'] = {'model': model, 'y_pred': yp, 'y_test': yt}

        models['test_start ' + str(test_start)[:9]] = temp
        
    return pd.DataFrame(collections.OrderedDict(r)), models


#--------- running models --------------#

def run_bagging(train, test, features, y_col, param_dict):
    '''
    PURPOSE: make and run a bagging classifier
    
    INPUT:
        train (training df)
        test (testing df)
        features (list): list of features to include
        y_col (str): y column to predict 
        param_dict (dict): parameters found in hyper parameter optimization 
        
    RETURNS:
        y_test (series): y test series from testing set
        y_pred (proba array): predicted probabilities
        bag (classifier)
    '''
    X_train, X_test, y_train, y_test = train[features], test[features], train[y_col], test[y_col]
    if 'bagging' not in param_dict.keys():
        bag = Pipeline([('scaler', StandardScaler()), ('clf', ensemble.BaggingClassifier())])
    else:
        bag = Pipeline([
            ('scaler',StandardScaler()),
            ('clf',ensemble.BaggingClassifier(n_estimators=param_dict['bagging']['n_estimators'],
                                    max_samples=param_dict['bagging']['max_samples'],
                                    max_features=param_dict['bagging']['max_features']))
        ])
    bag.fit(X_train, y_train)
    y_pred = bag.predict_proba(X_test)   
    
    return y_test, y_pred, bag

def run_svm(train, test, features, y_col, param_dict):
    '''
    PURPOSE: make and run an SVM classifier
    
    INPUT:
        train (training df)
        test (testing df)
        features (list): list of features to include
        y_col (str): y column to predict 
        param_dict (dict): parameters found in hyper parameter optimization 
        
    RETURNS:
        y_test (series): y test series from testing set
        y_pred (proba array): predicted probabilities
        bag (classifier)
    '''
    X_train, X_test, y_train, y_test = train[features], test[features], train[y_col], test[y_col]
    sv = Pipeline([('scaler', StandardScaler()), ('clf', svm.SVC())])      
    # could add if I added SVM to hyperparameter search
    sv.fit(X_train, y_train)
    y_pred = sv.predict(X_test)  
    return y_test, y_pred, sv

def run_rf(train, test, features, y_col, param_dict):
    '''
    PURPOSE: make and run a rf classifier
    
    INPUT:
        train (training df)
        test (testing df)
        features (list): list of features to include
        y_col (str): y column to predict 
        param_dict (dict): parameters found in hyper parameter optimization 
        
    RETURNS:
        y_test (series): y test series from testing set
        y_pred (proba array): predicted probabilities
        bag (classifier)
    '''
    X_train, X_test, y_train, y_test = train[features], test[features], train[y_col], test[y_col]
    if 'random_forest' not in param_dict.keys():
        dt = Pipeline([('scaler', StandardScaler()), ('clf', ensemble.RandomForestClassifier())])
    rf = Pipeline([
        ('scaler',StandardScaler()),
        ('clf',ensemble.RandomForestClassifier(max_features=param_dict['random_forest']['max_features'],
                                    max_depth=param_dict['random_forest']['max_depth'],
                                    min_samples_split=param_dict['random_forest']['min_samples_split'],
                                    min_samples_leaf=param_dict['random_forest']['min_samples_leaf']))
    ])
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_test)   
    return y_test, y_pred, rf

def run_dt(train, test, features, y_col, param_dict):
    '''
    PURPOSE: make and run a decision tree classifier
    
    INPUT:
        train (training df)
        test (testing df)
        features (list): list of features to include
        y_col (str): y column to predict 
        param_dict (dict): parameters found in hyper parameter optimization 
        
    RETURNS:
        y_test (series): y test series from testing set
        y_pred (proba array): predicted probabilities
        bag (classifier)
    '''
    X_train, X_test, y_train, y_test = train[features], test[features], train[y_col], test[y_col]
    if 'decision_tree' not in param_dict.keys():
        dt = Pipeline([('scaler', StandardScaler()), ('clf', tree.DecisionTreeClassifier())])
    dt = Pipeline([
        ('scaler',StandardScaler()),
        ('clf',tree.DecisionTreeClassifier(max_features=param_dict['decision_tree']['max_features'],
                                    max_depth=param_dict['decision_tree']['max_depth'],
                                    min_samples_split=param_dict['decision_tree']['min_samples_split'],
                                    min_samples_leaf=param_dict['decision_tree']['min_samples_leaf']))
    ])
    dt.fit(X_train, y_train)
    y_pred = dt.predict_proba(X_test)   
    return y_test, y_pred, dt

def run_boosting(train, test, features, y_col, param_dict):
    '''
    PURPOSE: make and run a boosting classifier
    
    INPUT:
        train (training df)
        test (testing df)
        features (list): list of features to include
        y_col (str): y column to predict 
        param_dict (dict): parameters found in hyper parameter optimization 
        
    RETURNS:
        y_test (series): y test series from testing set
        y_pred (proba array): predicted probabilities
        bag (classifier)
    '''
    X_train, X_test, y_train, y_test = train[features], test[features], train[y_col], test[y_col]
    if 'boosting' not in param_dict.keys():
        boost = Pipeline([('scaler', StandardScaler()), ('clf', ensemble.AdaBoostClassifier())])
    boost = Pipeline([
        ('scaler',StandardScaler()),
        ('clf',ensemble.AdaBoostClassifier(n_estimators=param_dict['boosting']['n_estimators'],
                                    learning_rate=param_dict['boosting']['learning_rate']))
    ])
    boost.fit(X_train, y_train)
    y_pred = boost.predict_proba(X_test)   
    return y_test, y_pred, boost

def run_knn(train, test, features, y_col):
    '''
    PURPOSE: make and run a bagging classifier
    
    INPUT:
        train (training df)
        test (testing df)
        features (list): list of features to include
        y_col (str): y column to predict 
        
    RETURNS:
        y_test (series): y test series from testing set
        y_pred (proba array): predicted probabilities
        bag (classifier)
    '''
    X_train, X_test, y_train, y_test = train[features], test[features], train[y_col], test[y_col]
    knn = Pipeline([('scaler', StandardScaler()), ('clf', neighbors.KNeighborsClassifier())])
        
    # could add if I added KNN to hyperparameter search using param dict
    knn.fit(X_train, y_train)
    y_pred = knn.predict_proba(X_test)   
    return y_test, y_pred, knn

def run_logistic(train, test, features, y_col):
    '''
    PURPOSE: make and run a logistic regression classifier
    
    INPUT:
        train (training df)
        test (testing df)
        features (list): list of features to include
        y_col (str): y column to predict 
        
    RETURNS:
        y_test (series): y test series from testing set
        y_pred (proba array): predicted probabilities
        bag (classifier)
        par: parameters used
    '''
    X_train, X_test, y_train, y_test = train[features], test[features], train[y_col], test[y_col]
    # no need -- CV is included!
    logi = Pipeline([
        ('scaler',StandardScaler()),
        ('clf',linear_model.LogisticRegressionCV())
    ])
    logi.fit(X_train, y_train)
    y_pred = logi.predict_proba(X_test)   
    par = logi.named_steps['clf'].get_params()
    return y_test, y_pred, par, logi
   
#--------- evaluating models generally --------------#

def evaluate_clf(type_list=None, y_test=None, y_pred=None, threshold=None, svm=False):
    '''
    PURPOSE: evaluate classifiers given methods of evaluation
    
    INPUT:
        type_list (list): list of evaluation metric types
        y_test (series): the y col of the test dataset
        y_pred (probability array): the proba predicted by each model
        threshold (float): the threshold to convert probabilities by
        
    RETURNS:
        r (dict) the table
    '''
    rv = {} 
    
    count = 0 
    prec_flag = 0
   
    type_list = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall'] if type_list is None else type_list
    
    if svm:
        y_metric = y_pred
    
    else:
        y_metric = [1 if (i >= threshold) else 0 for i in y_pred[:,1]] 
        
    if 'accuracy' in type_list:
        accuracy = metrics.accuracy_score(y_test,y_metric)
        rv['accuracy at '+str(threshold)] = accuracy
        count = count + 1
        
    if 'f1' in type_list:
        f1 = metrics.f1_score(y_test,y_metric)
        rv['f1 at '+str(threshold)] = f1
        count = count + 1

    if 'precision' in type_list:
        precision = metrics.precision_score(y_test,y_metric)
        rv['precision at '+str(threshold)] = precision
        count = count + 1
        prec_flag = 1


    if 'recall' in type_list:
        recall = metrics.recall_score(y_test,y_metric)
        rv['recall at '+str(threshold)] = recall
        count = count + 1
        prec_flag = 2


    if 'roc_auc' in type_list:
        roc_auc = roc_auc_score(y_test, y_metric)
        rv['roc_auc at '+str(threshold)] = roc_auc
        count = count + 1
        
    if prec_flag == 2:
        precision_recall_curve(y_test, y_metric)
        
    assert count == len(type_list), "You seem to have included a type of metric that I cannot accomodate"
    
    return rv

#--------- setup model reporting --------------#

def setup_return_dict(threshold_list=None, type_list=None):
    '''
    PURPOSE: generate the dataframe to store results in 
    
    INPUT:
        threshold_list (list): list of thresholds for precision, recall, etc
    '''
    r = {'model':[], 'train_start':[], 'train_end':[], 'test_start':[], 'test_end':[], 'test_baseline':[], 'params': []}
    for t in threshold_list:
        for y in type_list:
            str_add = str(y) + " at " + str(t)
            r[str_add] = []
    return r

def add_metrics_to_dict(r, yt, yp, type_list, threshold_list, svm=False):
    '''
    PURPOSE: To add standard metrics to the dictionary table
    
    INPUT:
        r (dict)
        yt (series) y test
        yp (series) y predicted
        type_list (list) list of evaluation methods
        threshold_list (list) list of potential thresholds
        
    RETURNS:
        r (dict)
    '''
    for threshold in threshold_list:
        results = evaluate_clf(type_list=type_list, y_test=yt, y_pred=yp, threshold=threshold, svm=svm)
        for x in type_list:
            get_str = str(x) + " at "+str(threshold)
            if results[get_str]:
                r.get(get_str, []).append(results[get_str])
            else:
                r.get(get_str, []).append(0)     
    return r
 
def update_dict_row(train_start, train_end, test_start, test_end, yt, r):
    '''
    PURPOSE: Helper function to aid in creating table rows. 
    
    INPUT: 
        train_start (time)
        train_end (time)
        test_start (time)
        test_end (time)
        yt (y test series)
        r (dict)
        
    RETURNS:
        r (dict)
    '''
    r['train_start'].append(train_start)
    r['train_end'].append(train_end)
    r['test_start'].append(test_start)
    r['test_end'].append(test_end)
    r['test_baseline'].append(np.mean(yt))
    return r

def add_params_to_table(r, str_check, param_dict):
    '''
    PURPOSE: Helper function to see if params can be added from the param dict
    
    INPUTS:
        r (results dict)
        str_check (str): the str to search for
        param_dict (dict): the hyperparameter dictionary
        
    RETURNS:
        r (dict)
    '''
    if str_check in param_dict.keys():
        r['params'].append(param_dict[str_check])
    else:
        r['params'].append('default') 
    return r

#--------- adding models --------------#

def add_bagging_eval(train, test, features, y_column, final_params, r):
    '''
    PURPOSE: add evaluation for bagging
    
    INPUT:
        train (df)
        test (df)
        features (list)
        y_column (str)
        final_params (dict)
        r (dict)
        
    RETURNS:
        r (dict)
        yt (series of y test)
        yp (series of yp)
        model (clf)
    '''
    yt, yp, model = run_bagging(train, test, features, y_column, final_params)
    r['model'].append('bagging') 
    add_params_to_table(r, 'bagging', final_params)
    return r, yt, yp, model

def add_boosting_eval(train, test, features, y_column, final_params, r):
    '''
    PURPOSE: add evaluation for boosting
    
    INPUT:
        train (df)
        test (df)
        features (list)
        y_column (str)
        final_params (dict)
        r (dict)
        
    RETURNS:
        r (dict)
        yt (series of y test)
        yp (series of yp)
        model (clf)
    '''
    yt, yp, model = run_boosting(train, test, features, y_column, final_params)
    r['model'].append('boosting') 
    add_params_to_table(r, 'boosting', final_params)
    return r, yt, yp, model

def add_knn_eval(train, test, features, y_column, final_params, r):
    '''
    PURPOSE: add evaluation for knn
    
    INPUT:
        train (df)
        test (df)
        features (list)
        y_column (str)
        r (dict)
        
    RETURNS:
        r (dict)
        yt (series of y test)
        yp (series of yp)
        model (clf)
    '''
    yt, yp, model = run_knn(train, test, features, y_column)
    r['model'].append('knn') 
    add_params_to_table(r, 'knn', final_params)
    return r, yt, yp, model

def add_logistic_eval(train, test, features, y_column, r):
    '''
    PURPOSE: add evaluation for logistic
    
    INPUT:
        train (df)
        test (df)
        features (list)
        y_column (str)
        r (dict)
        
    RETURNS:
        r (dict)
        yt (series of y test)
        yp (series of yp)
        model (clf)
        par (parameters)
    '''
    yt, yp, par, model = run_logistic(train, test, features, y_column)
    r['model'].append('logistic_regression')
    r['params'].append(par)
    return r, yt, yp, model

def add_dt_eval(train, test, features, y_column, final_params, r):
    '''
    PURPOSE: add evaluation for decision tree
    
    INPUT:
        train (df)
        test (df)
        features (list)
        y_column (str)
        final_params (dict)
        r (dict)
        
    RETURNS:
        r (dict)
        yt (series of y test)
        yp (series of yp)
        model (clf)
    '''
    yt, yp, model = run_dt(train, test, features, y_column, final_params)
    r['model'].append('decision_tree') 
    add_params_to_table(r, 'decision_tree', final_params)
    return r, yt, yp, model

def add_rf_eval(train, test, features, y_column, final_params, r):
    '''
    PURPOSE: add evaluation for random forest
    
    INPUT:
        train (df)
        test (df)
        features (list)
        y_column (str)
        final_params (dict)
        r (dict)
        
    RETURNS:
        r (dict)
        yt (series of y test)
        yp (series of yp)
        model (clf)
    '''
    yt, yp, model = run_rf(train, test, features, y_column, final_params)
    r['model'].append('random_forest') 
    add_params_to_table(r, 'random_forest', final_params)
    return r, yt, yp, model

def add_svm_eval(train, test, features, y_column, final_params, r):
    '''
    PURPOSE: add evaluation for svm
    
    INPUT:
        train (df)
        test (df)
        features (list)
        y_column (str)
        final_params (dict)
        r (dict)
        
    RETURNS:
        r (dict)
        yt (series of y test)
        yp (series of yp)
        model (clf)
    '''
    yt, yp, model = run_svm(train, test, features, y_column, final_params)
    r['model'].append('svm') 
    add_params_to_table(r, 'svm', final_params)
    return r, yt, yp, model

# -------- PLOTTING ------------- #
def plot_precision_recall(model_dict=None, color_list=None):
    '''
    PURPOSE: makes precision recall plot
    '''
    for time in model_dict.keys():
        handles = []
        fig, ax = plt.subplots()

        for i, model in enumerate(model_dict[time].keys()):
            if model != 'svm':
                cmap=colors.ListedColormap(colors=color_list[i])
                yp, yt = model_dict[time][model]['y_pred'], model_dict[time][model]['y_test']
                t = 'Precisision_recall for ' + str(time)
                skplt.metrics.plot_precision_recall(yt, yp, classes_to_plot=[1], 
                                            plot_micro=False, cmap=cmap, ax=ax, title=t) 
                handles.append(lines.Line2D([], [], color=color_list[i], label=model))  
        plt.legend(handles=handles)