# -*- coding: utf-8 -*-

__author__ = "Boutros El-Gamil"
__copyright__ = "Copyright 2021"
__version__ = "0.1"
__maintainer__ = "Boutros El-Gamil"
__email__ = "contact@data-automaton.com"
__status__ = "Development"  

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

pd.options.mode.chained_assignment = None

def import_csv_data(path, file, header_arg, separator):
    '''
    this function imports data from .csv file into pandas dataframe

    Parameters
    ----------
    path : string
        data path
    file : string
        file name
    header_arg : string
        type of header inferring
    separator : string
        columns separator

    Returns
    -------
    pandas dataframe

    '''
    print("import data file: ", file)
    try:
        # set file path
        file_path= os.path.join(path, file)
        df= pd.read_csv(file_path, header=header_arg, sep=separator)
        print("done!")
        
    except Exception as e:
        print(e)
        
    # returns dataframe corresponds to csv file
    return df
    
def name_unnamed_cols(df, col_idx, names):
    '''
    this function adds names to unnamed cols in a dataframe
    
    Parameters
    ----------
    df : pandas df
        dataframe
    col_idx : list
        list of unnamed cols indeces
    names : list
        list of cols names

    Returns
    -------
    pandas dataframe
    '''
    
    print("name unnamed columns in raw df... ")
    try:
        for i in range(0,len(col_idx)):        
            df.columns.values[col_idx[i]] = names[i] 
        
        print("done!")
        
    except Exception as e:
        print(e)
    return df

def merge_data(df1, df2, cols):
    '''
    this function merges 2 dataframes into one using list of common columns

    Parameters
    ----------
    df1 : pandas df
        first dataframe.
    df2 : pandas df
        second dataframe.
    cols : list
        list of common columns

    Returns
    -------
    pandas df
        merged dataframe.

    '''
    
    print("merge raw & target dfs... ")
    try:
        df = pd.merge(df1, df2, on=(cols))
        print("done!")
        
    except Exception as e:
        print(e)
        
    return df

def split_data(df, col, n_splits, test_size, rand_stat):
    
    '''
    this function splits dataframe into train/ test datasets

    Parameters
    ----------
    df : pandas df
        dataframe
    col : string
        groupby column to be considered during sampling
    n_splits: int
        number of shuffling iterations
    test_size: float
        size of test dataset
    rand_stat: int
        integer for reproducible output over multiple calls

    Returns
    -------
    train : pandas df
    test : pandas df
    '''
    
    print("split data into train/test datasets... ")
    try:
        split = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=rand_stat)

        for train_index, test_index in split.split(df, df[col]):
            train = df.loc[train_index]
            test = df.loc[test_index]
        
        print("done!")
        
    except Exception as e:
        print(e)
        
    return train, test

def drop_low_inf_cols(df, threshold):
    
    '''
    this function drops columns with high ratio of missing data 
    i.e. above given threshold

    Parameters
    ----------
    df : pandas df
        dataframe
    threshold : float
        max ratio of allowed nan values in a columns

    Returns
    -------
    df : pandas df

    '''
    print("drop columns with high ratio of missing data... ")
    
    try:
        df = df.loc[:,df.isna().mean() <= threshold]        
        print("done!")
        
    except Exception as e:
        print(e)
        
    return df

def get_datetime_features(df):
    
    '''
    this function returns list of columns of time datetime in dataframe

    Parameters
    ----------
    df : pandas df
        dataframe    

    Returns
    -------
    df : pandas df

    '''
    
    print("get datetime features... ")
    
    try:
        for col in df.columns:
            if df[col].dtype == 'object':            
                df[col] = pd.to_datetime(df[col], errors='ignore')
        
        print("done!")
        
    except Exception as e:
        print(e)
        
    return list(df.columns[df.dtypes=='datetime64[ns]'])

def get_cat_features(df, threshold):
    '''
    this function gets list of categorical columns based on max. number 
    of unique values

    Parameters
    ----------
    df : pandas df
        dataframe
    threshold: int
        max. number of unique values

    Returns
    -------
    cat_list:list
        list of df categorical columns

    '''
    
    print("get categorical features... ")
    
    try:
        # get string columns
        df= df.convert_dtypes(
            infer_objects=True, convert_string=True, 
            convert_integer=True, convert_floating=True)
        
        # get cat features
        cat_list= []    
        
        # append to cat_list
        [cat_list.append(f) 
         for f in list(df.columns) if df[f].nunique() <= threshold]          
        
        # get string cols
        str_list= list(df.columns[df.dtypes=='string'])
    
        print("done!")
        
    except Exception as e:
        print(e)
        
    return list(set (cat_list + str_list))

def encode_cat_features(df, cat_list):
        
    '''
    this function encodes categorical features into numerical (i.e. dummy) 
    columns using pandas get_dummies

    Parameters
    ----------
    df : pandas df
        dataframe
    cat_list: list
        list of cat columns 

    Returns
    -------
    df : pandas df

    '''
    print("encode categorical features... ")
    
    try:
        df = pd.get_dummies(df, columns=cat_list)
    
        print("done!")
        
    except Exception as e:
        print(e)
        
    return df

def encode_datetime_features(df, dt_list):
    
    '''
    this function encodes datetime features into numerical columns 

    Parameters
    ----------
    df : pandas df
        dataframe
    dt_list: list
        list of datetime columns 

    Returns
    -------
    df : pandas df

    '''
    print("encode datetime features... ")
    
    # set targeted date/time features
    targeted_features= ["year", "month", "day", "dayofweek", 
                         "hour", "minute", "second"]
    
    try:
        # set columns names to lowercase
        df.columns= df.columns.str.lower()
        
        # remove spaces from columns names
        df.columns = df.columns.str.replace(' ', '_') 

        # generate numerical features out of datetime features
        for f in dt_list:
            for t in targeted_features:                           
                df[f + '_' + t] = getattr(df[f].dt, t)                                     
            
        print("done!")
        
    except Exception as e:
        print(e)
        
    return df

def add_new_timediff_features(df):
    '''
    this function adds new time difference feetures between existing
    datatime ones

    Parameters
    ----------
    df : pandas df
        dataframe

    Returns
    -------
    df : pandas df
        dataframe

    '''
    
    print("generate new timediff features... ")
    try:
        # generate time difference attributes
        df["expected_start_start_process_diff"] = \
        (df['start_process']-df['expected_start']).astype('timedelta64[m]')
        
        df["start_process_process_end_diff"] = \
        (df['process_end']-df['start_process']).astype('timedelta64[m]')
        
        df["start_subprocess1_subprocess1_end_diff"] = \
        (df['subprocess1_end']-df['start_subprocess1']).astype('timedelta64[m]')
        
        df["start_critical_subprocess1_subprocess1_end_diff"] = \
        (df['subprocess1_end']-df['start_critical_subprocess1']).astype('timedelta64[m]')    
        
        df["start_process_reported_on_tower_diff"] = \
        (df['reported_on_tower']-df['start_process']).astype('timedelta64[m]')
        
        df["process_end_reported_on_tower_diff"] = \
        (df['reported_on_tower']-df['process_end']).astype('timedelta64[m]')
    
        print("done!")
        
    except Exception as e:
        print(e)
    
    return df

def drop_datetime_cols(df, dt_cols):
    '''
    this function drops datetime columns from a given dataframe

    Parameters
    ----------
    df : dataframe
        dataframe.
    dt_cols : list
        list of datetime column.

    Returns
    -------
    df : dataframe

    '''
    
    print("delete datetime features after encoding them... ")
    try:
        df= df.drop(dt_cols, axis = 1)
    
        print("done!")
        
    except Exception as e:
        print(e)
        
    return df

def impute_missing_data(df, group, criteria):
    '''
    this fuunction fill in missing data into pandas dataframe
    using imputation criteria and list of grouping columns

    Parameters
    ----------
    df : dataframe
        data frame
    group_list : list
        list of grouping columns


    Returns
    -------
    df : dataframe
        data frame

    '''
    print("impute missing data... ")
    try:
        for c in list(df.columns):    
            df[c] = df[c].fillna(
            df.groupby(group)[c].transform(criteria))
        
        print("done!")
        
    except Exception as e:
        print(e)
        
    return df
    
def drop_low_var_cols(df, threshold):
    
    '''
    this function drops columns with low variance/Std 
    i.e. std below given threshold

    Parameters
    ----------
    df : pandas df
        dataframe
    threshold : float
        min ratio of allowed std value in a column

    Returns
    -------
    df : pandas df

    '''
    
    print("drop low-variance features... ")
    try:
        # drop all na columns
        df= df.dropna(axis=1, how='all')    
        
        # keep features only with high variance threshold
        df = df.loc[:, df.std() > threshold]
        
        print("done!")
        
    except Exception as e:
        print(e)
        
    return df

def split_features_labels(df, label_col, non_features_list):
    
    '''
    this function splits a dataset into features and labels subsets
    
    Parameters
    ----------
    df : pandas df
        dataframe
    label_col : string
        name of target variable
    non_features_list: list
        list of indexing cols

    Returns
    -------
    features : pandas df
    labels: pandas df
    '''
    
    print("split labels from dataset... ")
    try:
        # separate features and target into 2 variables
        features= df.drop(non_features_list, axis=1)
        labels = df[label_col].copy()
    
        print("done!")
        
    except Exception as e:
        print(e)
        
    return features, labels

def scale_data(df):
    '''
    this function scaled data features around 0 average (standardization)
    
    Parameters
    ----------
    df : pandas df
        dataframe

    Returns
    -------
    pandas df
    '''
    
    print("scale data around 0 average (standardization)... ")
    try:
        # create scaler instance
        trans = StandardScaler()
    
        # fit scaler using numerical data
        scaled_features = trans.fit_transform(df.values)
        
        # convert the array back to a dataframe
        df= pd.DataFrame(scaled_features, index=df.index, 
                            columns=df.columns)
        
        print("done!")
        
    except Exception as e:
        print(e)
        
    return df    

def data_transformation(df, df_name):
    '''
    this function does the following tasks:
        1- drop columns with high ratio of missing data
        2- get datetime features
        3- get categorical features
        4- encode categorical features into dummy ones
        5- encode datetime features to numerical ones
        6- generate new timediff features out of existing datetime features
        7- delete datetime features after encoding them
        8- impute missing data
        9- drop low-variance features
        10-split labels & features of data
        11-scale data around 0 average (standardization)
        
    Parameters
    ----------
    df : pandas df
        dataframe
    df_name: string
        type of dataframe(train/test)

    Returns
    -------
    f : df
        data features ready for learning
    l: vector
        target variable    
    '''
    
    print("\nStart Data Transformation for: ", df_name)
    
    # remove low informative columns
    df = drop_low_inf_cols(df, 0.5)
    
    # get timestamp features    
    timestamp_features= get_datetime_features(df)
    
    # get categorical features
    th= int(len(df)*0.003)
    cat_features = get_cat_features(df, th)
        
    # drop heterogeneous column (opened)
    if 'opened' in df.columns:
        df= df.drop(['opened'], axis=1)
        
    if 'opened' in cat_features:
        cat_features= cat_features.remove('opened')    
        
    # encode categorical features
    df= encode_cat_features(df, cat_features)    
    
    # encode datetime features
    df= encode_datetime_features(df, timestamp_features)
    
    # add new datetime features
    df= add_new_timediff_features(df)

    # drop datetime columns
    df = drop_datetime_cols(df, timestamp_features)
    
    # impute missing data
    df= impute_missing_data(df, 'groups', 'median')    
        
    # drop columns with low variance
    df = drop_low_var_cols(df, 0.05)    
    
    # separate features and target into 2 variables    
    f, l= split_features_labels(df, "target", ["index", "groups","target"])    
    
    # scale train dataset
    f= scale_data(f)      
    
    print("\nFinish Data Transformation for: ", df_name, "\n")    
    # return df features and labels
    return f, l

def standardize_train_test_features(df1, df2):
    '''
    this function standardize set of features bw train and test datasets 
    (to fit properly in the training algorithm )

    Parameters
    ----------
    df1 : pandas df
        dataframe      
    df2 : pandas df
        dataframe

    Returns
    -------
    df1, df12

    '''
    print("set train/test datasets to common features... ")
    try:
        # drop nan columns
        df1= df1.dropna(axis='columns')
        df2= df2.dropna(axis='columns')

        # get list of common columns between train & test datasets
        common_cols= list(df1.columns.intersection(df2.columns))
        
        # reset train & test sets on shared columns (for training purpose)
        df1 = df1[common_cols]
        df2 = df2[common_cols]
        
        print("done!")
        
    except Exception as e:
        print(e)
        
    return df1, df2   
    
if __name__ == "__main__":
    
    # import data
    DATA_PATH= ""
    FEATURES_FILENAME= "md_raw_dataset.csv"   
    TARGET_FILENAME= "md_target_dataset.csv"    
    TEST_SIZE= 0.2
    
    raw_data= import_csv_data(DATA_PATH, FEATURES_FILENAME, "infer", ";")
    target= import_csv_data(DATA_PATH, TARGET_FILENAME, "infer", ";")    
    
    # name unnamed columns in raw_data
    column_idx= [0,7,17]
    columns_names= ["index","unnamed01", "unnamed02"]
    
    # add names to unnamed columns
    raw_data= name_unnamed_cols(raw_data, column_idx, columns_names)    
    
    # merge raw and tagret dataframes into one
    df_total= merge_data(raw_data, target, ["index", "groups"])
    
    # split data
    train_set, test_set = split_data(df_total, "groups", 10, TEST_SIZE, 0)
    
    # clean memory
    del raw_data, target, df_total
        
    # apply data transformation on train dataset    
    train, train_labels= data_transformation(train_set,"train data")        
   
    # apply data transformation on test dataset    
    test, test_labels= data_transformation(test_set, "test data")
    
    # set train/test datasets to common cols
    train, test = standardize_train_test_features(train, test)        
 
    ###############################################################
    ## 1- linear regression with cross validation
    lin_reg = LinearRegression()
    lin_reg.fit(train, train_labels)
    lin_scores = cross_val_score(lin_reg, train, train_labels,
    scoring="neg_mean_squared_error", cv=10)
    lr_rmse_scores = np.sqrt(-lin_scores)
    
    print("\nLinear Regression Model - 10-fold Cross Validation RMSE:")
    print("RMSE Mean: ", lr_rmse_scores.mean())
    print("RMSE Std: ", lr_rmse_scores.std())
    
    lin_predictions = lin_reg.predict(test)
    lin_mse = mean_squared_error(test_labels, lin_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("RMSE on Test data: ", lin_rmse)
    
    ###############################################################
    ## 2- decision tree regressor with cross validation
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(train, train_labels)

    tree_scores = cross_val_score(tree_reg, train, train_labels,
    scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-tree_scores)
    
    print("\nDecision Tree Model - 10-fold Cross Validation RMSE:")
    print("RMSE Mean: ", tree_rmse_scores.mean())
    print("RMSE Std: ", tree_rmse_scores.std())
    
    dt_predictions = tree_reg.predict(test)
    dt_mse = mean_squared_error(test_labels, dt_predictions)
    dt_rmse = np.sqrt(dt_mse)
    print("RMSE on Test data: ", dt_rmse)
    
    