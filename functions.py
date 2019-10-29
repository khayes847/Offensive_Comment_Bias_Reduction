#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Oct 10 10:28 2019

@author: khayes847
"""

import os
import boto3
import re
import copy
import time
from time import gmtime, strftime
import pandas as pd
import numpy as np
import swifter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import data_prep as d
import functions as f
from sagemaker import get_execution_role


def open_s3(bucket, data_key):
    """Pulls data from s3. Bucket is S3 name, key is file name"""
    role = get_execution_role()
    region = boto3.Session().region_name
    prefix = 'sagemaker/xgboost-mnist'
    bucket_path = (f'https://s3-{region}.amazonaws.com/{bucket}')
    data_location = (f's3://{bucket}/{data_key}')
    data = pd.read_csv(data_location)
    return data


def open_s3_txt(bucket, data_key):
    """Pulls data from s3. Bucket is S3 name, key is file name"""
    role = get_execution_role()
    region = boto3.Session().region_name
    prefix = 'sagemaker/xgboost-mnist'
    bucket_path = (f'https://s3-{region}.amazonaws.com/{bucket}')
    data_location = (f's3://{bucket}/{data_key}')
    text = open(data_location, 'rb')
    return data


def reduce_mem_usage(data):
    """ 
    iterate through all the columns of a dataframe and 
    modify the data type to reduce memory usage.
    Original author unknown.
    Obtained from Puneet Grover,
    https://towardsdatascience.com/how-to-learn-from-bigdata-files-on-low-memory-incremental-learning-d377282d38ff
    """
    start_mem = data.memory_usage().sum() / 1024**2
    print(('Memory usage of dataframe is {:.2f}' 
                     'MB').format(start_mem))
    
    for col in data.columns:
        col_type = data[col].dtype
        
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max <\
                  np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max <\
                   np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max <\
                   np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max <\
                   np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max <\
                   np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max <\
                   np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)
        else:
            data[col] = data[col].astype(col_type)
    end_mem = data.memory_usage().sum() / 1024**2
    print(('Memory usage after optimization is: {:.2f}' 
                              'MB').format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) 
                                             / start_mem))
    
    return data


def upload_s3(file, bucket, s3_name):
    """Uploads file to s3"""
    s3_client = boto3.client('s3')
    s3_client.upload_file(file, bucket, s3_name)


def id_index(data):
    """Sets index as id"""
    data = data.set_index('id')
    data = data.sort_index(axis = 0)
    return data


def id_column(data):
    """Sets id as column"""
    data = data.reset_index(drop=False)
    return data


def train_test(x_val, y_val, test=.25):
    """Seperates values into train and test data"""
    x_train, x_test, y_train, y_test = train_test_split(x_val, y_val,
                                                        test_size = test,
                                                        random_state=42,
                                                        stratify=y)
    return x_train, x_test, y_train, y_test


def initial():
    """Provides initial confusion matrix based on Civil Comments classification"""
    data = pd.read_csv('data_cleaned.csv', index_col='id')
    print('Overall')
    print(f'Total: {len(data)}')
    initial_cf(data)
    print('\nOnly bias data')
    print(f'Total: {len(data.loc[~(data.asian.isna())])}')
    initial_cf(data.loc[~(data.asian.isna())])
    print('\nBias data, no group')
    print(f'Total: {len(data.loc[data.no_group==1])}')
    initial_cf((data.loc[data.no_group==1]))
    print('\nBias data, group')
    print(f'Total: {len(data.loc[data.no_group==0])}')
    initial_cf((data.loc[data.no_group==0]))


def initial_cf(data):
    """Produces initial accuracy scores"""
    TP = len(data.loc[(data['target_binary']==1)&(data['cc_rejected']==1)])
    FP = len(data.loc[(data['target_binary']==0)&(data['cc_rejected']==1)])
    FN = len(data.loc[(data['target_binary']==1)&(data['cc_rejected']==0)])
    TN = len(data.loc[(data['target_binary']==0)&(data['cc_rejected']==0)])
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    Accuracy = (TP+FN)/len(data)
    f1 = 2*((Precision*Recall)/(Precision+Recall))
    print(f'TP: {TP}, {TP/len(data)}')
    print(f'FP: {FP}, {FP/len(data)}')
    print(f'FN: {FN}, {FN/len(data)}')
    print(f'TN: {TN}, {TN/len(data)}')
    print(f'Precision: {Precision}')
    print(f'Recall: {Recall}')
    print(f'Accuracy: {Accuracy}')
    print(f'F1: {f1}')


def min_max(x_train, x_test, x_group_test, x_no_test):
    """Scales variables to min_max"""
    scaler = MinMaxScaler(copy=False)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_group_test = scaler.transform(x_group_test)
    x_no_test = scaler.transform(x_no_test)
    return x_train, x_test, x_group_test, x_no_test


def scores(y_test, y_pred):
    """Returns precision, recall, accuracy, and F1."""
    print('Test precision score: ', precision_score(y_test, y_pred))
    print('Test Recall score: ', recall_score(y_test, y_pred))
    print('Test Accuracy score: ', accuracy_score(y_test, y_pred))
    print('Test f1 score: ', f1_score(y_test, y_pred))


def c_matrix(model, x_train, x_test, y_train, y_test):
    """Creates confusion matrix"""
    cm = ConfusionMatrix(model, classes=[0,1])
    cm.fit(x_train, y_train)
    cm.score(x_test, y_test)
    cm.poof()


def dummy_classifier(x_train, x_test, y_train, y_test):
    """Returns dummy classifier"""
    print('Dummy')
    y_train = y_train
    y_test = y_test
    dummy = DummyClassifier(strategy='most_frequent',
                            random_state=42).fit(x_train, y_train)
    y_pred = dummy.predict(x_test)
    scores(y_test, y_pred)
    c_matrix(dummy, x_train, x_test, y_train, y_test)


def logreg(x_train, x_test, y_train, y_test, ret=False):
    """Returns logistic regression"""
    print('Logreg')
    y_train = y_train
    y_test = y_test
    logreg = LogisticRegression(fit_intercept = False, C = 1e12,
                                solver='saga', random_state=42)
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    scores(y_test, y_pred)
    c_matrix(logreg, x_train, x_test, y_train, y_test)
    if ret:
        return logreg


def cc_logreg():
    """Runs preliminary analysis of cc relationship to target"""
    col_list = ['id', 'target_binary', 'cc_rejected', 'cc_toxicity_annotator_count',
       'cc_identity_annotator_count', 'cc_likes', 'cc_disagree', 'cc_funny',
       'cc_sad', 'cc_wow']
    data = pd.read_csv('data_cleaned.csv', usecols=col_list, index_col = 'id')
    data = data.loc[data.asian.isna()]
    x_val = data.drop(columns = ['target_binary'])
    y_val = data['target_binary']
    x_train, x_test, y_train, y_test = train_test(x_val, y_val, test=.25)
    xs_train, xs_test = min_max(x_train, x_test)
    dummy_classifier(xs_train, xs_test, y_train, y_test)
    logreg(xs_train, xs_test, y_train, y_test)


def bias_logreg():
    """Needs cleaning"""
    col_list = ['target_binary', 'cc_rejected', 'cc_toxicity_annotator_count',
    'cc_identity_annotator_count', 'cc_likes', 'cc_disagree', 'cc_funny',
    'cc_sad', 'cc_wow']
    data = pd.read_csv('data_cleaned.csv', index_col = 'id')
    data_train = data.loc[data.asian.isna()]
    data_train = data_train[col_list]
    X = (data_train[col_list]).drop(columns=['target_binary'])
    y = data_train['target_binary']
    X_train, X_test, y_train, y_test = train_test(X, y)
    xs_train, xs_test = f.min_max(X_train, X_test)
    print('Dummy - Total Dataset\n')
    dummy_classifier(xs_train, xs_test, y_train, y_test)
    print('Logreg - Total Dataset\n')
    logreg = f.logreg(xs_train, xs_test, y_train, y_test, ret=True)
    data_no_group = data.loc[data.no_group == 1]
    data_no_group = data_no_group[col_list]
    data_group = data.loc[data.no_group == 0]
    data_group = data_group[col_list]
    X_no_group = data_no_group.drop(columns = ['target_binary'])
    y_no_group = data_no_group['target_binary']
    scaler = MinMaxScaler(copy=False)
    X_no_group = scaler.fit_transform(X_no_group)
    y_no_group_pred = logreg.predict(X_no_group)
    print('Logreg - Group annotations data: no group\n')
    f.scores(y_no_group, y_no_group_pred)
    cm = ConfusionMatrix(logreg, classes=[0,1])
    cm.fit(xs_train, y_train)
    cm.score(X_no_group, y_no_group)
    cm.poof()
    X_group = data_group.drop(columns = ['target_binary'])
    y_group = data_group['target_binary']
    scaler = MinMaxScaler(copy=False)
    X_group = scaler.fit_transform(X_group)
    y_group_pred = logreg.predict(X_group)
    print('Logreg - Group annotations data: one or more groups\n')
    f.scores(y_group, y_group_pred)
    cm = ConfusionMatrix(logreg, classes=[0,1])
    cm.fit(xs_train, y_train)
    cm.score(X_group, y_group)
    cm.poof()


def remove_cc(data):
    """Removes cc columns from data"""
    data = pd.read_csv('data_cleaned.csv', index_col='id')
    data = data.drop(columns=['cc_rejected', 'cc_toxicity_annotator_count',
       'cc_identity_annotator_count', 'cc_likes', 'cc_disagree', 'cc_funny',
       'cc_sad', 'cc_wow'])
    data = f.id_column(data)
    data.to_csv('data_no_cc.csv', index=False)
    targets = data[['id', 'target', 'target_binary', 'severe_toxicity',
                    'obscene', 'identity_attack', 'insult', 'threat',
                    'sexual_explicit']]
    targets.to_csv('quora_targets.csv', index=False)
    features = data.drop(columns =
                     ['target', 'target_binary', 'severe_toxicity',
                      'obscene', 'identity_attack', 'insult', 'threat',
                      'sexual_explicit'])
    features.to_csv('quora_features.csv', index=False)
    data_comments = features['comment_text']
    data_comments.to_csv('quora_comments.csv', index=False)
    data_identity = features.drop(columns = ['comment_text'])
    data_identity.to_csv('quora_identity.csv', index=False)


def vader(x_val, s_type):
    """Returns specified vader score"""
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(x_val)
    return score[s_type]


def sentiment_initial(data):
    """Assigns vader score columns"""
    data['compound_initial'] = data.comment_text.swifter.apply(lambda x: vader(x, 'compound'))
    return data