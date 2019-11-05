#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Oct 10 10:28 2019

@author: khayes847
"""

import pandas as pd
import numpy as np
# pylint: disable=unused-import
import swifter
# pylint: enable=unused-import
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.classifier import ConfusionMatrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import boto3


def open_s3(bucket, data_key):
    """Pulls data from s3. Bucket is S3 name, key is file name"""
    data_location = (f's3://{bucket}/{data_key}')
    data = pd.read_csv(data_location)
    return data


def reduce_mem_usage(data):
    """
    iterate through all the columns of a dataframe and
    modify the data type to reduce memory usage.
    Original author unknown.
    Obtained from Puneet Grover,
    https://towardsdatascience.com/
    how-to-learn-from-bigdata-files-on-low-memory-incremental-learning-d377282d38ff
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
                elif (c_min > np.iinfo(np.int16).min
                      and c_max < np.iinfo(np.int16).max):
                    data[col] = data[col].astype(np.int16)
                elif (c_min > np.iinfo(np.int32).min and
                      c_max < np.iinfo(np.int32).max):
                    data[col] = data[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min
                      and c_max < np.iinfo(np.int64).max):
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max <\
                   np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min
                      and c_max < np.finfo(np.float32).max):
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
    data = data.sort_index(axis=0)
    return data


def id_column(data):
    """Sets id as column"""
    data = data.reset_index(drop=False)
    return data


def train_test(x_val, y_val, test=.25):
    """Seperates values into train and test data"""
    x_train, x_test, y_train, y_test = train_test_split(x_val, y_val,
                                                        test_size=test,
                                                        random_state=42,
                                                        stratify=y_val)
    return x_train, x_test, y_train, y_test


def initial():
    """Provides initial confusion matrix based on
    Civil Comments classification"""
    data = pd.read_csv('data_cleaned.csv', index_col='id')
    print('Overall')
    print(f'Total: {len(data)}')
    initial_cf(data)
    print('\nOnly bias data')
    print(f'Total: {len(data.loc[~(data.asian.isna())])}')
    initial_cf(data.loc[~(data.asian.isna())])
    print('\nBias data, no group')
    print(f'Total: {len(data.loc[data.no_group==1])}')
    initial_cf((data.loc[data.no_group == 1]))
    print('\nBias data, group')
    print(f'Total: {len(data.loc[data.no_group==0])}')
    initial_cf((data.loc[data.no_group == 0]))


def initial_cf(data):
    """Produces initial accuracy scores"""
    tp_val = len(data.loc[(data['target_binary'] == 1) &
                          (data['cc_rejected'] == 1)])
    fp_val = len(data.loc[(data['target_binary'] == 0) &
                          (data['cc_rejected'] == 1)])
    fn_val = len(data.loc[(data['target_binary'] == 1) &
                          (data['cc_rejected'] == 0)])
    tn_val = len(data.loc[(data['target_binary'] == 0) &
                          (data['cc_rejected'] == 0)])
    precision = tp_val/(tp_val+fp_val)
    recall = tp_val/(tp_val+fn_val)
    accuracy = (tp_val+fn_val)/len(data)
    f1_val = 2*((precision*recall)/(precision+recall))
    print(f'TP: {tp_val}, {tp_val/len(data)}')
    print(f'FP: {fp_val}, {fp_val/len(data)}')
    print(f'FN: {fn_val}, {fn_val/len(data)}')
    print(f'TN: {tn_val}, {tn_val/len(data)}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Accuracy: {accuracy}')
    print(f'F1: {f1_val}')


def min_max(x_train, x_test):
    """Scales variables to min_max"""
    scaler = MinMaxScaler(copy=False)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def scores(y_test, y_pred):
    """Returns precision, recall, accuracy, and F1."""
    print('Test precision score: ', precision_score(y_test, y_pred))
    print('Test Recall score: ', recall_score(y_test, y_pred))
    print('Test Accuracy score: ', accuracy_score(y_test, y_pred))
    print('Test f1 score: ', f1_score(y_test, y_pred))


def c_matrix(model, x_train, x_test, y_train, y_test):
    """Creates confusion matrix"""
    cm_model = ConfusionMatrix(model, classes=[0, 1])
    cm_model.fit(x_train, y_train)
    cm_model.score(x_test, y_test)
    cm_model.poof()


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


def logreg(x_train, x_test, y_train, y_test):
    """Returns logistic regression"""
    print('Logreg')
    logreg_new = LogisticRegression(fit_intercept=False, C=1e12,
                                    solver='saga', random_state=42)
    logreg_new.fit(x_train, y_train)
    y_pred = logreg_new.predict(x_test)
    scores(y_test, y_pred)
    c_matrix(logreg_new, x_train, x_test, y_train, y_test)


def cc_logreg():
    """Runs preliminary analysis of cc relationship to target"""
    col_list = ['id', 'target_binary', 'cc_rejected',
                'cc_toxicity_annotator_count',
                'cc_identity_annotator_count',
                'cc_likes', 'cc_disagree', 'cc_funny',
                'cc_sad', 'cc_wow']
    data = pd.read_csv('data_cleaned.csv', usecols=col_list, index_col='id')
    data = data.loc[data.asian.isna()]
    x_val = data.drop(columns=['target_binary'])
    y_val = data['target_binary']
    x_train, x_test, y_train, y_test = train_test(x_val, y_val, test=.25)
    xs_train, xs_test = min_max(x_train, x_test)
    dummy_classifier(xs_train, xs_test, y_train, y_test)
    logreg(xs_train, xs_test, y_train, y_test)


def remove_cc(data):
    """Removes cc columns from data"""
    data = pd.read_csv('data_cleaned.csv', index_col='id')
    data = data.drop(columns=['cc_rejected',
                              'cc_toxicity_annotator_count',
                              'cc_identity_annotator_count',
                              'cc_likes', 'cc_disagree',
                              'cc_funny', 'cc_sad', 'cc_wow'])
    data = id_column(data)
    data.to_csv('data_no_cc.csv', index=False)
    targets = data[['id', 'target', 'target_binary', 'severe_toxicity',
                    'obscene', 'identity_attack', 'insult', 'threat',
                    'sexual_explicit']]
    targets.to_csv('quora_targets.csv', index=False)
    features = data.drop(columns=['target', 'target_binary',
                                  'severe_toxicity',
                                  'obscene',
                                  'identity_attack',
                                  'insult', 'threat',
                                  'sexual_explicit'])
    features.to_csv('quora_features.csv', index=False)
    data_comments = features['comment_text']
    data_comments.to_csv('quora_comments.csv', index=False)
    data_identity = features.drop(columns=['comment_text'])
    data_identity.to_csv('quora_identity.csv', index=False)


def vader(x_val, s_type):
    """Returns specified vader score"""
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(x_val)
    return score[s_type]


def sentiment_initial(data):
    """Assigns vader score columns"""
    data['compound_initial'] = (data.comment_text
                                .swifter.apply
                                (lambda x:
                                 vader(x, 'compound')))
    return data
