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
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.classifier import ConfusionMatrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('stopwords')
from keras import backend as K


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


def xyvalues():
    """Returns X and y values for data"""
    x_val = pd.read_csv('features.csv')
    y_val = pd.read_csv('target.csv')
    y_val = reduce_mem_usage(y_val)
    return x_val, y_val


def test_group(model, y_test, x_val, y_val, vectorize==True):
    """Returns test section with group classification"""
    x_val = x_val[x_val.index.isin(y_test.index)]
    y_val = y_val[y_val.index.isin(y_test.index)]
    data = pd.concat([y_val, x_val], axis=1)
    data = data.loc[(data.offensive_and_identity==1)|
                    (data.offensive_and_identity==3)]
    x_group = data[['comment_text']]
    y_group = data['offensive']
    clean_group_data = []
    for testdata in x_group['comment_text']:
        clean_group_data.append(testdata)
    if vectorize:
        clean_group_data = model.transform(clean_group_data)
    return clean_group_data, y_group 


def train_test(x_val, y_val, test=.33, rs=42):
    """Seperates values into train and test data"""
    x_train, x_test, y_train, y_test = train_test_split(x_val, y_val,
                                                        test_size=test,
                                                        random_state=rs,
                                                        stratify=y_val)
    return x_train, x_test, y_train, y_test


def scores(y_test, y_pred):
    """Returns precision, recall, accuracy, and F1."""
    print('Test Accuracy score: ', accuracy_score(y_test, y_pred))
    print('Test F1 score: ', f1_score(y_test, y_pred))
    print('Test Cost Function Score: ', cost_function(y_test, y_pred))


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


def recall_k(y_true, y_pred):
    """Creates custom recall measurement for Keras"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_k(y_true, y_pred):
    """Creates custom precision measurement for Keras"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

    
def f1_k(y_true, y_pred):
    """Creates custom F1 measurement for Keras"""
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    f1_val = 2*((precision*recall)/(precision+recall+K.epsilon()))
    return f1_val


def cost_function_k(y_true, y_pred):
    """Creates custom cost function measurement for Keras"""
    true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
    total_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    false_pos = pred_pos-true_pos
    false_neg = total_pos-true_pos
    true_neg = int((len(y_true))-(true_pos+false_pos+false_neg))
    score = 0
    score = (4*false_pos)+false_neg-(2*true_pos)-(0.0005*true_neg)
    return score
    

def cost_function(y_true, y_pred):
    """Creates score for custom cost function"""
    pred_labels = np.asarray(y_pred)
    true_labels = np.asarray(y_true)
    true_pos = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    true_neg = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    false_pos = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    false_neg = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    score = 0
    score = (4*false_pos)+false_neg-(2*true_pos)-(0.0005*true_neg)
    return score


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix"):
    """Creates graph of confusion matrix"""
    plt.grid(None)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='45')
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, cm[i,j], horizontalalignment="center",
                 color="white" if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('True \nlabel', rotation=0)
    plt.xlabel('Predicted label')
    plt.savefig('distribution.png')


def visualize_training_results(results):
    """Creates graph that visualizes loss and F1 score measurements at each
    epoch during Keras modeling"""
    history = results.history
    plt.figure()
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    plt.figure()
    plt.plot(history['val_f1_m'])
    plt.plot(history['f1_m'])
    plt.legend(['val_f1_m', 'f1_m'])
    plt.title('F1 Scores')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Scores')
    plt.show()


def vectorize_initial(x_train, x_test):
    """Creates count vectorizer model, fits it to the X_train dataset, and
    transforms the X train and test datasets."""
    
    stopwords_set = set(stopwords.words('english'))
    
    clean_train_data = []
    for traindata in x_train['comment_text']:
        clean_train_data.append(traindata)

    clean_test_data = []
    for testdata in x_test['comment_text']:
        clean_test_data.append(testdata)
    
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=stopwords_set,
                                 max_features=6000,
                                 ngram_range=(1, 3))

    train_features = vectorizer.fit_transform(clean_train_data)
    test_features = vectorizer.transform(clean_test_data)
    return vectorizer, train_features, test_features


def log_gridsearch(train_features, y_train):
    """Returns gridsearch result for Logistic Regression"""
    textreg = LogisticRegression(fit_intercept = False, solver='saga', random_state=42)
    C = [1,3,5,7,9]
    hyperparameters = dict(C=C)
    clf = GridSearchCV(textreg, hyperparameters, cv=3, n_jobs=-2)
    best_model = clf.fit(train_features, y_train)
    print('Best C:', best_model.best_estimator_.get_params()['C'])


def logreg(train_features, y_train, c_val):
    """Returns fitted logistic regression model"""
    textreg = LogisticRegression(C=c_val, fit_intercept = False, solver='saga', random_state=42)
    textreg.fit(train_features, y_train)
    return textreg


def logreg_cm(model, test_features, y_test, cm_labels):
    """Predicts classification using model, creates confusion matrix"""
    y_pred = model.predict(test_features)
    cm_val = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm_val, cm_labels)
    scores(y_test, y_pred)
    