#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Oct 10 10:28 2019

@author: khayes847
"""

import itertools
import pandas as pd
import numpy as np
# pylint: disable=unused-import
import swifter
# pylint: enable=unused-import
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import recall_score, precision_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
# nltk.download('stopwords')
from keras import backend as K
# pylint: disable=unused-import
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import models, layers, optimizers
from keras.layers import Dense, Dropout, Activation
# pylint: enable=unused-import


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


def test_data_group(model, y_test, x_val, y_val, vectorize=True):
    """Returns test section with group classification for
    initial logistic regression"""
    x_val = x_val[x_val.index.isin(y_test.index)]
    y_val = y_val[y_val.index.isin(y_test.index)]
    data = pd.concat([y_val, x_val], axis=1)
    data = data.loc[(data.offensive_and_identity == 1) |
                    (data.offensive_and_identity == 3)]
    x_group = data[['comment_text']]
    y_group = data['offensive']
    clean_group_data = []
    for testdata in x_group['comment_text']:
        clean_group_data.append(testdata)
    if vectorize:
        clean_group_data = model.transform(clean_group_data)
    return clean_group_data, y_group


def keras_group(x_test, y_test):
    """Returns identity section of test labelled as group comments
    for keras analysis"""
    data = pd.concat([y_test, x_test], axis=1)
    data = (data.loc[(data.offensive_and_identity == 1) |
                     (data.offensive_and_identity == 3)])
    x_group = data[['comment_text']]
    y_group = data['offensive_and_identity']
    return x_group, y_group


def train_test(x_val, y_val, test=.33, rs_val=42):
    """Seperates values into train and test data"""
    x_train, x_test, y_train, y_test = train_test_split(x_val, y_val,
                                                        test_size=test,
                                                        random_state=rs_val,
                                                        stratify=y_val)
    return x_train, x_test, y_train, y_test


def scores(y_test, y_pred):
    """Returns precision, recall, accuracy, and F1."""
    print('Test Accuracy score: ', accuracy_score(y_test, y_pred))
    print('Test F1 score: ', f1_score(y_test, y_pred))
    print('Test Cost Function Score: ', cost_function(y_test, y_pred))


def f1_score_custom(y_true, y_pred):
    """Creates custom F1 measurement for Keras"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_val = 2*((precision*recall)/(precision+recall+K.epsilon()))
    return f1_val


def cost_function(y_true, y_pred, multinary=False):
    """Creates score for custom cost function"""
    if multinary:
        y_true = change(y_true)
        y_pred = change(y_pred)
    pred_labels = np.asarray(y_pred)
    true_labels = np.asarray(y_true)
    true_pos = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    true_neg = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    false_pos = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    false_neg = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    score = 0
    score = (4*false_pos)+false_neg-(2*true_pos)-(0.0005*true_neg)
    return score


def plot_confusion_matrix(cm_val, classes, normalize=False,
                          title="Confusion Matrix"):
    """Creates graph of confusion matrix"""
    plt.grid(None)
    plt.imshow(cm_val, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='45')
    plt.yticks(tick_marks, classes)
    if normalize:
        cm_val = cm_val.astype('float') / cm_val.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm_val.max()/2
    for i, j in itertools.product(range(cm_val.shape[0]),
                                  range(cm_val.shape[1])):
        plt.text(j, i, cm_val[i, j], horizontalalignment="center",
                 color="white" if cm_val[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True \nlabel', rotation=0)
    plt.xlabel('Predicted label')
    plt.savefig('distribution.png')


def visualize_training_results(results):
    """Creates graph that visualizes Keras measurements at each epoch"""
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
    plt.plot(history['val_f1_score'])
    plt.plot(history['f1_score'])
    plt.legend(['val_f1_score', 'f1_score'])
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
    textreg = LogisticRegression(fit_intercept=False, solver='saga',
                                 random_state=42)
    c_val = [1, 3, 5, 7, 9]
    hyperparameters = dict(C=c_val)
    clf = GridSearchCV(textreg, hyperparameters, cv=3, n_jobs=-2)
    best_model = clf.fit(train_features, y_train)
    print('Best C:', best_model.best_estimator_.get_params()['C'])


def logreg(train_features, y_train, c_val):
    """Returns fitted logistic regression model"""
    textreg = LogisticRegression(C=c_val, fit_intercept=False,
                                 solver='saga', random_state=42)
    textreg.fit(train_features, y_train)
    return textreg


def logreg_cm(model, test_features, y_test, cm_labels):
    """Predicts classification using model, creates confusion matrix"""
    y_pred = model.predict(test_features)
    cm_val = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm_val, cm_labels)
    scores(y_test, y_pred)


def tokenizer_onehot(x_val, y_val):
    """Returns feature and target models for tokenizing
    and converting comments to matrix"""
    data = pd.concat([y_val, x_val], axis=1)
    comments = data["comment_text"]
    tokenizer = Tokenizer(num_words=6000)
    tokenizer.fit_on_texts(comments)
    one_hot_results = tokenizer.texts_to_matrix(comments, mode='binary')

    target = data['offensive_and_identity']
    le_val = preprocessing.LabelEncoder()
    le_val.fit(target)
    target_cat = le_val.transform(target)
    target_onehot = to_categorical(target_cat)
    return one_hot_results, target_onehot


def keras_prep(one_hot_results, target_onehot, x_test, group=False):
    """Prepares training and test datasets for Keras, with option
    to only prepare 'identity' section of test data."""
    test_index = list(x_test.index)
    test = one_hot_results[test_index]
    label_test = target_onehot[test_index]
    if group:
        return test, label_test
    train = np.delete(one_hot_results, test_index, 0)
    label_train = np.delete(target_onehot, test_index, 0)
    return train, label_train, test, label_test


def keras_model(train, label_train, epoch):
    """Creates Keras model and results"""
    model = models.Sequential()
    model.add(Dropout(.2))
    model.add(layers.Dense(1000, activation='relu', input_shape=(6000,)))
    model.add(Dropout(.2))
    model.add(layers.Dense(500, activation='relu'))
    model.add(Dropout(.2))
    model.add(layers.Dense(100, activation='relu'))
    model.add(Dropout(.2))
    model.add(layers.Dense(50, activation='relu'))
    model.add(Dropout(.2))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=[f1_score_custom])
    results = model.fit(train,
                        label_train,
                        epochs=epoch,
                        batch_size=250,
                        validation_split=0.2)
    return model, results


def model_evaluate(model, features, label, dataset='Training'):
    """Evaluates entropy loss and F1-score when predicting target using model
    against actual target data"""
    results = model.evaluate(features, label)
    print(f'{dataset} Entrpy Loss: {results[0]}')
    print(f'{dataset} F1-Score: {results[1]}')


def prediction_keras(model, test, y_test):
    """Creates a target prediction array using Keras model, and converts
    actual target data to array"""
    y_pred_classes = model.predict_classes(test)
    multi_label_test = np.array(y_test)
    return y_pred_classes, multi_label_test


def confusion_keras(y_pred_classes, multi_label_test, binary=False):
    """Returns confusion matrix for Keras predictions"""
    cm_val = confusion_matrix(multi_label_test, y_pred_classes)
    cm_labels = confusion_matrix(multi_label_test, y_pred_classes)
    if binary:
        cm_labels = ['Good', 'Bad']
    else:
        cm_labels = ['Good_none', 'Good_group',
                     'Bad_none', 'Bad_group']
    plot_confusion_matrix(cm_val, cm_labels)


def change(array):
    """Changes array from multinary to binary"""
    list_original = list(array)
    list_new = []
    for i in list_original:
        if i in [0, 1]:
            list_new.append(0)
        elif i in [2, 3]:
            list_new.append(1)
    array_new = np.array(list_new)
    return array_new


def final_score(x_test, y_test, one_hot_results, target_onehot, model):
    """Returns Keras model cost function score for only identity group test
    data"""
    x_group, y_group = keras_group(x_test, y_test)
    test_group, label_group = keras_prep(one_hot_results, target_onehot,
                                         x_group,
                                         group=True)
    y_pred_group, multi_label_group = prediction_keras(model, test_group,
                                                       y_group)
    cost_function(multi_label_group, y_pred_group, multinary=True)
