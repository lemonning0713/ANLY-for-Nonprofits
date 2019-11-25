# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:59:06 2019

@author: Jimny
"""

## Import needed packages
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV   #Perforing grid search

from sklearn.model_selection import train_test_split

from operator import itemgetter 

import pickle

from sklearn.metrics import mutual_info_score,roc_auc_score,confusion_matrix,accuracy_score, roc_curve, auc, classification_report

import itertools
from scipy import interp

def main():
    # Open and read in the cleaned data from 2013 to 2016
    with open('Cleaned Data/2013_2016_cleaned.csv', 'rb') as input_all:
        df_all = pd.read_csv(input_all , sep=',',  header = 0, encoding = 'utf-8')

    # Select the columns we needed
    select_columns = ['ev_month', 'ev_weekday', 'ev_state', 
                      'ev_highest_injury', 'damage', 'far_part',
                      'acft_category', 'type_fly', 'CICTTEvent', 'CICTTPhase']
    df_select = df_all.loc[ : , select_columns]
    
    # Change column names for convenience
    df_select.columns = ['Month', 'Weekday', 'State', 
                         'Injury_Level', 'Damage', 'Part',
                         'Aircraft', 'Flight_Type', 'Event', 'Phase']
    
    # Print NA values counts for checking tnad then dorp rows with NA values
    print('Check number of NA values from selected columns:\n',
          df_select.isnull().sum())
    
    df_select.dropna(axis=0, inplace = True)
    df_select.reset_index(drop = True, inplace = True)
    
    # Separate the two classes in the original dataset
    df_none = df_select.loc[df_select['Injury_Level'] == 'NONE']
    df_fatl = df_select.loc[df_select['Injury_Level'] == 'FATL']

    # Balance Dataset
    n_fatl = len(df_fatl)
    df_none = df_none.sample(n = n_fatl, replace = False, random_state = 117)
    
    df_sampled = pd.concat([df_none,df_fatl], ignore_index=True)
    df_sampled.reset_index(drop = True, inplace = True)

    # Separate predictors and response labels
    df_X = df_sampled.drop(['Injury_Level'], axis = 1)
    df_y = df_sampled.loc[: ,  'Injury_Level' ]
    
    # Using One-hot encoder to transform the dataset
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df_X)
    df_X = pd.DataFrame(enc.transform(df_X).toarray(), columns = enc.get_feature_names(list(df_X.columns)))
    
    '''
    # Normalize the dataset
    scaler = MinMaxScaler()
    scaler.fit(df_X)
    df_X = pd.DataFrame(scaler.transform(df_X), columns=df_X.columns)
    '''
    # Iterate through each label of train data
    for i in range(len(df_y)):
        # Convert text labels to number labels accordingly
        if df_y[i] == 'NONE':
            df_y[i] = '0'
        elif df_y[i] == 'FATL':
            df_y[i] = '1'
    
    # Split train and test dataset by the ratio of 80% and 20%
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=1378)
    
    # Call function to reduce the dimension of the dataset
    X_train, X_test = dimension_reduction(X_train, y_train, X_test, 100 , method = 'RFE')
    

    #Grid seach on subsample and max_features
    #Choose all predictors except target & IDcols
    param_test1 = {
                #'num_leaves': [10,15,20,25,30,40],
                #'min_data_in_leaf': [10,20,30,40,50,60],
                #'max_depth': [-1,3,5,7,9,11,15],
                #'bagging_fraction': [0.5,0.6,0.7,0.8,0.83,0.85,0.87,0.9,0.92],
                #'bagging_freq': [2,3,5,7,9,11,13,15],
                #'feature_fraction': [0.4,0.45,0.5,0.6,0.7,0.8,0.83,0.85,0.87,0.9,0.92],
                #'lambda_l1': [0, 0.0001, 0.0005, 0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2,0.3, 0.5, 1],
                #'lambda_l2': [0, 0.0001, 0.0005, 0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2,0.3, 0.5, 1]
                #'num_iterations': [50,80,100,120,150,200,300,500],
                'learning_rate': [0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2,0.3, 0.5, 1]
                }
    gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier(objective = 'binary',
                                                            boosting = 'gbdt',
                                                            metric = 'binary_logloss',
                                                            num_leaves = 15,
                                                            min_data_in_leaf = 10,
                                                            max_depth = 5,
                                                            bagging_fraction = 0.85,
                                                            bagging_freq = 11,
                                                            feature_fraction = 0.5,
                                                            lambda_l1 = 0.01,
                                                            lambda_l2 = 0.3,
                                                            num_iterations = 100,
                                                            learning_rate = 0.08,
                                                            random_state = 117),
                            param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=5)
    gsearch1.fit(X_train,y_train)
    
    print('Tuning: \n', gsearch1.best_params_, '\t', gsearch1.best_score_)
    
    
    
    
# This function is used to draw ROC curves
def draw_roc( Y, Y_score):
    
    # Make binarized multi classes labels
    #Y = label_binarize(Y, classes =  [0,1,2])
    #print(Y.shape)
    n_classes = Y.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], Y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    
    lw = 4
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(10,7))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
                   color='deeppink', linestyle=':', linewidth=lw)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
                   color='navy', linestyle=':', linewidth=lw)
    
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkred'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    
# This function is used to print validation/test resuls
def print_validate(Y_val, predictions):
    print("\nAccuracy score: ",accuracy_score(Y_val, predictions))
    print("\nConfusion Matrix: \n",confusion_matrix(Y_val, predictions))
    print(classification_report(Y_val, predictions))

# Define the function to reduce data dimensionality
def dimension_reduction(train_x, train_y, test_x, n_col, method = 'fact'):
    # Obtain column names
    attr_list = train_x.columns
    
    # Using RFE to rank feactures and then select
    if method == 'RFE':
        # Using RFE to rank attributes
        lin_reg = LinearRegression()
        rfe = RFE(lin_reg, n_col)
        fit = rfe.fit(train_x, train_y)
    
        # Selecte most relevant attributes for machien learning
        fit_list = fit.support_.tolist()
        indexes = [index for index in range(len(fit_list)) if fit_list[index] == True]
    
        # Print out attributes selected and ranking
        print('\nAttributes selected are: ', itemgetter(*indexes)(attr_list))
        print('\nAttributes Ranking: ', fit.ranking_)

        train_x_returned = train_x.iloc[:,indexes]
        test_x_returned = test_x.iloc[:,indexes]
    
    # Using factor analysis
    elif method == 'fact':
        fact_anal = FactorAnalysis(n_components=n_col)
        train_x_returned = pd.DataFrame(fact_anal.fit_transform(train_x))
        test_x_returned = pd.DataFrame(fact_anal.transform(test_x))
    
        train_x_returned.columns = [''.join(['feature_',str(i)]) for i in list(train_x_returned.columns)]
        test_x_returned.columns = [''.join(['feature_', str(i)]) for i in list(test_x_returned.columns)]
    
    # Using PCA
    elif method == 'PCA':
        pca_down = PCA(n_components=n_col)
        train_x_returned = pd.DataFrame(pca_down.fit_transform(train_x))
        test_x_returned = pd.DataFrame(pca_down.transform(test_x))
    
        train_x_returned.columns = [''.join(['feature_',str(i)]) for i in list(train_x_returned.columns)]
        test_x_returned.columns = [''.join(['feature_', str(i)]) for i in list(test_x_returned.columns)]
    
    # Returned selected or regenerated features
    return train_x_returned, test_x_returned
    
main()