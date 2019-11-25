# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:29:27 2019

@author: Jimny
"""

## Import needed packages
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.linear_model import LinearRegression
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

from sklearn.metrics import mutual_info_score,roc_auc_score,confusion_matrix,accuracy_score, roc_curve, auc, classification_report

from scipy import interp

# Pack things into the main function
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
    
    # CV on train set
    # Initialize a dataframe to record prediction probabilities
    oof_xgb = np.zeros(len(y_train))
    oof_lgb = np.zeros(len(y_train))
    oof_rf = np.zeros(len(y_train))

    # Define a 5-fold cross validatio scheme
    folds = KFold(n_splits=5,shuffle=True,random_state=30)
    
    train_columns = X_train.columns.values
    feature_importance_df = pd.DataFrame()
    
    # Iterate through each folder for cross-validation train and prediction
    for fold_,(trn_idx,val_idx) in enumerate(folds.split(X_train.values,y_train.values)):
        # Print which folder is being predicted
        print("fold n^{}".format(fold_))
        
        # Assign train and validation dataset
        x_trn,y_trn = X_train.iloc[trn_idx],y_train.iloc[trn_idx]
        x_val,y_val = X_train.iloc[val_idx],y_train.iloc[val_idx]
    
        print("start training...")
        
        # Define three classifiers
        # XGBoost
        clf_xgb = xgb.XGBClassifier(booster='gbtree',
                                       objective= 'binary:logistic',
                                       eval_metric='logloss',
                                       tree_method= 'auto',
                                       max_depth= 6,
                                       min_child_weight= 1,
                                       gamma = 0,
                                       subsample= 1,
                                       colsample_bytree = 1,
                                       reg_alpha = 0,
                                       reg_lambda = 1,
                                       learning_rate = 0.1,
                                       seed=27)
        
        # LightGBM
        clf_lgb = lgb.LGBMClassifier(objective = 'binary',
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
                                        random_state = 117)
        
        # Random Forest
        clf_rf = RandomForestClassifier(n_estimators='warn', criterion='gini', 
                                        max_depth=None, min_samples_split=2, 
                                        min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                        max_features='auto', random_state = 117)
        
        # Fit using train dataset and make prediction on validation fold
        clf_xgb.fit(x_trn,y_trn)
        clf_lgb.fit(x_trn,y_trn)
        clf_rf.fit(x_trn,y_trn)
        

        #feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = train_columns
        fold_importance_df["importance"] = clf_lgb.feature_importances_[:len(train_columns)]
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        # Record fitted probabilities
        oof_xgb[val_idx] = clf_xgb.predict_proba(x_val)[:,1]
        oof_lgb[val_idx] = clf_lgb.predict_proba(x_val)[:,1]
        oof_rf[val_idx] = clf_rf.predict_proba(x_val)[:,1]
        
    # Mappinng all elements of y_train into int
    y_train = list(map(int, y_train))
    
    # Print AUC scores
    print("XGB Train CV AUC score: {:<8.5f}".format(roc_auc_score(y_train,oof_xgb)))
    print("LGB Train CV AUC score: {:<8.5f}".format(roc_auc_score(y_train,oof_lgb)))
    print("RF Train CV AUC score: {:<8.5f}".format(roc_auc_score(y_train,oof_lgb)))
    
    # This line is now muted but was used to determine optimized threshold
    #det_thres(y_train, oof_rf, threshold = np.arange(0.3,0.6,0.01))
    
    # Feature Importance Plot
    cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance", ascending=False)[:15].index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)].sort_values(by="importance",ascending=False)
    plt.figure(figsize=(14,9))
    sns.barplot(x="importance", y="Feature", 
                data=best_features)
    plt.title('LightGBM Features Ranking (Top 15 Useful Features)', fontsize =16)
    plt.xlabel('Impportance by LightGBM',fontsize=13)
    plt.ylabel('Feature',fontsize=13)
    #plt.tight_layout()
    plt.show()

    # Initialize prediciton results
    pred_xgb = np.zeros(len(y_train))
    pred_lgb = np.zeros(len(y_train))
    pred_rf = np.zeros(len(y_train))
    
    # Set threshold for three classifiers
    thres_xgb = 0.44
    thres_lgb = 0.44
    thres_rf = 0.52
    
    # Make prediction by probability
    pred_xgb[oof_xgb >= thres_xgb] = 1
    pred_lgb[oof_lgb >= thres_lgb] = 1
    pred_rf[oof_rf >= thres_rf] = 1
    
    # ROC Curve for train
    plt.figure(figsize=(8,7))
    draw_roc(y_train, oof_xgb, 'XGBoost', 'royalblue', '-')
    draw_roc(y_train, oof_lgb, 'LightGBM', 'lightcoral', '--')
    draw_roc(y_train, oof_rf, 'Random Forest', 'seagreen', '-.')
    
    plt.plot([0, 1], [0, 1], 'k--', lw = 4)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Train Data Cross Validation')
    plt.legend(loc="lower right", fontsize = 14, handlelength=4)
    plt.show()
    
    
    # Print evaluation statistics
    print('XGB Train Cross Validation results: ')
    print_validate(y_train, pred_xgb, 'XGBoost on Train', colors = "Blues")
    
    # Print evaluation statistics
    print('LGB Train Cross Validation results: ')
    print_validate(y_train, pred_lgb, 'LightGBM on Train', colors = "Reds")
    
    # Print evaluation statistics
    print('RF Train Cross Validation results: ')
    print_validate(y_train, pred_rf, 'Random Forest on Train', colors = "Greens")
    
    ################### Test Data ###################
    # Define classifier again for test data
    clf_xgb = xgb.XGBClassifier(booster='gbtree',
                                       objective= 'binary:logistic',
                                       eval_metric='logloss',
                                       tree_method= 'auto',
                                       max_depth= 6,
                                       min_child_weight= 1,
                                       gamma = 0,
                                       subsample= 1,
                                       colsample_bytree = 1,
                                       reg_alpha = 0,
                                       reg_lambda = 1,
                                       learning_rate = 0.1,
                                       seed=27)
        
        
    clf_lgb = lgb.LGBMClassifier(objective = 'binary',
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
                                random_state = 117)
        
    clf_rf = RandomForestClassifier(n_estimators='warn', criterion='gini', 
                                    max_depth=None, min_samples_split=2, 
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                    max_features='auto', random_state = 117)
    
    # Fit using whole train dataset
    clf_xgb.fit(X_train,y_train)
    clf_lgb.fit(X_train,y_train)
    clf_rf.fit(X_train,y_train)
    
    # Record predicted probabilities
    xgb_test_prob = clf_xgb.predict_proba(X_test)[:,1]
    lgb_test_prob = clf_lgb.predict_proba(X_test)[:,1]
    rf_test_prob = clf_rf.predict_proba(X_test)[:,1]
    
    # Initialize prediction results from three classifiers
    pred_xgb_test = np.zeros(len(y_test))
    pred_lgb_test = np.zeros(len(y_test))
    pred_rf_test = np.zeros(len(y_test))
    
    # Make predictions
    pred_xgb_test[xgb_test_prob >= thres_xgb] = 1
    pred_lgb_test[lgb_test_prob >= thres_lgb] = 1
    pred_rf_test[rf_test_prob >= thres_rf] = 1
    
    # Mappinng all elements of y_train into int
    y_test = list(map(int, y_test))
    
    # ROC Curves for test dataset
    plt.figure(figsize=(8,7))
    draw_roc(y_test, xgb_test_prob, 'XGBoost', 'royalblue', '-')
    draw_roc(y_test, lgb_test_prob, 'LightGBM', 'lightcoral', '--')
    draw_roc(y_test, rf_test_prob, 'Random Forest', 'seagreen', '-.')
    
    plt.plot([0, 1], [0, 1], 'k--', lw = 4)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Test Result')
    plt.legend(loc="lower right", fontsize = 14, handlelength=4)
    plt.show()
    

    # Print evaluation statistics
    print('\n\nTest results: ')
    print('XGB Test Cross Validation results: ')
    print_validate(y_test, pred_xgb_test, 'XGBoost on Test', 
                   colors = "Blues",v_range = [0,160])
    
    # Print evaluation statistics
    print('LGB Test Cross Validation results: ')
    print_validate(y_test, pred_lgb_test, 'LightGBM on Test', 
                   colors = "Reds",v_range = [0,160])
    
    # Print evaluation statistics
    print('RF Test Cross Validation results: ')
    print_validate(y_test, pred_rf_test, 'Random Forest on Test', 
                   colors = "Greens",v_range = [0,160])
    

# This function is used to determine threshold for each classifier
def det_thres(y_true, oof_prob, threshold = np.arange(0.3,0.6,0.01)):
    
    for x in threshold:
        y_pred = np.zeros(len(y_true))
        y_pred[oof_prob>=x]=1

        ans=confusion_matrix(y_true,y_pred)
        ans = ans.astype(float)
        
        print("\nThreshold = ","{0:.2f}".format(x))
        print(ans)
        print("True positive ratio is "+str(ans[1,1]/(ans[1,0]+ans[1,1])))
        print("True negative ratio is "+str(ans[0,0]/(ans[0,0]+ans[0,1])))
        print("Overall correct is "+str((ans[0,0]+ans[1,1])/(ans[0,0]+ans[1,0]+ans[0,1]+ans[1,1])))

# This function is used to draw ROC curves
def draw_roc(y_true, y_proba, clf_name, col, style):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr,  lw=4, color = col, linestyle = style,
             label='ROC of {0} (AUC = {1:0.2f})'
                 ''.format(clf_name, roc_auc))
    
# This function is used to print validation/test resuls
def print_validate(Y_val, predictions, plt_title, 
                   colors = "Blues", v_range = [0,600]):
    conf_mat = confusion_matrix(Y_val, predictions)
    print("\nAccuracy score: ",accuracy_score(Y_val, predictions))
    print("\nConfusion Matrix: \n",conf_mat)
    print(classification_report(Y_val, predictions))

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(predictions, Y_val,
                          title='Confusion Matrix of '+plt_title,
                          col_map = colors, v_range = v_range)

def plot_confusion_matrix(y_pred, y_true, title, col_map = "Blues", 
                          v_range = [0,600]):
    conf_mat = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(conf_mat, columns=['Non-Fatal','Fatal'], index = ['Non-Fatal','Fatal'])
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (6,5))
    plt.title(title, fontsize = 16)
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, cmap=col_map, annot=True,annot_kws={"size": 16}, fmt='g',
                vmin=v_range[0], vmax=v_range[1])# font size
  

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