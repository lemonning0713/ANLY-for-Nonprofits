import numpy as np
import pandas as pd
import math
import glob
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import shap

############################## Data Processing ##############################
# Read data and calculate ratio mass/body length
raw = pd.read_csv('CleanDatav3.csv', sep=',',index_col = 0)#raw.columns
raw['ratio'] = raw['Mass']/raw['Body.Length']
select = raw.loc[:,('Phase..', 'Distance', 'S_fuel','S_hour', 'S_peeps', 'flights', 'Temp', 'Luna', 'Elev',
                    'Genus', 'Height', 'ratio')]
select.loc[select['Phase..'] == 5].S_fuel = select.loc[select['Phase..'] == 5].S_fuel.fillna(0)
# Replace missing value with mean
for i in select.Genus.unique():
    mask = (select['Genus'] == i)
    height_mean = select.loc[mask, 'Height'].mean()
    ratio_mean = select.loc[mask, 'ratio'].mean()
    select['Height'] = select['Height'].mask(mask,select['Height'].fillna(height_mean))
    select['ratio'] = select['ratio'].mask(mask,select['ratio'].fillna(ratio_mean))
select = select.dropna()
select['Phase..'].value_counts()
############################## Build Gradient Boosting Machine (GBM) model ##############################
y = select["ratio"]
X = select.drop(["Height", "Genus", "ratio"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
names = X_train.columns

cv_params = {'learning_rate': [0.1, 0.05,0.01], 'max_depth': [1,3,5,7,10], 'n_estimators': [100,200,300]}
ind_params = {'random_state': 10}
optimized_GBM = GridSearchCV(GradientBoostingRegressor(**ind_params),
							cv_params,
							scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=10) 
optimized_GBM.fit(X_train, y_train)
print(optimized_GBM.cv_results_ ['mean_test_score'].max())
# Optimized_GBM.cv_results
means = optimized_GBM.cv_results_['mean_test_score']
params = optimized_GBM.cv_results_['params']

for mean,param in zip(means,params):
    print("%f  with:   %r" % (mean,param))

params = {'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.05, 'loss': 'huber', 'random_state': 1}
clf = GradientBoostingRegressor(**params, verbose=0)
clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
#print("MSE: %.4f" % mse)
importance_frame = pd.DataFrame({'Importance': list(clf.feature_importances_), 'Feature': list(names)})
importance_frame.sort_values(by='Importance', inplace=True)
# Feature importance plot
importance_frame.plot(kind='barh', x='Feature', figsize=(8, 8), color='orange') #plt.show()
plt.savefig('ratio-FeatureImportance')
############################## Partial Dependence Plot ##############################
# 1-dimension: relationship between each variable and 'ratio'
feature_importance = clf.feature_importances_
from pdpbox import pdp, get_dataset, info_plots
for i in names:    
    pdp_goals = pdp.pdp_isolate(model = clf, dataset=X, 
                            model_features=names, feature=i)
    pdp.pdp_plot(pdp_goals, i) #plt.show()
    plt.savefig('ratio-pdp'+i)
# 2-dimention: Elevation & Distance
features_to_plot = ['S_peeps', 'S_fuel']
inter2  =  pdp.pdp_interact(model=clf, dataset=X, model_features=names, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter2, feature_names=features_to_plot, plot_type='grid')#plt.show()
plt.savefig('ratio-pdp-PeepsFuel')
############################## Shapley Value ##############################
shap_values = shap.TreeExplainer(clf).shap_values(X_train)
shap.summary_plot(shap_values, X_train) #plt.show()
plt.savefig('ratio-shap')

