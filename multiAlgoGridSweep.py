
import numpy as np
import sklearn.linear_model as linear
import sklearn.ensemble as ensemble
from sklearn import cross_validation
import pandas as pd
import sys
from collections import deque
import random
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import scipy.ndimage.filters as flt
from collections import defaultdict
#from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVR
import pickle as pk

training_file = 'C:/Users/kPasad/Box Sync/ML/Projects/africanSoilPred/data/training.csv'
test_file     = 'C:/Users/kPasad/Box Sync/ML/Projects/africanSoilPred/data/sorted_test.csv'

df_train = pd.read_csv(training_file,tupleize_cols =True)
df_test = pd.read_csv(test_file)
train_dims = df_train.shape

targets = ['Ca','P','pH','SOC','Sand']
train_cols_to_remove = ['PIDN']+targets

x_train=df_train.drop(train_cols_to_remove,axis=1)
y_train=df_train[targets]
train_feature_list = list(x_train.columns)
spectra_features  = list(train_feature_list)
non_spectra_feats=['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI','Depth'] 
for feats in non_spectra_feats:
    spectra_features.remove(feats)

#fltSpectra=flt.gaussian_filter1d(np.array(x_train[spectra_features]),sigma=10,order=1)    
fltSpectra = np.array(x_train[spectra_features])

x_train["Depth"] = x_train["Depth"].apply(lambda depth:0 if depth =="Subsoil" else 1)
x_train[spectra_features]=fltSpectra

n_train = df_train.shape[0]
cv_factor = 0.7
num_cv_folds =20
train_sample_idx = range(0,n_train)
num_cv_train_samples = int(df_train.shape[0]*cv_factor)
train_sample_idx = deque(range(0,n_train))

#pca = RandomizedPCA(n_components=400)

feat_imp=np.zeros([num_cv_folds,len(train_feature_list)])

algos = ['bayesianRidge'',adaBoost','decisionTree','gradBoost','extraTree','linear','ridge','svr','randForest']
clf  = defaultdict.fromkeys(algos)
clf['adaBoost']  = ensemble.AdaBoostRegressor()
clf['decisionTree'] = DecisionTreeRegressor(random_state=0)
clf['gradBoost'] = ensemble.GradientBoostingRegressor(loss='huber',max_depth=2,n_estimators=500)
clf['extraTree'] =ensemble.ExtraTreesRegressor(n_estimators=20)
clf['linear']=linear.LinearRegression()
clf['bayesianRidge'] = linear.BayesianRidge()
clf['ridge'] = linear.Lasso(alpha=0.1)
clf['svr'] = SVR(C=1000,kernel='poly',degree=5)
clf['randForest']=ensemble.RandomForestRegressor(n_estimators=10, criterion='mse')

algosToTry = ['svr']
feat_imp = pk.load(open('feature_imp_ca1.pk','r'))      
mean_imp = mean(feat_imp,axis=0)
sortIdx = np.argsort(mean_imp)
        
for reg in algosToTry:
    error= pd.DataFrame(columns=targets)
    for cv_fold_idx in range(0,num_cv_folds):
        num_samples_to_shift = int( random.uniform(-1,1)*n_train) #Generate a random, bidirectional, circular shift
        train_sample_idx.rotate(num_samples_to_shift) #Shift the data row indices by this random amount
        rand_train_sample_idx=list(train_sample_idx)  #Convert deque to list
        random.shuffle(rand_train_sample_idx)
        
        cv_train_sample_idx = rand_train_sample_idx[0:num_cv_train_samples]
        cv_test_sample_idx =  rand_train_sample_idx[num_cv_train_samples:n_train]
                  
        x_train_cv = x_train.ix[cv_train_sample_idx,train_feature_list]
        y_train_cv = y_train.ix[cv_train_sample_idx,targets]
        dims =  x_train_cv.shape
        print "transforming"        
        #x_train_cv = pca.fit_transform(x_train_cv)        
        
        x_test_cv  =x_train.ix[cv_test_sample_idx,train_feature_list]
        y_test_cv  =y_train.ix[cv_test_sample_idx,targets]
        #x_test_cv = pca.transform(x_test_cv)
        
        targetVarError=[]
        
        for targetVar in targets:
        #for targetVar in ['P']:
          #print 'fitting ',targetVar
          #clf[reg].fit(x_train_cv,y_train_cv[targetVar])
          #feat_imp[cv_fold_idx,range(len(x_train_cv.columns))]=clf['randForest'].feature_importances_
          #sortIdx = np.argsort(feat_imp[cv_fold_idx])
          
          #top_preds = x_test_cv.columns[sortIdx[0:1000]]
          top_preds = x_test_cv.columns
          x_train_cv = x_train.ix[cv_train_sample_idx,top_preds]
          x_test_cv  =x_train.ix[cv_test_sample_idx,top_preds]  
          
          #x_train_cv = x_train.ix[cv_train_sample_idx,spectra_features]
          #x_test_cv  =x_train.ix[cv_test_sample_idx,spectra_features]           
          
          print 'refitting ',targetVar
          clf[reg].fit(x_train_cv,y_train_cv[targetVar])
           
          print 'predicting'
          pred = clf[reg].predict(x_test_cv)
    
    
          targetVarError.append(sqrt(mean(pow((pred-np.array(y_test_cv[targetVar])),2),axis=0)))
          print targetVarError
        
        error.loc[cv_fold_idx]=targetVarError
    
    error.loc[cv_fold_idx+1]=mean(error)
    error.to_csv(reg+'.csv')
