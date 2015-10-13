
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
import myutils
import itertools

#load the files
training_file = 'C:/Users/kPasad/Box Sync/ML/Projects/africanSoilPred/data/training.csv'
test_file     = 'C:/Users/kPasad/Box Sync/ML/Projects/africanSoilPred/data/sorted_test.csv'
feat_imp = pk.load(open('C:/Users/kpasad/Box Sync/ML/Projects/africanSoilPred/data/feat_imp.pk','r')) 

df_train = pd.read_csv(training_file,tupleize_cols =True)
df_test = pd.read_csv(test_file)
train_dims = df_train.shape
algos = ['bayesianRidge,adaBoost','decisionTree','gradBoost','extraTree','linear']
targets = ['Ca','P','pH','SOC','Sand']


#All static data structures here
clf  = defaultdict.fromkeys(algos)
top_preds=defaultdict.fromkeys(targets)
master_obj = master()


#All parameters go here.
derivative_filt = 'disable'
feat_list = 'all'
cv_factor = 0.7
num_cv_folds =20
learner_id=0 


#Feature massage
train_cols_to_remove = ['PIDN']+targets
x_train=df_train.drop(train_cols_to_remove,axis=1) #Remove the training sample ID and the targets.
y_train=df_train[targets] #Extract the targets
if derivative_filt =='enable':

   train_feature_list = list(x_train.columns)
   spectra_features  = list(train_feature_list) #temp assign
   non_spectra_feats=['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI','Depth'] 
   for feats in non_spectra_feats:
      spectra_features.remove(feats) #Remove the non-spectra feats one at a time. (no easy way to remove variables from list)
   
   fltSpectra=flt.gaussian_filter1d(np.array(x_train[spectra_features]),sigma=10,order=1) #apply filter to x_train with only spectral features 
   x_train[spectra_features]=fltSpectra

x_train["Depth"] = x_train["Depth"].apply(lambda depth:0 if depth =="Subsoil" else 1) #Categoritcal to intege5


#Bootstrap parameters
n_train = df_train.shape[0]
train_sample_idx = range(0,n_train)
num_cv_train_samples = int(df_train.shape[0]*cv_factor)
train_sample_idx = deque(range(0,n_train))

#Initialise the classifiers
clf['adaBoost']  = ensemble.AdaBoostRegressor()
clf['decisionTree'] = DecisionTreeRegressor(random_state=0)
clf['gradBoost'] = ensemble.GradientBoostingRegressor(loss='huber',max_depth=2,n_estimators=50)
clf['extraTree'] =ensemble.ExtraTreesRegressor(n_estimators=20)
clf['linear']=linear.LinearRegression()
clf['bayesianRidge'] = linear.BayesianRidge()
clf['ridge'] = linear.Lasso(alpha=0.1)
clf['randForest']=ensemble.RandomForestRegressor(n_estimators=10, criterion='mse')


master_obj.add_variables(derivative_filt=derivative_filt,cv_factor=cv_factor,num_cv_folds=num_cv_folds)
algosToTry=['svr']
svrParams_C =[100,1000,10000]
svrDegree =[1,2,3,4,5]
gradBoost_maxD = [2,3,4]
gradBoost_n_est = [50,100,300]

max_predictors = len(x_train.columns)
C=10000
#for max_predictors in [len(x_train.columns)]:
# for C in svrParams_C:
for svr_kernel in ['poly',  'sigmoid']:     
   for degree in svrDegree:       
        clf['svr'] = SVR(C=C,kernel=svr_kernel,degree=degree)
        
        for reg in algosToTry:    
            error= pd.DataFrame(columns=targets)
            
            for cv_fold_idx in range(0,num_cv_folds):
                num_samples_to_shift = int( random.uniform(-1,1)*n_train) #Generate a random, bidirectional, circular shift
                train_sample_idx.rotate(num_samples_to_shift) #Shift the data row indices by this random amount
                rand_train_sample_idx=list(train_sample_idx)  #Convert deque to list
                random.shuffle(rand_train_sample_idx)
                
                cv_train_sample_idx = rand_train_sample_idx[0:num_cv_train_samples]
                cv_test_sample_idx =  rand_train_sample_idx[num_cv_train_samples:n_train]
                          
                y_train_cv = y_train.ix[cv_train_sample_idx,targets]
                y_test_cv  =y_train.ix[cv_test_sample_idx,targets]
                
                targetVarError=[]
                
                for targetVar in targets:
                  sortIdx = np.argsort(mean(feat_imp[targetVar],axis=0))          
                  top_preds[targetVar] = list(x_train.columns[sortIdx[0:max_predictors]])  
                  
                  x_train_cv = x_train.ix[cv_train_sample_idx,top_preds[targetVar]]
                  x_test_cv  =x_train.ix[cv_test_sample_idx,top_preds[targetVar]]  
                  
                  print 'CV fold = %d, fitting %s '%(cv_fold_idx,targetVar)
                  clf[reg].fit(x_train_cv,y_train_cv[targetVar])
                  pred = clf[reg].predict(x_test_cv)
                      
                  targetVarError.append(sqrt(mean(pow((pred-np.array(y_test_cv[targetVar])),2),axis=0)))
                  
                error.loc[cv_fold_idx]=targetVarError
            
            error_mean=dict(zip(targets,np.array(mean(error)).tolist()))
            error_std =dict(zip(targets,np.array( std(error)).tolist()))
            error_var =dict(zip(targets,np.array( var(error)).tolist()))
        
            for targetVar in targets:
                learner_id = master_obj.add_new_learner()        
                master_obj.learner[learner_id].add_new_var(targetVar=targetVar,algo=reg,rmse=error_mean[targetVar],std=error_std[targetVar],var=error_var[targetVar],C=C,degree=degree,max_predictors=max_predictors,svr_kernel=svr_kernel)
                master_obj.writeToFile()
