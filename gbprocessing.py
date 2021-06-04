#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 09:03:26 2019

@author: er17
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def read_data(fname,lname,id_name,label_id,confounds=None,id_struct=None):
    ''' read in data accordinging to file format
        output data and labels as single data
    '''
        
    if (fname.find('pk1') != -1) or (fname.find('pickle') != -1): 
        DATA=pd.read_pickle(fname)
    elif fname.find('csv') != -1: 
        DATA=pd.read_csv(fname)
    else:
        raise ValueError('read_data:filetype not currently recognised')
        
    if (lname.find('pk1')!=-1) or (lname.find('pickle') != -1): 
        L_FRAME=pd.read_pickle(lname)
    elif fname.find('csv') != -1: 
        L_FRAME=pd.read_csv(lname)
    else:
        raise ValueError('read_data:filetype not currently recognised')
    
    print('confounds', confounds)
    columns=[]

    if isinstance(confounds, list):
        columns=confounds
    elif confounds is not None:
        columns.append(confounds)
   
    columns.append(label_id)
    columns.append(id_name)    
    DATA=DATA.merge(L_FRAME[columns], on=[id_name],suffixes=('', '_y'))
    DATA.drop(list(DATA.filter(regex='_y$',axis=1)), axis=1, inplace=True)
    
    print(DATA)
      
    return DATA
        
def get_train_and_test(ALLDATA,id_name,label_id,size,train_subjs=None,test_subjs=None):
    
    if (train_subjs is not None) and (test_subjs is not None) : 
        
        if (train_subjs.find('txt') != -1) :
            training_subjects=np.loadtxt(train_subjs,dtype=str);
            testing_subjects=np.loadtxt(test_subjs,dtype=str);
        if (train_subjs.find('pk1') != -1) or (train_subjs.find('pickle') != -1): 
            training_subjects=pd.read_pickle(train_subjs)   
            testing_subjects=pd.read_pickle(test_subjs)     
            
        TRAINING=ALLDATA[ALLDATA[id_name].isin(training_subjects)]
        TESTING=ALLDATA[ALLDATA[id_name].isin(testing_subjects)]
        y_train=TRAINING[label_id].to_numpy()
        X_train=TRAINING.drop(columns=[id_name,label_id]).to_numpy()
        
        y_test=TESTING[label_id].to_numpy()
        X_test=TESTING.drop(columns=[id_name,label_id]).to_numpy()
    else:
        print('splitting data randomly',label_id,np.where(np.isnan(ALLDATA[label_id])==True))
        LABELS=ALLDATA[label_id].to_numpy()
        DATA=ALLDATA.drop(columns=[id_name,label_id]).to_numpy()
    
        X_train, X_test, y_train, y_test = train_test_split(DATA, LABELS, test_size=size, random_state=42)
        
    return X_train,X_test, y_train, y_test 
               
        
def normalize(data,norm_axis=0):
    """
        normalise feature maps

        Parameters
        ----------
        data : data as n_samples, n_features

        Returns
        -------
        datanormed : normalised data
    """

    datastd = np.std(data,axis=norm_axis)
    print('in normalise', norm_axis, datastd.shape )
    print('mean',np.mean(data,axis=norm_axis).shape )
    if np.where(datastd == 0)[0].shape != (0,):
        print('isnan')
        datastd[np.where(datastd==0)]=1;

    datanormed = (data - np.mean(data,axis=norm_axis)) / datastd
    if np.where(np.isnan(datanormed))[0].shape != (0,):
         print('isnan2')

    print('mean normalised',np.mean(datanormed,axis=norm_axis))
    print('std normalised',np.std(datanormed,axis=norm_axis))

    
    return datanormed

def optimise_random_forest(DATA,LABELS, depths,num_features,rand,run_classification=True):
    
    cross_val = pd.DataFrame(columns=['d','f','mean score']) 

    for d in depths:
        for f in num_features:
            if run_classification==True:
                model=GradientBoostingClassifier(max_depth=d,max_features=f,n_estimators=1000,random_state=rand)
            else:
                model=GradientBoostingRegressor(max_depth=d,max_features=f,n_estimators=1000,random_state=rand)
    
            scores=cross_val_score(model, DATA, LABELS, cv=5,n_jobs=1)
            this_scores={'depth':d,'features': f,'scores': np.mean(scores)}
            cross_val = cross_val.append(this_scores, ignore_index=True)
            print(d,f,'scores', np.mean(scores))
            
    
    opt_row=cross_val.loc[cross_val['scores'].idxmax()] 
    
    return opt_row['depth'],opt_row['features']  

def run_PCA(trainingData,testData):
    pca = PCA(n_components=trainingData.shape[0])
    new_trainingdata=pca.fit_transform(trainingData)  
    new_testdata=pca.transform(testData)  
    
    return new_trainingdata,new_testdata
      
# =============================================================================
# def feature_selection(kperc,trainingDATA,traininglabels,run_classification):
# 
#     if run_classification == True:
#         kbest=SelectPercentile(f_classif, percentile=kperc)  
#     else:
#         
#     featuresbest = kbest.fit_transform(trainingDATA,  traininglabels)
#     f_mask=kbest.get_support() # mask of features used
#     scores = -np.log10(kbest.pvalues_)
#     scores /= scores.max()
#     
#     X_indices = np.arange(traininglabels.shape[-1])
#     plt.bar(X_indices - .45, scores, width=.2,
#         label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
#         edgecolor='black')
#     
#     return featuresbest,f_mask
# =============================================================================

def project_netmats_to_feat_vector(netmats,dim,norm=False):
    ''' takes flattened netmats
        removes lower triangular values and outputs
        feature vectors
        
        Parameters
        ----------
        netmats : matrix of flattened netmats (nxdimxdim) where n = # of subjects and dim is matrix size
        dim: square matrix dimensions
        norm: optionally normalise?? 

        Returns
        -------
        features : upper triangular values only
        
    '''
    features=[]
    for i in np.arange(netmats.shape[0]):
        b=np.reshape(netmats[i,:],(dim,dim))
        
        features.append(b[np.triu_indices(360)])
        
    return np.asarray(features)
    
    
def project_netmats_feats_to_matrix(features):
    ''' takes important features from rf
        and maps back to netmat indices
    '''
    
