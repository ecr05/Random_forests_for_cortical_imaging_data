#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:18:15 2018

@author: er17
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_classif
import time

rand=42

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
                model=RandomForestClassifier(max_depth=d,max_features=f,n_estimators=1000,random_state=rand)
            else:
                model=RandomForestRegressor(max_depth=d,max_features=f,n_estimators=1000,random_state=rand)
    
            scores=cross_val_score(model, DATA, LABELS, cv=5,n_jobs=1)
            this_scores={'depth':d,'features': f,'scores': np.mean(scores)}
            cross_val = cross_val.append(this_scores, ignore_index=True)
            print(d,f,'scores', scores)
            
    
    opt_row=cross_val.loc[cross_val['scores'].idxmax()] 
    
    return opt_row['depth'],opt_row['scores']  

def feature_selection(kperc,trainingDATA,traininglabels,run_classification):
    
    trainingDATA_norm=normalize(trainingDATA)
    
# =============================================================================
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
# =============================================================================
    return featuresbest,f_mask
    
def train(args):
    
    if args.use_test_train_split:
        DATASET=pd.read_pickle(args.DATA)
    
    
        ALL_DATA=DATASET[DATASET.columns[args.data_range_train[0]:args.data_range_train[1]]].as_matrix()
        ALL_LABELS=DATASET[args.label_id].as_matrix()
    
        print('DATA shape', ALL_DATA.shape)
    #print('labels',ALL_LABELS)
    
        X_train, X_test, y_train, y_test = train_test_split(ALL_DATA, ALL_LABELS, test_size=0.1, random_state=rand)
    else:
        print('data paths:',args.train_data,args.test_data)
        TRAINING=pd.read_pickle(args.train_data)
        TESTING=pd.read_pickle(args.test_data)
        
        print('args.data_range',args.data_range_train,args.data_range_test,args.optimise,args.run_classification)
        # should be able to filter by column names
        X_train=TRAINING[TRAINING.columns[args.data_range_train[0]:args.data_range_train[1]]].as_matrix()
        X_test=TESTING[TESTING.columns[args.data_range_test[0]:args.data_range_test[1]]].as_matrix()
        #print(X_test)
        y_train=TRAINING[args.label_id].as_matrix()
        y_test=TESTING[args.label_id].as_matrix()
        print('DATA shapes', X_train.shape,X_test.shape)
        print('labels',y_train,type(y_train))
        
        
# =============================================================================
#     # run feature selection
#     if args.run_feature_select:
#         feature_selection(args.kperc,X_train,y_train,arg.run_classification)
# =============================================================================
    # optimise model
    if args.optimise:
        d,f=optimise_random_forest(X_train,y_train, args.depths,args.num_features,rand,args.run_classification)
    else:
        d=args.opt_depth
        f=args.opt_feat
    
    print('using max depth {} and max features {} '.format(d,f))
    # run model
    if args.run_classification==True:
        model=RandomForestClassifier(max_depth=d,max_features=f,n_estimators=1000,random_state=rand)
    else:
        model=RandomForestRegressor(max_depth=d,max_features=f,n_estimators=1000,random_state=rand)
    
    print('train is nan',np.where(np.isnan(X_train)==True))
    print('label is nan',np.where(np.isnan(y_train)==True))
    t0 = time.clock()
    model.fit(X_train,y_train)
    t1 = time.clock()
    print('training time = ', t1-t0)
    #pred_train=model.predict(X_train)
    pred=model.predict(X_test)
    print('orig labels', y_test)
    print('pred', pred)

    scores=[model.score(X_train,y_train),model.score(X_test,y_test)]
    print('Performance on train {} and test {} data'.format(scores[0],scores[1]))
    
    
# =============================================================================
#     x=range(y_train.min(),y_train.max(),1)
#     
#     plt.plot(pred,y_test,'+',pred_train,y_train,'or',x,x,'k')
#     plt.show()
# =============================================================================
    print(model.n_features_,'feature importances',np.argsort(model.feature_importances_)[::-1],type(np.argsort(model.feature_importances_)[::-1]))
    print(model.feature_importances_[np.argsort(model.feature_importances_)[::-1]])
    
    np.savetxt(os.path.join(args.output,'scores'),np.argsort(model.feature_importances_)[::-1])

    np.savetxt(os.path.join(args.output,'most_important_features.txt'),np.argsort(model.feature_importances_)[::-1])
    np.savetxt(os.path.join(args.output,'feature_importances_ordered.txt'),model.feature_importances_[np.argsort(model.feature_importances_)[::-1]])
# =============================================================================
#     # plot behaviour of most predictive features against label
#     for col in DATASET.columns[np.argsort(model.feature_importances_)[::-1]][0:5]:
#         print(col)
#         plt.plot(DATASET[args.label_id],DATASET[col].as_matrix(),'+')
#         plt.show()
# =============================================================================
        
    
if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='RF training script')
    parser.add_argument('--DATA', default='/data/PROJECTS/dHCP/PROCESSED_DATA/githubv1.1/ROI_data/roi_datamat_abscurv_wlabels.pickle')
    parser.add_argument('--train_data', default='/data/PROJECTS/dHCP/PROCESSED_DATA/githubv1.1/TRAINING_excl_all_PT_2ndscan/TRAINING_excl_all_PT_2ndscannwrois.pk1')
    parser.add_argument('--test_data', default='/data/PROJECTS/dHCP/PROCESSED_DATA/githubv1.1/TESTING_controls_nopreterms/TESTING_controls_nopretermsdfwrois.pk1')
    parser.add_argument('--data_range_train', nargs='+',type=int, default=[10,112])#106
    parser.add_argument('--data_range_test',nargs='+',type=int)#106
    parser.add_argument('--label_id', default='scan_ga')
    parser.add_argument('--run_feature_select', action='store_true')
    parser.add_argument('--optimise', action='store_true')
    parser.add_argument('--depths', nargs='+', default=[3], type=int)
    parser.add_argument('--num_features',  nargs='+', default=[10], type=int)
    parser.add_argument('--opt_depth', default=7)
    parser.add_argument('--opt_feat', default='auto')
    parser.add_argument('--kperc', default=90)
    parser.add_argument('--run_classification', action='store_true')
    parser.add_argument('--use_test_train_split', action='store_true')
    parser.add_argument('--output')
     
    args = parser.parse_args()
   
    if args.data_range_test is None:
        args.data_range_test=args.data_range_train
     
    # Call training
    train(args)
     