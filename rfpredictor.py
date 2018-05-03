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

rand=42

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
    
        y_train=TRAINING[args.label_id].as_matrix()
        y_test=TESTING[args.label_id].as_matrix()
        print('DATA shapes', X_train.shape,X_test.shape)
        #print('labels',ALL_LABELS)
    
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
        
    model.fit(X_train,y_train)
    pred_train=model.predict(X_train)
    pred=model.predict(X_test)
    
    x=range(28,44,1)
    
    plt.plot(pred,y_test,'+',pred_train,y_train,'or',x,x,'k')
    plt.show()
    print(model.n_features_,'feature importances',np.argsort(model.feature_importances_)[::-1])
    print(model.feature_importances_[np.argsort(model.feature_importances_)[::-1]])
    print(DATASET.columns[np.argsort(model.feature_importances_)[::-1]][0:10])
    
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
    parser.add_argument('--data_range_train', default=[5,107])#106
    parser.add_argument('--data_range_test', default=[10,112])#106
    parser.add_argument('--label_id', default='scan_ga')
    parser.add_argument('--optimise', action='store_true')
    parser.add_argument('--depths', default=[3])
    parser.add_argument('--num_features', default=[10])
    parser.add_argument('--opt_depth', default=7)
    parser.add_argument('--opt_feat', default='auto')
    parser.add_argument('--run_classification', action='store_true')
    parser.add_argument('--use_test_train_split', action='store_true')
     
    args = parser.parse_args()
   

    # Call training
    train(args)
     