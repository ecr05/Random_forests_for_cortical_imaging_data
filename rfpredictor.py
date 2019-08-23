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
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import mean_absolute_error

import rfprocessing as rf

import time

rand=42


    
def train(args):
    
    ALL_DATA=rf.read_data(args.DATA,args.LABELS,args.id,args.label_id, args.confounds)
    
    X_train, X_test, y_train, y_test=rf.get_train_and_test(ALL_DATA, args.id,args.label_id,args.test_split,args.train_subjs,args.test_subjs)
    
    if args.netmats:
         X_train=rf.project_netmats_to_feat_vector(X_train,args.netmats_dim)
         X_test=rf.project_netmats_to_feat_vector(X_test,args.netmats_dim)
    
        
    # run feature selection
    if args.run_feature_select:
        print('run pca, new data size 1',X_train.shape,X_test.shape)

        X_train,X_test=rf.run_PCA(X_train,X_test)
        print('run pca, new data size',X_train.shape,X_test.shape)
        #feature_selection(args.kperc,X_train,y_train,arg.run_classification)
    # optimise model
    if args.optimise:
        d,f=rf.optimise_random_forest(X_train,y_train, args.depths,args.num_features,rand,args.run_classification)
    else:
        print("don't optimise",type(args.opt_feat),type(args.opt_depth),args.opt_feat)
        d=args.opt_depth
        f=args.opt_feat
        
    print('using max depth {} and max features {} '.format(d,f))
    print('shape',X_train.shape,f,args.opt_feat)
    # run model
    if args.run_classification==True:
        model=RandomForestClassifier(max_depth=d,max_features=f,n_estimators=1000,random_state=rand)
    else:
        model=RandomForestRegressor(max_depth=d,max_features=f,n_estimators=1000,random_state=rand)
    

    t0 = time.clock()
    model.fit(X_train,y_train)
    t1 = time.clock()
    print('training time = ', t1-t0)
    pred_train=model.predict(X_train)
    pred=model.predict(X_test)
    print('orig labels', y_test)
    print('pred', pred)

    if args.run_classification==True:
        scores=[model.score(X_train, y_train),model.score(X_test, y_test)]
    else:
        scores=[mean_absolute_error(y_train, pred_train),mean_absolute_error(y_test,pred)]
    
    print('Performance on train {} and test {} data'.format(scores[0],scores[1]))
    
    
    #x=range(y_train.min(),y_train.max(),1)
    
    plt.plot(pred,y_test,'ro',pred_train,y_train,'kx')#,x,x,'k')
    plt.show()
    print(model.n_features_,'feature importances',np.argsort(model.feature_importances_)[::-1],type(np.argsort(model.feature_importances_)[::-1]))
    print(model.feature_importances_[np.argsort(model.feature_importances_)[::-1]])
    
    np.savetxt(os.path.join(args.OUTPUT,'scores'),np.argsort(model.feature_importances_)[::-1])

    np.savetxt(os.path.join(args.OUTPUT,'most_important_features.txt'),np.argsort(model.feature_importances_)[::-1])
    np.savetxt(os.path.join(args.OUTPUT,'feature_importances_ordered.txt'),model.feature_importances_[np.argsort(model.feature_importances_)[::-1]])
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
    parser.add_argument('DATA', help='path to data csv')
    parser.add_argument('LABELS',help='path to labels csv')
    parser.add_argument('OUTPUT',help='output path')
    parser.add_argument('--train_subjs',help='list of training subjects' )
    parser.add_argument('--test_subjs', help='list of test subjects')
    parser.add_argument('--id', default='id')
    parser.add_argument('--label_id', default='scan_ga')
    parser.add_argument('--test_split', default=0.1)
    parser.add_argument('--netmats',  action='store_true')
    parser.add_argument('--netmats_dim',default=360)
    parser.add_argument('--run_feature_select', action='store_true')
    parser.add_argument('--optimise', action='store_true')
    parser.add_argument('--depths', nargs='+', default=[3], type=int)
    parser.add_argument('--num_features',  nargs='+', default=[1.0], type=float)
    parser.add_argument('--opt_depth', default=7,type=int)
    parser.add_argument('--opt_feat', default=1.0, type=float)
    parser.add_argument('--kperc', default=90)
    parser.add_argument('--run_classification', action='store_true')
    parser.add_argument('--use_test_train_split', action='store_true')
    parser.add_argument('--confounds',nargs='+', help='list of confounding labels')
     
    args = parser.parse_args()
     
    # Call training
    train(args)
     