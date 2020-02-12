#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:35:59 2020
Code for mapping most important features back to the image domain 
- currently would not work if you had done feature selection
@author: er17
"""

import pandas as pd
import numpy as np
import nibabel
import argparse


def main(args):
    print(args.features,args.dataframe,args.labels)
    ######### load files ######################################
    if args.importances is not None:
        importances=np.load(args.importances)
    
    most_important_features=np.load(args.features)
    data=pd.read_pickle(args.dataframe)
    labels=nibabel.load(args.labels)
    
    ## return data frame columns that contain ROI names
    feature_names=list(data.columns.values[:-1])
    
    # create new data array which has size (num_channels,vertices) and set to zeros
    features_image=np.zeros((args.num_channels,labels.darrays[0].data.shape[0]))
    
    for index,roi in enumerate(most_important_features[:args.n_features]):
        channel=int(feature_names[roi][3])
        print(index,roi,channel,feature_names[roi])
        
        feature=int(feature_names[roi][9:].replace('_right', ''))
        if args.importances==None:
            features_image[channel][labels.darrays[0].data==feature]=index
        else:
            features_image[channel][labels.darrays[0].data==feature]=importances[index]
    
    labels.darrays[0].data=features_image[0].astype('float32')
    for i in np.arange(1,args.num_channels):
        labels.add_gifti_data_array(nibabel.gifti.GiftiDataArray(features_image[i].astype('float32')))
        
    nibabel.save(labels,args.outname)

if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='RF training script')
    parser.add_argument('features', help='list of most important features')
    parser.add_argument('dataframe',help='path to dataframe')
    parser.add_argument('labels',help='path to label file')
    parser.add_argument('outname',help='output path')
    parser.add_argument('--importances', default=None, help='file with importance values, if true output image will colour ROI by importance')
    parser.add_argument('--num_channels', default=3, type=int,help='number of different cortical imaging channels (default 3  -thickness,myelin,folding)')
    parser.add_argument('--n_features', default=30,type=int)
 
     
    args = parser.parse_args()
    print(args.importances) 
    # Call training
    main(args)