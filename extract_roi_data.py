#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:21:41 2018
Script to summarise multi-modality dHCP data as mean values for a series of cortical ROIS
@author: er17
"""

import nibabel
import os
import pandas as pd
import numpy as np
import argparse


def extract_rois(args):
    
    participants=pd.read_pickle(args.data_csv)
    # use templatelabels to define label dictionary
    labeldict=[]
    labelL=nibabel.load(args.templatelabel.replace('%hemi%','L'))
    labeldict.append(labelL.labeltable.get_labels_as_dict())
    
    labelR=nibabel.load(args.templatelabel.replace('%hemi%','R'))
    labeldict.append(labelR.labeltable.get_labels_as_dict())

    
    df = pd.DataFrame() 
    for index, row in participants.iterrows():
        print(row['id'])
        
        for h_ind, hemi in enumerate([ 'left', 'right']): # need to standardise naming of HCP and dHCP files!
        #for h_ind, hemi in enumerate([ 'L', 'R']):
            if args.usegrouplabels:
                if hemi=='L':
                    label=labelL
                else:
                    label=labelR
            else:
                label=nibabel.load(row['label_file'].replace('%hemi%',hemi))
            func=nibabel.load(row['data_path'].replace('%hemi%',hemi))
           # print(hemi,'labels',np.unique(label.darrays[0].data),labeldict[h_ind])
            #newlabel=copy.deepcopy(label)
            data=nibabel.load(row['data_path'].replace('%hemi%',hemi))
            #### for debugging####
# =============================================================================
#             for j in np.arange(data.numDA):
#                 func.darrays[0].data=np.zeros((func.darrays[0].data.shape)).astype('float32')
# =============================================================================
            ########################  
            features={};
            features[args.idfield]= row[args.idfield]
            for k in labeldict[h_ind]:
                #print(k,row['id'])
                if np.where(label.darrays[0].data==k)[0].shape[0]>0:
                    inds=np.where(label.darrays[0].data==k)[0]
                    for j in np.arange(data.numDA):
                                
                        labelname='DA_' + str(j) + '_roi_' + str(k) + '_' + hemi
                        
                        features[labelname]=np.mean(data.darrays[j].data[inds])#np.mean(abs(data.darrays[j].data[inds]))
                        func.darrays[j].data[inds]=np.mean(data.darrays[j].data[inds])#np.mean(abs(data.darrays[j].data[inds]))
# =============================================================================
#                         if j < 5:
#                             print(j,k,labeldict[h_ind][k],data.numDA,np.mean(data.darrays[j].data[inds]),data.darrays[j].data[inds].shape,len(inds),func.darrays[j].data[inds[0]])
#             
# =============================================================================
            #print('save',os.path.join(args.indir,'sub-'+str(row['id'])+'_' + hemi + '_ROIS.func.gii'))
#        nibabel.save(func,os.path.join(args.outdir,'sub-'+str(row['id'])+'_' + hemi + '_ROIS.func.gii'))
        df = df.append(features, ignore_index=True)
                #print(count,np.asarray(features).shape,df.shape,len(DATAMAT))  
    
    df.to_pickle(args.oname)


if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Example: summarise image data as mean values per ROI')
    parser.add_argument('data_csv',  help='csv file with meta data and "data_path" field (incl wildcard %hemi% for multihemisphere data)') 
    parser.add_argument('oname',  help='output name for dataframe')
    parser.add_argument('templatelabel',  help='template label file (incl wildcard %hemi% for multihemisphere data)')
    parser.add_argument('--outdir',  help='output directory')
    parser.add_argument('--hemis',  help='list of hemis e.g. ["L","R"]')
    parser.add_argument('--usegrouplabels',  action='store_true')
    parser.add_argument('--idfield',  help='field which uniquely identifies data',default='id')

    args = parser.parse_args()
    extract_rois(args)
    
