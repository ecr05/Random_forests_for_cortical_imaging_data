#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 17:55:49 2018

@author: er17
"""

import nibabel
import os
import pandas as pd
import numpy as np

# read subjlist
with open('/data/PROJECTS/HCP/HCP_PARCELLATION/TRAININGDATA/TRAININGlist') as f:
     SUBJLIST=f.readlines()
SUBJLIST = [x.strip() for x in SUBJLIST]

### define directory paths
labeldir='/data/PROJECTS/HCP/HCP_PARCELLATION/TRAININGDATA/classifiedlabels/'
featdir='/data/PROJECTS/HCP/HCP_PARCELLATION/TRAININGDATA/featuresets/'

###### get directory dir
labeldictL=nibabel.load(os.path.join(labeldir,SUBJLIST[0] +  '.L.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')).labeltable.get_labels_as_dict()
labeldictR=nibabel.load(os.path.join(labeldir,SUBJLIST[0] +  '.R.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')).labeltable.get_labels_as_dict()

df = pd.DataFrame() 

# read all data into dataframe
for subjid, subj in enumerate(SUBJLIST):
    print(subj)
    roi_data={}
    roi_data['id']=subj
    
    for hemi in ['.L.', '.R.']:        
        label=nibabel.load(os.path.join(labeldir,subj + hemi + 'CorticalAreas_dil_Final_Individual.32k_fs_LR.label.gii'))
        features=nibabel.load(os.path.join(featdir,subj + hemi +'MultiModal_Features_MSMAll_2_d41_WRN_DeDrift.FULLVISUO.32k_fs_LR.func.gii'))
        if hemi=='L':
            labeldict=labeldictL
            
        else:
            labeldict=labeldictR
            
        for key in labeldict:
            inds=np.where(label.darrays[0].data==key)[0]
            
            for DA in range(features.numDA):
                roi_label=labeldict[key] + '_' + str(DA)
                if len(inds)>0:    
                    if(features.darrays[DA].data[inds].shape[0]==0):
                        print('shape 0')
                        
                    roi_data[roi_label]=np.mean(features.darrays[DA].data[inds])               
                else:
                    roi_data[roi_label]=0
                    
                features.darrays[DA].data[inds]=roi_data[roi_label]
                #print(roi_label,roi_data[roi_label] )
        if subjid==0:
            nibabel.save(features,os.path.join(featdir,subj + hemi +'MultiModal_Features_MSMAll_2_d41_WRN_DeDrift.FULLVISUO.MEAN.32k_fs_LR.func.gii'))            
    
    df = df.append(roi_data, ignore_index=True)
    
                
                
                
            
       