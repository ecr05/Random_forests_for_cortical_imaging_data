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

# choose to normalise data (or not)
normalise=True

# define data
group={'data':'VALIDATIONSET', 'subjlist':'VALIDATIONlist'}
topdir='/data/PROJECTS/HCP/HCP_PARCELLATION/'
dirname=os.path.join(topdir,group['data'])

# read subjlist
with open(os.path.join(dirname,group['subjlist'])) as f:
     SUBJLIST=f.readlines()
SUBJLIST = [x.strip() for x in SUBJLIST]

# read demographic data
demo=pd.read_pickle('/data/PROJECTS/HCP/HCP_PARCELLATION/DEMOGRAPHICDATA/unrestricted_emmar_1_10_2018_9_44_32.pk1')
demo_2=demo[['Subject', 'Gender', 'PMAT24_A_CR','PMAT24_A_SI','PMAT24_A_RTCR']].copy()
demo_2=demo_2.rename(columns={'Subject':'id'})

### define directory paths
labeldir=os.path.join(dirname,'classifiedlabels')
featdir=os.path.join(dirname,'featuresets')

###### get directory dir
labeldictL=nibabel.load(os.path.join(labeldir,SUBJLIST[0] +  '.L.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')).labeltable.get_labels_as_dict()
labeldictR=nibabel.load(os.path.join(labeldir,SUBJLIST[0] +  '.R.CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii')).labeltable.get_labels_as_dict()

df = pd.DataFrame() 

# read all data into dataframe
for subjid, subj in enumerate(SUBJLIST):
    print(subj)
    roi_data={}
    roi_data['id']=subj
    tmp=0 
    for hemi in ['.L.', '.R.']:        
        label=nibabel.load(os.path.join(labeldir,subj + hemi + 'CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii'))
        features=nibabel.load(os.path.join(featdir,subj + hemi +'MultiModal_Features_MSMAll_2_d41_WRN_DeDrift.FULLVISUO.32k_fs_LR.func.gii'))
        if hemi=='.L.':
            labeldict=labeldictL
            
        else:
            labeldict=labeldictR
          
        for key in range(1,181):
            
            inds=np.where(label.darrays[0].data==key)[0]
            #print(key,hemi,subj,labeldict[key],inds.shape)

            for DA in range(features.numDA):
                if normalise==True:
                    
                    feat_array=(features.darrays[DA].data-np.mean(features.darrays[DA].data))/np.std(features.darrays[DA].data)
                else:
                    feat_array=features.darrays[DA].data
                    
                tmp+=1
                roi_label=labeldict[key] + '_' + str(DA)
                if len(inds)>0:    
                    if(feat_array[inds].shape[0]==0):
                        print('shape 0')
                        
                    roi_data[roi_label]=np.mean(feat_array[inds])   
                 
                else:
                    roi_data[roi_label]=0
                    
                features.darrays[DA].data[inds]=roi_data[roi_label]
                    #print(roi_label,roi_data[roi_label] )
        if subjid==0:
            nibabel.save(features,os.path.join(featdir,subj + hemi +'MultiModal_Features_MSMAll_2_d41_WRN_DeDrift.FULLVISUO.MEAN.32k_fs_LR.func.gii'))            

    df = df.append(roi_data, ignore_index=True)

# merging roi data with behavioural labels    
df['id']=df['id'].astype('int64')
df2=df.merge(demo_2)

# converting gender column to boolean categories
df2['Gender']=df2['Gender'].astype('category')
cat_columns = df2.select_dtypes(['category']).columns
df2[cat_columns]=df2[cat_columns].apply(lambda x: x.cat.codes)

# save
if normalise:
    df2.to_pickle(os.path.join(dirname,group['data']+'roimat_norm.pk1')) 
else:
    df2.to_pickle(os.path.join(dirname,group['data']+'roimat.pk1'))
    
                
                
                
            
       