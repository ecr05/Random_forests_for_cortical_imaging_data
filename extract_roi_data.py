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

image_fields=['thickness.shape.gii', 'myelin_map.func.gii', 'curvature.shape.gii',]
out_type='thickness_myelin_curv.func.gii'
dirname='/data/PROJECTS/dHCP/PROCESSED_DATA/reconstructions_june2018/fsaverage_32k_data/separate_files/'
outdir='/data/PROJECTS/dHCP/PROCESSED_DATA/reconstructions_june2018/DL_DATASETS/'
participants=pd.read_csv(os.path.join(dirname,'participants.tsv'),sep='\t')

labeldict=[]
labelL=nibabel.load('/data/PROJECTS/dHCP/PROCESSED_DATA/TEMPLATES/new_surface_template/labels/40.L.label.gii')
labeldict.append(labelL.labeltable.get_labels_as_dict())

labelR=nibabel.load('/data/PROJECTS/dHCP/PROCESSED_DATA/TEMPLATES/new_surface_template/labels/40.R.label.gii')
labeldict.append(labelR.labeltable.get_labels_as_dict())

DATAMAT = {}
df = pd.DataFrame(columns=['id','session','scan_ga','birth_ga']) 
count=0
for index_p, row_p in participants.iterrows():
    print(index_p,row_p['participant_id'])
    subj_tdir=os.path.join(outdir,'sub-' + row_p['participant_id'] )# get id
   
         
    sessions=dirname + '/sub-' + row_p['participant_id'] +'/sub-'+ row_p['participant_id'] +'_sessions.tsv'
    sessions_df=pd.read_csv(sessions,sep='\t')
    
    sessions_df=sessions_df.drop_duplicates(subset='session_id')
    for index, row in sessions_df.iterrows():
        #print(row_p['participant_id'],row['session_id'],row['age_at_scan'])
        subjdir='/projects/dhcp-pipeline-data/icl/derivatives/sub-' + row_p['participant_id']+'/ses-'+  str(row['session_id'])+'/anat/Native/'
        subj_odir=os.path.join(subj_tdir,'ses-'+ str(row['session_id']) +'/anat/Native')
        if  os.path.isfile(os.path.join(subjdir,'sub-'+row_p['participant_id']+'_ses-'+ str(row['session_id'])+'_left_myelin_map.func.gii')) and \
        os.path.isfile(os.path.join(subjdir,'sub-'+row_p['participant_id']+'_ses-'+ str(row['session_id'])+'_left_thickness.shape.gii')) and \
        os.path.isfile(os.path.join(subjdir,'sub-'+row_p['participant_id']+'_ses-'+ str(row['session_id'])+'_right_myelin_map.func.gii')) and \
        os.path.isfile(os.path.join(subjdir,'sub-'+row_p['participant_id']+'_ses-'+ str(row['session_id'])+'_right_thickness.shape.gii')):
            features={};
            features['id']= row_p['participant_id']
            features['session']= row['session_id']
            features['scan_ga']= row['age_at_scan']
            features['birth_ga']= row_p['birth_ga']
 
            count+=1
            for h_ind, hemi in enumerate([ 'left', 'right']):
                alldata=nibabel.gifti.gifti.GiftiImage(header=None)
                label=nibabel.load(os.path.join(subjdir,'sub-'+row_p['participant_id']+'_ses-'+ str(row['session_id'])+'_' + hemi + '_drawem.label.gii'))
                
                    
               
                
                #print(k,h_ind,hemi,inds.shape)
                for img_type in image_fields:
                    data=nibabel.load(os.path.join(subjdir,'sub-'+row_p['participant_id']+'_ses-'+ str(row['session_id'])+'_' + hemi + '_'+ img_type))
                    for k in labeldict[h_ind]:
                        inds=np.where(label.darrays[0].data==k)[0]
                        labelname=img_type.split('.')[0] + '_roi_' + str(k) + '_' + hemi
                        
                        features[labelname]=np.mean(abs(data.darrays[0].data[inds]))
                        #print(labelname,features[labelname])
                      
            #DATAMAT[row['session_id']]=features
            df = df.append(features, ignore_index=True)
            #print(count,np.asarray(features).shape,df.shape,len(DATAMAT))  

df.to_pickle(os.path.join(outdir,'roi_datamat_abscurv_wlabels.pickle'))


# use this data to filter by cohort/group

for group in ['TRAIN_prem_vs_term/TRAIN_prem_vs_term.pk1','TEST_prem_vs_term/TEST_prem_vs_term.pk1', \
              'TRAIN_ga_regression/TRAIN_ga_regression.pk1','TEST_ga_regression/TEST_ga_regression.pk1']:
    

    groupdf=pd.read_pickle(os.path.join(outdir,group))
    groupdf=groupdf.rename(index=str, columns={"age_at_scan": "scan_ga", "age_at_birth": "birth_ga"})
    groupdf_plusrois=groupdf.merge(df, on=['id','session',"scan_ga","birth_ga"])
    groupdf_plusrois.to_pickle(os.path.join(outdir,group.replace('.pk1','wrois.pk1')))

