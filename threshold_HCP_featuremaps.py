#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 12:53:13 2018

@author: Emma C Robinson

Function for thresholding HCP featuremaps to remove noise from fMRI derived features

"""
import pandas as pd
from scipy import stats

# load training and test data
training=pd.read_pickle('/data/PROJECTS/HCP/HCP_PARCELLATION/TRAININGDATA/TRAININGDATAroimat_withval_norm.pk1')
testing=pd.read_pickle('/data/PROJECTS/HCP/HCP_PARCELLATION/TESTINGDATA/TESTINGDATAroimat_norm.pk1')

ALLDATA=pd.concat((training,testing))
ALLDATA=ALLDATA.drop(['id', 'Gender', 'PMAT24_A_CR','PMAT24_A_SI','PMAT24_A_RTCR'],axis=1)

# remove all columns for which the distribution across the data set is not significant from 0
keep=['id', 'Gender', 'PMAT24_A_CR','PMAT24_A_SI','PMAT24_A_RTCR']
for column in ALLDATA:
     t,p=stats.ttest_1samp(ALLDATA[column].as_matrix(),0)
     if p < 0.01:
         print(column,t,p)
         keep.append(column)
         
filtered_train=training[keep].copy()
filtered_test=testing[keep].copy()

filtered_train.to_pickle('/data/PROJECTS/HCP/HCP_PARCELLATION/TRAININGDATA/TRAININGDATAroimat_withval_norm_filtered.pk1')
filtered_test.to_pickle('/data/PROJECTS/HCP/HCP_PARCELLATION/TESTINGDATA/TESTINGDATAroimat_norm_filtered.pk1')