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

# read all data into dataframe
for subjid, subj in enumerate(SUBJLIST):
    print(subj)
    for hemi in ['.L.', '.R.']:        
        label=nibabel.load(os.path.join(labeldir,subj + hemi + 'CorticalAreas_dil_Final_Individual.Colour.32k_fs_LR.label.gii'))
        features=nibabel.load(os.path.join(featdir,subj + hemi +'MultiModal_Features_MSMAll_2_d41_WRN_DeDrift.FULLVISUO.32k_fs_LR.func.gii'))
        
       