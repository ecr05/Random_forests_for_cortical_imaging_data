# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:41:53 2017

@author: emmar
"""

import numpy as np
import pickle
import pandas as pd
import scipy.io as sio
import nibabel
import copy
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from sklearn import svm
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt



##### FUNCTIONS FOR PRE-PROCESSING (EXTRACTING DATA MATRIX FROM IMAGE DATA) UNTIL END ########### 

def average_multimodal_parcellation_features(featurefunc,labelfunc,labelrange):
    # average the surface results within each parcel and save in a data matrix
    features=np.zeros((len(labelrange)*featurefunc.numDA))
    average_func=np.zeros((labelfunc.darrays[0].data.shape[0],featurefunc.numDA))
    
    for index,val in enumerate(labelrange):
       # print(index,val,len(labelrange),featurefunc.numDA)
        x=np.where(labelfunc.darrays[0].data ==val)
        if x[0].shape != (0,):
            for i in np.arange(featurefunc.numDA):
                features[index*featurefunc.numDA+i]=np.mean(featurefunc.darrays[i].data[x[0]])
                average_func[[x],i]=np.mean(featurefunc.darrays[i].data[x[0]])
                if np.isnan(features[index*featurefunc.numDA+i]) :
                    print('isnan',index,val,x[0].shape,np.where(np.isnan(featurefunc.darrays[i].data[x[0]])))
        else:
            for i in np.arange(featurefunc.numDA):
                features[index*featurefunc.numDA+i]=0
                average_func[[x],i]=0
                
    return features,average_func
    
def average_multimodal_parcellation_features_all(files):
    dataset=pd.read_csv(files,sep=' ')
    #giftipaths=pd['features'] #np.genfromtxt(files,dtype=str);
   # labelpaths = pd['labels']

    trainingDATA=[]
    trainingfunc=[]

    for i in np.arange(len(dataset)):
        featuresLR=[]
        for hemi in ['L', 'R']:
            featurepath=dataset['features'][i].replace('%hemi%',hemi)
            labelpath=dataset['labels'][i].replace('%hemi%',hemi)
            print(dataset['features'][i],featurepath)

            funcdata=nibabel.load(featurepath)
            labeldata=nibabel.load(labelpath)
            #print(hemi, np.unique(labeldata.darrays[0].data))
            features,avfunc=average_multimodal_parcellation_features(funcdata,labeldata,np.arange(1,181))
            np.savetxt(featurepath.replace('func.gii','average.txt'),avfunc)
      
            if hemi=='L':
                featuresLR=features
            else:
                featuresLR2=np.concatenate((featuresLR,features))
        trainingDATA.append(featuresLR2)
    trainingDATA=np.asarray(trainingDATA)
    alldata={}
    alldata['subjid'] = dataset['subjid']
    alldata['features'] = trainingDATA

    #alldata = pd.DataFrame(trainingDATA)
    return alldata

######         
def map_feature_importances_back_to_image_space(flist,indices,labelfunc):
    
    mappings=np.zeros[(len(flist),labelfunc.shape[0])]
    
    for i in np.arange(len(flist)):
        regions=indices[flist[i]]
        x=np.where(labelfunc.darrays[0].data ==regions[0])
        y=np.where(labelfunc.darrays[0].data ==regions[1])
        mappings[i,x]=1;
        mappings[i,y]=2;
        
    return mappings
    
def map_feature_importances_back_to_image_space_HCP(numfeatures,numregions,importances,labelfuncL, labelfuncR):
    # map feature importances back to image domain from HCP multimodal features
    # fmask is the mask output from an initial feature selection step used to remove noisy features
    # importances is feature importances from final random forest 
    # labelfunc (one fore each hemisphere) is a gifti that contains label membership for regions across the surface
    

    mappingsL = np.zeros(labelfuncL.darrays[0].data.shape)
    mappingsR = np.zeros(labelfuncL.darrays[0].data.shape)
    importantfeatures=np.zeros((len(importances),2))
    for index,val in enumerate(importances):


        true_index=val
        region=int(true_index/numfeatures)+1 # add one because background is not included
        opt_feature=true_index -region*numfeatures
        importantfeatures[index,0]=region
        importantfeatures[index,1]=opt_feature
     #   print(index,true_index,region,opt_feature,len(importances))

        if region<=180:
            # then it is a left hemisphere feature     
            x=np.where(labelfuncR.darrays[0].data ==region)[0]
         #   print('R',x.shape)
            mappingsL[x]=index;
            #mappingsR.append(mappings)
            #labeltmpR.darrays[0].data=mappings[x]
            #labelRout.add_gifti_data_array(nibabel.gifti.GiftiDataArray(mappings[x]))
        else:
            x=np.where(labelfuncL.darrays[0].data ==region)[0]
      #      print('L',x.shape)

            mappingsR[x] = index;
            #print(np.unique(mappings),np.where(mappings>0)[0].shape)
            #mappingsL.append(mappings)

    #print(np.unique(np.asarray(mappingsL)), np.where(np.asarray(mappingsL) > 0)[0].shape)
    return mappingsL,mappingsR,importantfeatures
    
def return_kbest_features(features,labels,features_test,perc,method):
    # use statistical tests to remove noisiest features
     if method=='regression':
         print('kbest: regression',perc)
         kbest=SelectPercentile(score_func=f_regression, percentile=perc)
     else:
         #print('kbest: classification')
         kbest=SelectPercentile(percentile=perc)

     featuresperc = kbest.fit_transform(features, labels)
     return featuresperc,kbest.transform(features_test)
     

