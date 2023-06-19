#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:08:34 2020

@author: wxr
"""

import os
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth,Birch,MiniBatchKMeans
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import umap
from sklearn.decomposition import PCA, NMF,TruncatedSVD,FastICA
from sklearn import manifold 
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score,silhouette_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from skopt import BayesSearchCV
from sklearn import random_projection
import csv
from scipy.stats import norm
import Tools 

import time
import sys
import shutil
from sklearn.base import BaseEstimator
from communityc2st import getAccuracyByCommunity

from Utils import plot_roc_curve,save_accuracy_data,plot_confusion_matrix
from numpy import random,mat
import torch
import random

seed = 0
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)   
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) 

import warnings
warnings.filterwarnings("ignore")



def init_parameter():

    '''''
    -----------------------------parameters------------------------------
    '''''
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='Clustering',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='Community', choices=['SNP'])
    parser.add_argument('--dataset_hospital', default=-1,choices=[-1,20])
    # 0 HC   1 MDD     2 SZ     3 BD
    parser.add_argument('--types', default=[0,1])  
    parser.add_argument('--hc_type', default=0)
    parser.add_argument('--classification_type', default='svm',choices=['Twoaccessmodel', 'resnet','CNN3DNet'])
    
    parser.add_argument('--cluster_type', default='Kmeans', choices=['GMM','Spe', 'Kmeans', 'Agg','HDBSCAN'])  #Cluster
    parser.add_argument('--dr_type', default='UMAP-y', choices=['UMAP','PCA', 't-SNE', 'NMF'])   #dimension reduction
    parser.add_argument('--dr_dim', default=3)
    parser.add_argument('--save_path', default='./results')    
    parser.add_argument('--seed',default=seed)
    parser.add_argument('--threshold',default=0.7)
    args = parser.parse_args() 
    
    file_name = Path(__file__).name[:-1]
    args.save_path = args.save_path+'/'+file_name
    
    return args
            
class CCD(BaseEstimator):
    def __init__(self,dr_type='UMAP',dr_dim=2,cluster_type='GMM',k=2,overwrite=False,threshold=0.7):
        self.dr_type = dr_type
        self.dr_dim = dr_dim
        self.cluster_type = cluster_type
        self.k = k
        self.overwrite = overwrite
        self.threshold = threshold
    def cluster(self,type_,classes,x):
        if type_ == 'Spe':
            model = SpectralClustering(n_clusters=classes,random_state=0)
        elif type_ == 'Kmeans':
            model = KMeans(n_clusters=classes,random_state=0)
        elif type_ == 'Agg':
            model = AgglomerativeClustering(n_clusters=classes)
        elif type_ == 'GMM':
            model = GaussianMixture(n_components=classes,random_state=0)
        elif type_ == 'MiniBatchKMeans':
            model = MiniBatchKMeans(n_clusters=classes,random_state=0)
            
        results = model.fit_predict(x)
        return results

    def dimention_reduction(self,type_,dr_dim,x,y=None):
        if type_ == 'PCA':
            model = PCA(n_components=dr_dim,random_state=0)
        elif type_ == 'Isomap':
            model = manifold.Isomap( n_components=dr_dim, n_neighbors=20)
        elif type_ == 't-SNE':
            model = manifold.TSNE(n_components=dr_dim, init='pca',random_state=0)
        elif type_ == 'NMF':
            model = NMF(n_components=dr_dim, init='random',random_state=0)
        elif type_ == 'UMAP' or type_ == 'UMAP-Y':
            model = umap.UMAP(n_components=dr_dim,random_state=0)
        elif type_ == 'RandomProj':
            model = random_projection.SparseRandomProjection(n_components=dr_dim,random_state=0)
        elif type_ == 'SVD':
            model = TruncatedSVD(n_components=dr_dim,random_state=0)
        elif type_ == 'FastICA':
            model = FastICA(n_components=dr_dim,random_state=0)
        
        if type_ == 'UMAP-Y':
            x_dr = model.fit_transform(x,y)
        else:
            x_dr = model.fit_transform(x)
        return x_dr
        
    def save_clustering_metric(self,x,y_cluster):
        CH_DB_SC_pd = pd.DataFrame(columns=['CH','DB','SC'])
    
        CH = calinski_harabasz_score(x,y_cluster)
        DB = davies_bouldin_score(x,y_cluster)
        SC = silhouette_score(x,y_cluster)
        CH_DB_SC_pd.loc[self.k] = {'CH':CH,'DB':DB,'SC':SC}
        
        CH_DB_SC_pd.to_excel(self.save_path+'/%d/CH_DB_SC.xlsx'%self.k)
         
    def init_parameters(self):
        args = init_parameter()
        self.classification_type = args.classification_type
        self.save_path = args.save_path + '/%s_%s_%d_%s_%s_%f'%(args.dataset,self.dr_type,self.dr_dim,self.cluster_type,
                                                                     args.classification_type,self.threshold)
        
        self.dataset = args.dataset
        self.hc_type = args.hc_type
        self.types = args.types    
        Tools.check_and_create_directory(self.save_path)
        
        args.cluster_type = self.cluster_type
        args.dr_type = self.dr_type
        args.dr_dim = self.dr_dim
        args.threshold = self.threshold
        args.save_path = self.save_path

        
        f1 = open(os.path.join(self.save_path,'parameter.txt'),'w')
        f1.write(str(vars(args)))
        f1.close()
        print(args)
        return args
            
    def fit(self,x):
        self.failed = False
        self.init_parameters()
        
        if Tools.check_has_it_alread_ran(self.save_path,self.k) and self.overwrite == False:
            print('skip this run !!!')
            return
        # index_hc = y==self.hc_type
        x_p = x
        x_p_dr = self.dimention_reduction(self.dr_type, self.dr_dim, x_p)
        self.y_cluster = self.cluster(self.cluster_type,self.k,x_p_dr)
        
        y_cluster_dict = dict(Counter(self.y_cluster))
        for key,value in y_cluster_dict.items():
            if value < 30 or len(y_cluster_dict.items()) != self.k:
                self.failed=True
                print('less than 30 abandon!!')
                # shutil.rmtree(self.save_path)
                return
            
        save_path_clustering = self.save_path+'/%d/results'%self.k
        Tools.check_and_create_directory(save_path_clustering)
        print('Clustering result:',Counter(self.y_cluster))
        clustering_label_path = save_path_clustering+'/label_false_0.npy'
        np.save(save_path_clustering+'/feature_dr_0',x_p_dr)
        np.save(save_path_clustering+'/label_false_0',self.y_cluster)
        
        self.save_clustering_metric(x_p_dr,self.y_cluster)
        
        if x_p_dr.shape[1] > 2:
            x_p_dr = self.dimention_reduction('PCA',2,x_p_dr)
        Tools.plot_data(x_p_dr,self.y_cluster,save_path_clustering)



  
    def score(self,x=None,y=None):
        if self.failed:
            return 0

        y_cluster = np.load(self.save_path+'/%d/results/label_false_0.npy'%self.k)
        #save_path:
        subfile_name = '%s_%d_%s_%d_%f'%(self.dr_type,self.dr_dim,self.cluster_type,self.k,self.threshold)
        bloc_subfile_name = '%s_%d_%s_%d'%(self.dr_type,self.dr_dim,self.cluster_type,self.k)

        with open('./results/score_sum_svm_OR_addThresh.txt','r') as openfile:
            for line in openfile:
                parts = line.strip().split('\t')
                if subfile_name==parts[0]:
                    print('-'*20,'subfile_name==parts[0]---',float(parts[1]))
                    return float(parts[1])

        getAccuracyByCommunity(y_cluster,self.threshold,self.save_path+'/%d/results/'%self.k,bloc_subfile_name)
        accuracypath = self.save_path+'/%d/results/res/some_accuracy_data/'%self.k
        acc_dir_name_list = os.listdir(accuracypath)
        
        orpath = 'bloc/%s/'%(bloc_subfile_name)

        max_subtype_or = 0.0 
        for cluster in range(self.k):
            #svm
            dataor = pd.DataFrame(pd.read_table(orpath+'%d/%d_cluster_result_%f.xlsx'%(cluster,cluster,self.threshold),header=None,encoding="utf-8",sep='\t'))
            dataor.columns = ['commuity_id','node_num','case_num','sum-case_num','case_sum','case_fre','control_num','sum-control_num','control_sum','control_fre','OR','CI','fre_ratio']
            dataornew1 = dataor[(dataor['OR'] >=1.5) & (dataor['CI'].str.contains('\\[0|\\[-nan,',regex = True) == False)]
            data_or_sort = dataornew1.sort_values(by="OR" , ascending=False).head(10)
            data_arr = pd.DataFrame(pd.read_excel(accuracypath+acc_dir_name_list[cluster]))

            data_or_acc = pd.merge(data_or_sort, data_arr, left_index=True, right_index = True,how='left')

            subtypeMaxOr = float(data_or_acc['OR'].sum())
            data_or_acc.to_excel(orpath+'%d/%d_result_top_%f.xlsx'%(cluster,cluster,self.threshold))
           
            lable_false = y_cluster.tolist()
            num = lable_false.count(cluster)

            max_subtype_or = max_subtype_or+subtypeMaxOr*(float(num)/len(lable_false))


        with open('./results/score_sum_svm_OR_addThresh.txt','a+') as openfile:
            openfile.writelines(subfile_name+'\t'+str(max_subtype_or)+'\n')
        return max_subtype_or     
    

def search_all_parameters(save_path):
    opt = BayesSearchCV(
        CCD(),
        {
            'dr_type':['PCA','Isomap','UMAP','UMAP-Y','RandomProj','SVD','FastICA'],
            'dr_dim':(2,30),
            'cluster_type':['Spe','Kmeans','Agg','GMM','MiniBatchKMeans'],
            'k':(2,3),
            'threshold':[0.7]
        },
        n_iter = 300,
        cv=[(slice(None), slice(None))],
        verbose=1,
        random_state=1
    )
    
    opt.fit(x)
    
    np.save(save_path,opt.cv_results_)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(x))
    
if __name__ == "__main__":
    data1 = np.load('genetype_sz_QC_case_0.68.npy')
    x = data1.reshape((data1.shape[0],-1))
    
    save_path = './result1111/'
    search_all_parameters(save_path)
    
    model = CCD(dr_type='PCA',dr_dim=12,cluster_type='MiniBatchKMeans',k=3,overwrite=True)
    model.fit(x)
    score = model.score()
    print(score)