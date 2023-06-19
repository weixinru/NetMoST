#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:02:27 2020

@author: wxr
"""
import os
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt 
def check_and_create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_has_it_alread_ran(save_path,k):
    file_path = save_path+'/%d/results/label_false_0.npy'%k
    if not os.path.exists(file_path):
        return False
        
    
    return True
        
def plot_data(x,y,save_path='',n=0,hospital_id=False): 
    # import random
    numSamples, dim = x.shape  
    mark = ['or', 'ob', 'og', 'ok', 'yo','oc', 'om', 'oy', 'ow', '+r', 'sr', 'dr', '<r', 'pr','+b', 'sb', 'db', '<b', 'pb','+g', 'sg', 'dg', '<g', 'pg','+k', 'sk', 'dk', '<k', 'pm','+c', 'sc', 'dc', '<c', 'pc']  
    # random.shuffle(mark)
    temp = Counter(y)
    key_pair = {}
    j = 0
    for (k,v) in temp.most_common():
        key_pair[k] = j
        j+=1
    
    index = np.argsort(y)
    y = y[index]
    x = x[index]

    plt.figure(figsize=(5,5),dpi=100)
    label_list = []
    pre_text = 'subtype %d'
    if hospital_id:
        pre_text = 'S%d'
    for i in range(numSamples):  
        markIndex = int(key_pair[y[i]])  
        if not int(y[i]) in label_list:
            label_list.append(int(y[i]))
            plt.plot(x[i, 0], x[i, 1], mark[markIndex],markersize=1,label = pre_text%int(y[i]))  
        else:
            plt.plot(x[i, 0], x[i, 1], mark[markIndex],markersize=1)  
    plt.legend(loc='upper right', fontsize=8)
    if save_path != '':
        if hospital_id:
            plt.savefig(save_path+'/cluste'+str(n)+"_hospital_id.png",dpi=400)
        else:
            plt.savefig(save_path+'/cluste'+str(n)+".png",dpi=400)
    plt.show()