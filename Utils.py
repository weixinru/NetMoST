#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 09:12:13 2020

@author: wxr
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from scipy import interp
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
import seaborn as sns
import torch.nn.functional as F
import torch
import os
def plot_roc_curve(y_true_list,y_pred_list,k,save_path=None,file_name=None,image_flag=True):
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if y_pred_list[0].ndim != 1:
        argmax = True
    else:
        argmax = False
    if k > 2:
        mean_auc, std_auc = plot_roc_curve_multiclasses(y_true_list,y_pred_list,k,image_flag,save_path,file_name,argmax)
    else:
        mean_auc, std_auc = plot_roc_curve_2lcasses(y_true_list,y_pred_list,image_flag,save_path,file_name,argmax)
        
    return mean_auc, std_auc

def plot_roc_curve_2lcasses(y_true_list,y_pred_list,image_flag,save_path=None,file_name = None, argmax=True):
    k = len(y_true_list)
    aucs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    if image_flag:
        plt.figure(figsize=(8,8))
    
    classes = []
    for i in range(2):
        classes.append(i)
    for i in range(k):
        y_true = label_binarize(y_true_list[i], classes=classes)
        if argmax:
            y_pred = F.softmax(torch.tensor(y_pred_list[i],dtype=torch.float32),dim=1)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred[:, 1])
        else:
            y_pred = y_pred_list[i]
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        if image_flag:
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
    
    if image_flag:
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    if image_flag == False:
        return mean_auc, std_auc
    
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title('ROC',fontsize=20)
    plt.legend(loc="lower right")

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        plt.savefig(save_path+file_name,dpi=400)
    plt.show()
    return mean_auc, std_auc

def plot_roc_curve_multiclasses(y_true_list,y_pred_list,n_classes,image_flag,save_path=None,argmax=True):
    
    kfold = len(y_true_list)
    
    tprs = []    
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    if image_flag:
        plt.figure(figsize=(8,8))
    
    for k in range(kfold):
        
        y_true = y_true_list[k] #n*1
        y_pred = y_pred_list[k] #n*calsses
        
        classes = []
        for i in range(n_classes):
            classes.append(i)

        y_true = label_binarize(y_true, classes=classes)
        y_pred = F.softmax(torch.tensor(y_pred,dtype=torch.float32),dim=1)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_pred[:,i])
        
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        tprs.append(interp(mean_fpr, fpr["macro"], tpr["macro"]))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc["macro"])
        
        if image_flag:
#         label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc)
            plt.plot(fpr["macro"], tpr["macro"], lw=1, alpha=0.3,
                     label='macro-average ROC fold '+str(k+1)+' (AUC = {0:0.2f})'.format(roc_auc["macro"]))
    
    if image_flag:
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    if image_flag == False:
        return mean_auc, std_auc
    
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title('ROC',fontsize=20)
    plt.legend(loc="lower right")

    if save_path is not None:
        data = pd.DataFrame(columns=['mean_auc','std_auc'])
        data.loc[0] = {'mean_auc':mean_auc,'std_auc':std_auc}
        data.to_excel(save_path+'/roc.xlsx') 
        plt.savefig(save_path+'/roc_curve',dpi=400)
    plt.show()
    return mean_auc, std_auc
    
def save_accuracy_data(y_true_list,y_pred_list,type,data):
    if y_pred_list[0].ndim != 1:
        argmax = True
    else:
        argmax = False
    k = len(y_true_list)
    a_all = 0
    f1_all = 0
    p_all = 0
    r_all= 0
    for i in range(k):
        y_true = y_true_list[i]
        y_pred = y_pred_list[i]
        if argmax:
            y_pred = np.argmax(y_pred_list[i],axis=1)
        a = accuracy_score(y_true,y_pred)
        f1 = f1_score( y_true, y_pred, average='macro' )
        p = precision_score(y_true, y_pred, average='macro')
        r = recall_score(y_true, y_pred, average='macro')
        a_all += a
        f1_all += f1
        p_all += p
        r_all += r
    data.loc[type] = {'accuracy':a_all/k,'F1':f1_all/k,'precession':p_all/k,'recall':r_all/k}
    
    
def plot_confusion_matrix(y_true_list,y_pred_list,classes,save_path=None,file_name=None,labels=None,names=None):
    if y_pred_list[0].ndim != 1:
        argmax = True
    else:
        argmax = False
    line_n = 3 #
    font_size = 15
    n = len(y_true_list)
    line = int((n-1)/line_n)+1
#    left = n - line*line_n
    
    if labels == None:
        labels = []
        for i in range(classes):
            labels.append(i)
    if names == None:
        names = []
        for i in range(classes):
            names.append(str(i))
    
    f,ax=plt.subplots(line,line_n,figsize=(18,5*line))
    for i in range(n):    
        l = int(i/line_n)
        m = i-l*line_n
    
        y_true = y_true_list[i]
        if argmax == False:
            y_pred = y_pred_list[i]
        else:
            y_pred = np.argmax(y_pred_list[i],axis=1)
        if line == 1:
            ax_item = ax[m]
        else:
            ax_item = ax[l,m]
        sns.heatmap(confusion_matrix(y_true,y_pred,labels=labels),ax=ax_item,annot=True,
                    annot_kws={'size':font_size},fmt='d',cmap="YlGnBu",
                    xticklabels=names,
                    yticklabels=names)
        
        ax_item.set_title('Fold'+str(i+1),fontsize=font_size)
        bottom, top = ax_item.get_ylim()
        # ax_item.ylabel('True label')
        # ax_item.xlabel('Predicted label')
        ax_item.set_ylim(bottom, top)
        ax_item.tick_params(labelsize=font_size)
        ax_item.set_yticklabels(ax_item.get_yticklabels(), rotation=0)

    for m in range(line_n):
        if line_n*(line-1)+m >= n:
            if line == 1:
                ax[m].remove()
            else:
                ax[line-1,m].remove()
            
    plt.subplots_adjust(hspace=0.2,wspace=0.2)
    plt.tight_layout()

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+file_name,dpi=400)
    plt.show()