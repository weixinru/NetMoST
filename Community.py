#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
import os
import keras
import numpy as np
import random
import os
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA 
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.models import Model
from keras.models import load_model,model_from_json
from keras import backend as K
import shutil
from scipy import stats
from sklearn import svm
from scipy.stats import norm
import csv
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score,silhouette_score
import time
from sklearn.model_selection import permutation_test_score, cross_val_score, cross_validate,KFold,StratifiedKFold
from Utils import plot_roc_curve,save_accuracy_data,plot_confusion_matrix
import community
import networkx as nx
import sys
import multiprocessing
import shutil
import time
import gc

def freeMemory(Vari):
    del(Vari)
    gc.collect()

def getCommunitySnp(Communityfile):
    Communitydict = {}
    with open(Communityfile,'r') as datafile:
        lines = datafile.read().strip().split("\n")
        for positionid in range(len(lines)):
            list = lines[positionid].strip().split('\t')
            list_int = [eval(i) for i in list ]
            Communitydict[positionid] = list_int
    return Communitydict 

def getPersonCommunity(Communitydict,PersonSnpfile):
    with open(PersonSnpfile) as inputfile:
        ComPerdict = {}
        for line in inputfile:
            parts = line.strip().split('\t')
            for Communityid in range(len(Communitydict)):
                if Communityid in ComPerdict:
                    list_singlesnp = []
                    for col in Communitydict[Communityid]:
                        if (col > 404078):
                            str2 = parts[col - 404078].strip().split('/')[1]
                            list_singlesnp.append(str2)
                        else:
                            str1 = parts[col].strip().split('/')[0]
                            list_singlesnp.append(str1)
                    ComPerdict[Communityid] = np.concatenate((ComPerdict[Communityid],list_singlesnp),axis = 0)
                else:
                    list_singlesnp = []
                    for col in Communitydict[Communityid]:
                        if(col > 404078):
                            str2 = parts[col-404078].strip().split('/')[1]
                            list_singlesnp.append(str2)
                        else:
                            str1 = parts[col].strip().split('/')[0]
                            list_singlesnp.append(str1)
                    ComPerdict[Communityid] = list_singlesnp
    return ComPerdict
def getOneHotSNP(dict):
    one_hot = {
        'A':[0,0,0,1],
        'T':[0,0,1,0],
        '0':[0,0,0,0],
        'C':[0,1,0,0],
        'G':[1,0,0,0]
    }
    onehotdic = {}
    for communityid in range(len(dict)):
        if communityid in onehotdic:
            tmp = []
            for i in range(len(dict[communityid])):
                tmp.extend(one_hot[dict[communityid][i]])
            #print(tmp)
            onehotdic[communityid] = np.concatenate((onehotdic[communityid],tmp),axis=0)
        else:
            tmp = []
            for i in range(len(dict[communityid])):
                tmp.extend(one_hot[dict[communityid][i]])
            #print(tmp)
            onehotdic[communityid] = tmp
    return onehotdic
    
def getone_hot_community_control(one_hotdict):
    for i in range(len(one_hotdict)):
        one_hotdict[i] = np.array(one_hotdict[i]).reshape(283, -1)
    return one_hotdict
def getone_hot_community_case(one_hotdict,key_num):
    for i in range(len(one_hotdict)):
        one_hotdict[i] = np.array(one_hotdict[i]).reshape(key_num, -1)
    return one_hotdict


def getCommunity(netfile,communityfile):
    initnode = [-1 for i in range(808156)]
    # Replace this with your networkx graph loading depending on your format !
    G = nx.read_weighted_edgelist(netfile, delimiter='\t', nodetype=str,encoding='utf-8')
    partition = community.best_partition(G)
    communitylist = set(partition.values())
    cluster_num = len(communitylist)

    for nodeid,communityid in partition.items():
        initnode[int(nodeid)-1] = int(communityid)
    line = " ".join(str(i) for i in initnode)
    with open(communityfile, 'w')as f:
        f.writelines("808156 nodes " + str(cluster_num)+" clusters 4444444 edges\n")
        f.writelines(line)
    f.close()

def neural_test(p, q, count, num_train, epochs, batch_size,test_size):

    K.clear_session()
    scores = []
    p_values = []
    if (count % 5 == 0):
        print("======{}======".format(str(count)))

    for i in range(num_train):
        x_train, x_test, y_train, y_test = train_test_split(p, q, test_size=0.2, random_state=i)
        std = (0.25 / x_test.shape[0]) ** 0.5
        mean = 0.5
        normalDistribution = norm(mean, std)
        model = Sequential()
        model.add(Dense(units=100, activation='relu', input_dim=p.shape[1]))
        model.add(Dense(units=50, activation='relu', input_dim=100))
        model.add(Dense(units=20, activation='relu', input_dim=50))
        model.add(Dense(units=1, activation='sigmoid'))
        #    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9,beta_2=0.999, epsilon=1e-08)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        hist = model.fit(x_train, y_train, epochs=epochs, verbose=0, batch_size=batch_size)
        score = model.evaluate(x_test, y_test)[-1]
        scores.append(score)
        print("score is {}".format(str(score)))
        p_value = normalDistribution.cdf(score)
        p_values.append(1-p_value)
        del model

    #    p_value.append(1-value)
    return np.mean(scores), np.mean(p_values)

def knn_test(p,q,count,n_splits,test_size):
    print("====knn====")
    n_neighbors = int((p.shape[0]*(1-test_size))**0.5)
    sklearn_knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    scores = cross_val_score(sklearn_knn_clf,p,q,cv=cv,scoring='accuracy')
    scores = sorted(scores)[1:-2]
    score = np.mean(scores)
    std = (0.25 / p.shape[0]*test_size) ** 0.5
    mean = 0.5
    normalDistribution = norm(mean, std)
    p_value = 1-normalDistribution.cdf(score)
#    print(scores)
    return score,p_value

# random forest classifier
def rf_test(p,q,count,n_splits,test_size):
    print("====rf====")
    from sklearn.ensemble import RandomForestClassifier
    n_estimators=200
    max_depth=15
    min_samples_leaf=2
    rfc = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf,random_state=90)
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    scores = cross_val_score(rfc,p,q,cv=cv,scoring='accuracy')
    scores = sorted(scores)[1:-2]
    score = np.mean(scores)
    std = (0.25 / p.shape[0]*test_size) ** 0.5
    mean = 0.5
    normalDistribution = norm(mean, std)
    p_value = 1-normalDistribution.cdf(score)
#    print(scores)
    return score,p_value


def svm_test_old(p,q,count,n_splits,test_size):
    print(p)
    print(q)
    from sklearn.model_selection  import GridSearchCV
    clf = svm.SVC(C=0.8, kernel='rbf', decision_function_shape='ovr',max_iter=1000,gamma='auto', class_weight='balanced')
#    clf = svm.LinearSVC(C = 0.8,penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, multi_class='ovr', 
#                        fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=0, max_iter=1000)
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    scores = cross_val_score(clf,p,q,cv=cv,scoring='accuracy')
    scores = sorted(scores)[1:-2]
    score = np.mean(scores)
    std = (0.25 / p.shape[0]*test_size) ** 0.5
    mean = 0.5
    normalDistribution = norm(mean, std)
    p_value = 1-normalDistribution.cdf(score)
    
    return score,p_value



def svm_test(datanumpy,labellist):
    label = np.array(labellist)
    data = np.mat(datanumpy)
    
    kernel = 'rbf'
    if(data.shape[0]<data.shape[1]):
        kernel = 'linear'
    clf = svm.SVC(C=100., kernel=kernel, decision_function_shape='ovr', max_iter=1000, gamma='auto',
                  class_weight='balanced',probability=True)
    KF = StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
#    KF = KFold(n_splits=5,random_state=0,shuffle=True)
    scores = []
    yPredList=[]
    yTrueList = []
    tprs=[]
    aucs=[]
    mean_fpr=np.linspace(0,1,100)
    for train_index,test_index in KF.split(data,label):
        X_train,X_test = data[train_index],data[test_index]
        Y_train,Y_test = label[train_index],label[test_index]
        clf.fit(X_train,Y_train)
        y_pred = clf.predict_proba(X_test)
        meanScore = clf.score(X_test,Y_test)
        scores.append(meanScore)
        yPredList.append(y_pred)
        yTrueList.append(Y_test)
    return yPredList,yTrueList,np.mean(scores)

def lr_test(p,q,count,n_splits,test_size):
    print("====lr====")
    logreg = LogisticRegression(solver='liblinear')
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    scores = cross_val_score(logreg,p,q,cv=cv,scoring='accuracy')
    scores = sorted(scores)[1:-2]
    score = np.mean(scores)
    std = (0.25 / p.shape[0]*test_size) ** 0.5
    mean = 0.5
    normalDistribution = norm(mean, std)
    p_value = 1-normalDistribution.cdf(score)
    print(scores)
    return score,p_value

def neural_test_all(p, q, count, num_train, epochs, batch_size,test_size):
    K.clear_session()
    scores = []
    p_values = []
    if (count % 5 == 0):
        print("======{}======".format(str(count)))

    for i in range(num_train):
        x_train, x_test, y_train, y_test = train_test_split(p, q, test_size=test_size, random_state=i)
        std = (0.25 / x_test.shape[0]) ** 0.5
        mean = 0.5
        normalDistribution = norm(mean, std)
        model = Sequential()
        model.add(Dense(units=500, activation='relu', input_dim=p.shape[1]))
        model.add(Dense(units=50, activation='relu', input_dim=500))
        model.add(Dense(units=20, activation='relu', input_dim=50))
        model.add(Dense(units=1, activation='sigmoid'))
        #    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9,beta_2=0.999, epsilon=1e-08)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        hist = model.fit(x_train, y_train, epochs=epochs,verbose=0, batch_size=batch_size)
        score = model.evaluate(x_test, y_test)[-1]
        scores.append(score)
        print("score is {}".format(str(score)))
        p_value = normalDistribution.cdf(score)
        p_values.append(p_value)
        del model
    #    p_value.append(1-value)
    return np.mean(scores), np.mean(p_values)

def c2st(cluter_num,data_first, data_second, label_first, label_second, community_num,first_cluster_num,second_cluster_num=-1,num_train=50, base_fileName='/res/', epochs=50, batch_size=16,test_size=0.1,classifier_type='svm'):
    length = min(len(label_first), len(label_second))
    print("__________________________start_________________________")
    acc_dict = {}
    p_value_dict = {}
    index = 0
    community_list = range(0,community_num)
    some_accuracy_data = pd.DataFrame(columns=['accuracy','F1','precession','recall'], index=community_list)
    for key, value in data_first.items():
        health_npy = value
        patient_npy = data_second.get(key)
        health_npy = health_npy.reshape((len(label_first), np.size(health_npy) // len(label_first)))
        patient_npy = patient_npy.reshape((len(label_second), np.size(patient_npy) // len(label_second)))

        np.random.seed(7)
        row_rand_array = np.arange(health_npy.shape[0])
        np.random.shuffle(row_rand_array)
        health_value = health_npy[row_rand_array[0:length]]
        row_rand_array = np.arange(patient_npy.shape[0])
        np.random.shuffle(row_rand_array)
        patient_value = patient_npy[row_rand_array[0:length]]
        x_data = np.concatenate((health_value, patient_value), axis=0)
        y_data = random.sample(label_first, length) + random.sample(label_second, length)
        #print(x_data)
        #print(y_data)
        
        np.savetxt("x_data.txt", x_data,fmt='%d',delimiter='\t')
        np.savetxt("y_data.txt", y_data,fmt='%d',delimiter='\t')
        #    x_data,y_data=shuffle(x_data,y_data)
        index += 1
        acc = 0
        p_value = 0
        if(classifier_type=="knn"):
            acc, p_value = knn_test(x_data, y_data, index, num_train, test_size=test_size)
        elif(classifier_type=="svm"):
            yPredList,yTrueList,acc = svm_test(x_data, y_data)
            save_accuracy_data(yTrueList,yPredList,key,some_accuracy_data)
            if acc > 0.58:
                plot_roc_curve(yTrueList,yPredList,2,save_path=base_fileName+'roc/',file_name="{}_{}_roc.jpg".format(str(first_cluster_num),str(key)),image_flag=True)           
                plot_confusion_matrix(yTrueList,yPredList,2,save_path=base_fileName+'confusion_matrix/',file_name="{}_{}_matrix.jpg".format(str(first_cluster_num),str(key)))
        elif(classifier_type=="lr"):
            acc, p_value = lr_test(x_data, y_data, index, num_train, test_size=test_size)
        elif (classifier_type == "rf"):
            acc, p_value = rf_test(x_data, y_data, index, num_train, test_size=test_size)
        acc_dict[key] = acc
        p_value_dict[key] = 1-p_value
    acc_fileName = base_fileName + 'c2st_accuracy/'
    dict_to_csv(cluter_num,acc_fileName,acc_dict,first_cluster_num,second_cluster_num)
    
    some_accuracy_datapath = base_fileName+ 'some_accuracy_data/'
    if not os.path.exists(some_accuracy_datapath):
        os.makedirs(some_accuracy_datapath)
    fileName =some_accuracy_datapath+ '{}.xlsx'.format(first_cluster_num)
    writer = pd.ExcelWriter(fileName)
    some_accuracy_data.to_excel(writer,index=True) 
    writer.save()
    #p_value_fileName = base_fileName + 'p_value/'
    #dict_to_csv(cluter_num,p_value_fileName, p_value_dict,first_cluster_num,second_cluster_num)
    return acc_dict, p_value_dict

def dict_to_csv(cluter_num,path,datas,first_cluster_num,second_cluster_num):
    if not os.path.exists(path):
        os.makedirs(path)
    filename=''
    if(second_cluster_num==-1):
        filename = path+'cluster_{}_class{}.csv'.format(str(cluter_num),str(first_cluster_num))
    else:
        filename = path+'cluster_{}_class{}_{}.csv'.format(str(cluter_num),str(first_cluster_num),str(second_cluster_num))
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for key, value in datas.items():
            writer.writerow([key, value])

def getclusterdata(key,column,subname):
    if os.path.exists('bloc/%s/%d/%s_cluster_genotype_%d.txt'%(subname,key,subname,key)):
        return
    with open('bloc/genotype_sz_case_QC_exchange.txt','r') as openfile:
        with open('bloc/%s/%d/%s_cluster_genotype_%d.txt'%(subname,key,subname,key),'w') as inputfile:
            index = 0
            for line in openfile:
                if index in column:
                    index = index +1
                    inputfile.writelines(line)
                else:
                    index = index+1
                    continue
        inputfile.close()
    openfile.close()
    freeMemory(inputfile)
    freeMemory(openfile)
def add(num,subname,key,key_num,threshold):
    startnum = 15000*num+1
    mid = 15000*(num+1)
    end = startnum+1
    print(num,startnum,mid,end)
    inputfile = 'bloc/%s/%d/%s_cluster_genotype_%d.txt'%(subname,key,subname,key)

    subnetfile = 'bloc/%s/%d/net/%d.txt'%(subname,key,num)
    cmd = "./ccc %s %s %f %d 404078 0 1 %d %d %d 404078"%(inputfile,subnetfile,threshold,key_num,startnum,mid,end)
    os.system(cmd)
    os.system("exit")
    freeMemory(inputfile)
    freeMemory(subnetfile)

def mergefile(path,filename):
    filedir = path 
    filenames=os.listdir(filedir)
    f=open(filename,'w')
    for filename in filenames:
        filepath = filedir+'/'+filename
        for line in open(filepath):
            f.writelines(line)
    f.close()
    shutil.rmtree(path)

#accuracy
def accuracy(resultor_communityInfo,inputfile,key_num,savepath,key):

    community_data = getCommunitySnp(resultor_communityInfo)
        
    community_num = len(community_data.keys())
    dict_case = getPersonCommunity(community_data,inputfile)
    dict_control = getPersonCommunity(community_data,'singlesnp/genotype_sz_control_QC_exchange.txt')
    one_hotdict_case = getOneHotSNP(dict_case)
    one_hotdict_control = getOneHotSNP(dict_control) 
    data_healthy = getone_hot_community_control(one_hotdict_control)
    data_patient = getone_hot_community_case(one_hotdict_case,key_num)
    label_healthy = []
    for l in range(0,283):
        label_healthy.append(0)
    np.array(label_healthy)
    label_patient = []
    for p in range(0,key_num):
        label_patient.append(1)
    np.array(label_patient)
    basefile = savepath+'/res/'
    if not os.path.exists(basefile):
        os.makedirs(basefile)
    #print(label_healthy)
    #print(label_patient)
    #print(inputfile,key_num,resultor_communityInfo)
    c2st(1, data_healthy,data_patient, label_healthy, label_patient, community_num,key,second_cluster_num=-1,num_train=5, base_fileName=basefile, epochs=50)
def communityBycarriers(subname,subnetfile,inputfile,key_num,key,resultor,resultor_communityInfo,threshold):
    subnetcommunity = 'bloc/%s/%d/%d_cluster_community_%f.bfs'%(subname,key,key,threshold)    
    getCommunity(subnetfile,subnetcommunity)
    cmd_carries = "./carriers %s %s bloc/genotype_sz_control_QC_exchange.txt 0 1 %d 283 404078 %s %s"%(subnetcommunity,inputfile,key_num,resultor,resultor_communityInfo)
    os.system(cmd_carries)
def ccc(subname,key,key_num,inputfile,subnetfile,threshold):
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    results=[] 
    path = 'bloc/%s/%d/net/'%(subname,key)
    if not os.path.exists(path):
        os.makedirs(path)
        
        
    print(os.getcwd())
    for i in range(0,26):           
        r = pool.apply_async(add,args=(i,subname,key,key_num,threshold,)) 
        results.append(r)     
    result_list = [xx.get() for xx in results]
    print(result_list)
    print(len(pool._cache))
    pool.close() 
    pool.join()  
    print(len(pool._cache))
    finalnetfile = 'bloc/%s/%d/net/27.txt'%(subname,key)
    cmd = "./ccc %s %s %f %d 404078 0 1 390000 404078 390001 404078"%(inputfile,finalnetfile,threshold,key_num)
    os.system(cmd)
    mergefile("bloc/%s/%d/net"%(subname,key),subnetfile)
    freeMemory(inputfile)
    freeMemory(finalnetfile)

def thresh(originsubnetfile,subnetfile,threshold):
    with open(originsubnetfile,'r') as FromFile:
        with open(subnetfile,'w') as OutFile:
            for line in FromFile:
                parts = line.strip().split('\t')
                weight = float(parts[2])
                if weight >= threshold:
                    OutFile.writelines(line)
    OutFile.close()
    FromFile.close()

def getAccuracyByCommunity(cluster,threshold,savepath,subname):
    y_list = cluster.reshape(-1).tolist()
    dict = {}
    for index in range(len(y_list)):
        if y_list[index] in dict:
            dict[y_list[index]].append(index)
        else:
            mylist = []
            mylist.append(index)
            dict[y_list[index]] = mylist
            
    
    for key in sorted(dict.keys()):

        subtypenetworkdir = 'bloc/%s/%d/'%(subname,key)
        if not os.path.exists(subtypenetworkdir):
            os.makedirs(subtypenetworkdir)
        key_num = len(dict[key])

        inputfile = 'bloc/%s/%d/%s_cluster_genotype_%d.txt'%(subname,key,subname,key)
        if not os.path.exists(inputfile):
            getclusterdata(key,dict[key],subname)
       
        subnetfile = 'bloc/%s/%d/%d_cluster_network_%f.txt'%(subname,key,key,threshold)
        originsubnetfile = 'bloc/%s/%d/%d_cluster_network.txt'%(subname,key,key)
        if not os.path.exists(subnetfile):
            if not os.path.exists(originsubnetfile):
                ccc(subname,key,key_num,inputfile,subnetfile,threshold) 
            else:
                thresh(originsubnetfile,subnetfile,threshold) 
        
        resultor = 'bloc/%s/%d/%d_cluster_result_%f.xlsx'%(subname,key,key,threshold)
        resultor_communityInfo = 'bloc/%s/%d/%d_cluster_communityid_%f.txt'%(subname,key,key,threshold)
        if not os.path.exists(resultor_communityInfo):
            communityBycarriers(subname,subnetfile,inputfile,key_num,key,resultor,resultor_communityInfo,threshold)
        
        if not os.path.exists(savepath+'/res/some_accuracy_data/%s.xlsx'%(key)):
            accuracy(resultor_communityInfo,inputfile,key_num,savepath,key)

        freeMemory(inputfile) 
        freeMemory(subnetfile)

        
    


