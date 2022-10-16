# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:01:56 2022

@author: dcdang
"""

import os
import pandas as pd
import numpy as np
import scipy
import random
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def DrawKmeansSseWithMaxCnum(df_boundary_region_combine, K, save_name = ''):
    """
    Draw the sum of square error of K-means with maximal cluster number
    
    Parameters
    ----------
    df_boundary_region_combine : pandas.DataFrame
        DataFrame contains the boundary region defined from the boundary score profile.
    K : int
        Maximal cluster number
    save_name : TYPE, optional
        save path and name of figure. The default is ''.

    Returns
    -------
    None.

    """
    X = np.array(df_boundary_region_combine[['length', 'ave_score']])
    T = preprocessing.StandardScaler().fit(X)
    X=T.transform(X)
    SSE = []
    for k in range(1, K):
        estimator = KMeans(n_clusters=k)
        estimator.fit(X)
        SSE.append(estimator.inertia_)
    plt.figure(figsize=(5.5, 4.2))
    plt.xlabel('Cluster Number', FontSize = 12)
    plt.ylabel('SSE', FontSize = 12)
    plt.plot(range(1, K), SSE, 'o-', linewidth = 2)
    plt.xticks(FontSize = 12)
    plt.yticks(FontSize = 12)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)    
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() 
    plt.close(fig)
    

def DrawKmeansSilhouetteScoreWithMaxCnum(df_boundary_region_combine, K, save_name = ''):
    """
    Draw the silhouette score of K-means with maximal cluster number
    
    Parameters
    ----------
    df_boundary_region_combine : pandas.DataFrame
        DataFrame contains the boundary region defined from the boundary score profile.
    K : int
        Maximal cluster number
    save_name : TYPE, optional
        save path and name of figure. The default is ''.

    Returns
    -------
    None.

    """
    X = np.array(df_boundary_region_combine[['length', 'ave_score']])
    T = preprocessing.StandardScaler().fit(X)
    X=T.transform(X)
    Scores = [] 
    for k in range(2, K):
        estimator = KMeans(n_clusters=k)
        estimator.fit(X)
        Scores.append(silhouette_score(X, estimator.labels_, metric='euclidean'))
    plt.figure(figsize=(6, 4))
    plt.xlabel('Cluster Number', FontSize = 12)
    plt.ylabel('Silhouette score', FontSize = 12)
    plt.plot(range(2, K), Scores, 'o-', linewidth = 2)
    plt.xticks(FontSize = 12)
    plt.yticks(FontSize = 12)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)    
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)


def GetBoundaryRegionsWithKmeans(df_boundary_region_combine, K, color_use, save_name = '', permute=True):
    """
    Identify three types of boundary regions with Kmeans 
    based on the length and average scores of boundary regions 

    Parameters
    ----------
    df_boundary_region_combine : pandas.DataFrame
        DataFrame contains the boundary region defined from the boundary score profile.
    K : int
        Number of clusters (Types of boundary regions).
    color_use : list
        Color used for drawing scatter plot of boundary regions.
    save_name : str, optional
        save path and name of figure. The default is ''.
    permute : bool, optional
        Whether permute scatters to avoid overlap. The default is True.

    Returns
    -------
    df_boundary_region_combine : pandas.DataFrame
        DataFrame contains the boundary region with types.

    """
    X = np.array(df_boundary_region_combine[['length', 'ave_score']])
    T = preprocessing.StandardScaler().fit(X)
    Xn=T.transform(X)
    estimator = KMeans(n_clusters=K)
    estimator.fit(Xn)
    label_pred = estimator.labels_
    df_boundary_region_combine['region_label'] = list(label_pred)
    length_r_l = []
    score_r_l = []
    for i in range(K):
        x = X[label_pred == i]
        #print(len(x))
        length_r_l.append(np.mean(x[:, 0]))
        score_r_l.append(np.mean(x[:, 1]))    
    ind_len_max = np.argmax(length_r_l)
    ind_score_max = np.argmax(score_r_l)
    region_type = {}
    region_type[ind_len_max] = 'Wide'
    region_type[ind_score_max] = 'Narrow-strong'
    for k in range(K):
        if k == ind_len_max or k == ind_score_max:
            continue
        else:
            region_type[k] = 'Narrow-weak'
    #print(region_type)
    region_type_l = []
    for i in range(len(label_pred)):
        region_type_l.append(region_type[label_pred[i]])
    df_boundary_region_combine['region_type'] = list(region_type_l)
    
    '''
    plt.figure(figsize=(5, 4.2))
    type_list_r = ['Narrow-strong', 'Narrow-weak', 'Wide']
    for i in range(len(type_list_r)) :
        type_r = type_list_r[i]
        x = X[df_boundary_region_combine['region_type'] == type_r]
        #print(len(x))
        if permute == True:
            random_per = []
            for j in range(len(x)):
                random_per.append(random.uniform(-0.3,0.3))
            plt.scatter(x[:, 0] + np.array(random_per) , x[:, 1], c = color_use[i], marker='o', s=10, label=type_r)
        else:
            plt.scatter(x[:, 0], x[:, 1], c = color_use[i], marker='o', s=10, label=type_r)
    plt.legend(fontsize = 12, frameon = False)
    plt.xlabel('length',  fontsize = 12)
    plt.ylabel('Average Score',  fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    #plt.ylim([0,1])
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)   
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    #plt.show()
    #fig = plt.gcf()
    #plt.close(fig)
    ''' 
    return df_boundary_region_combine 
  
    
def AdjustBoundaryTypesByCutoff(df_boundary_region_with_type, color_use, save_name = '', permute=True):    
    """
    Adjust boundary region type by setting cut-off of boundary region length and average score 
    based on the K-means results

    Parameters
    ----------
    df_boundary_region_with_type : pandas.DataFrame
        DataFrame contains the boundary region with types.
    color_use : list
        Color used for drawing scatter plot of boundary regions.
    save_name : str, optional
        save path and name of figure. The default is ''.
    permute : bool, optional
        Whether permute scatters to avoid overlap. The default is True.

    Returns
    -------
    df_boundary_region_with_type : pandas.DataFrame
        DataFrame contains the adjusted boundary region types.

    """
    df_wide_part = df_boundary_region_with_type[df_boundary_region_with_type['region_type'] == 'Wide']
    cut_length = np.percentile(df_wide_part['length'], 10)    
    df_sharp_strong_part = df_boundary_region_with_type[df_boundary_region_with_type['region_type'] == 'Narrow-strong']    
    cut_score = np.percentile(df_sharp_strong_part['ave_score'], 10)
    cut_score = np.ceil(cut_score)
    print('Length cut-off:' + str(cut_length))
    print('Score cut-off:' + str(cut_score))
    region_type_adjust = []
    for i in range(len(df_boundary_region_with_type)):
        length = df_boundary_region_with_type['length'][i]
        score = df_boundary_region_with_type['ave_score'][i]
        if length >= cut_length:
            region_type_adjust.append('Wide')
        elif score >= cut_score:
            region_type_adjust.append('Narrow-strong')
        else:
            region_type_adjust.append('Narrow-weak')
    df_boundary_region_with_type['region_type_adjust'] = region_type_adjust
    #print('Adjust number:' + str(np.sum(df_boundary_region_with_type['region_type_adjust'] != df_boundary_region_with_type['region_type'])))
    
    plt.figure(figsize=(5, 4.2))
    X = np.array(df_boundary_region_with_type[['length', 'ave_score']])
    type_list_r = ['Narrow-strong', 'Narrow-weak', 'Wide']
    for i in range(len(type_list_r)) :
        type_r = type_list_r[i]
        x = X[df_boundary_region_with_type['region_type_adjust'] == type_r]
        print('Boundary region types: ' + type_r + ', Number: ' + str((len(x))))
        if permute == True:
            random_per = []
            for j in range(len(x)):
                random_per.append(random.uniform(-0.3,0.3))
            plt.scatter(x[:, 0] + np.array(random_per) , x[:, 1], c = color_use[i], marker='o', s=10, label=type_r)
        else:
            plt.scatter(x[:, 0], x[:, 1], c = color_use[i], marker='o', s=10, label=type_r)
    plt.legend(fontsize = 12, frameon = False)
    plt.xlabel('length (#bins)',  fontsize = 12)
    plt.ylabel('Average Score',  fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    #plt.ylim([0,1])
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)   
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    #plt.close(fig)  
    return df_boundary_region_with_type
            

def IdentifyBoundaryRegions(df_boundary_region_combine, K, color_use, save_name = '', permute=True):
    """
    Identify three types of boundary regions with Kmeans 
    based on the length and average scores of boundary regions 

    Parameters
    ----------
    df_boundary_region_combine : pandas.DataFrame
        DataFrame contains the boundary region defined from the boundary score profile.
    K : int
        Number of clusters (Types of boundary regions).
    color_use : list
        Color used for drawing scatter plot of boundary regions.
    save_name : str, optional
        save path and name of figure. The default is ''.
    permute : bool, optional
        Whether permute scatters to avoid overlap. The default is True.
   
    Returns
    -------
    df_boundary_region_with_types : pandas.DataFrame
        DataFrame contains the boundary region with types.
    """
    df_boundary_region_types = GetBoundaryRegionsWithKmeans(df_boundary_region_combine, K, color_use, save_name = '', permute=True)
    df_boundary_region_with_types = AdjustBoundaryTypesByCutoff(df_boundary_region_types, color_use, save_name = '', permute=True)
    return df_boundary_region_with_types






















