# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:05:08 2022

@author: dcdang
"""

import os
import pandas as pd
import numpy as np
import copy
import random
import seaborn as sns
import pickle
import scipy
from . import source


def BuildTargetMat(bin_st, bin_ed, mat_use, resolution, cut_dist = 5000000):
    """
    Get target Hi-C matrix to calculate domain value. 
    Parameters
    ----------
    bin_st : int
        Index of start bin
    bin_ed : int
        Index of end bin
    mat_use : numpy.array
        Hi-C matrix
    resolution : resolution
        Resolution of Hi-C data
    cut_dist : int, optional
        Range of distance to normalize matrix. The default is 5000000.

    Returns
    -------
    mat_target : numpy.array
        Distance dependent normalized Hi-C matrix between start and end bin. 
        Same shape with mat_use, fill zero in other positions.
    mat_target_norm_z : numpy.array
        Distance dependent normalized Hi-C matrix between start and end bin.
        Part of matrix
    mat_ori : numpy.array
        Original Hi-C matrix between start and end bin.
        Part of matrix
    """

    mat_target = np.zeros([len(mat_use), len(mat_use)])
    mat_use_target = mat_use[bin_st : bin_ed, bin_st:bin_ed] 
    norm_type = 'z-score'
    mat_target_norm_z = source.DistanceNormalizedMatrix(mat_use_target, resolution, norm_type, cut_dist)
    mat_target[bin_st : bin_ed, bin_st : bin_ed] =  mat_target_norm_z
    mat_ori = mat_use[bin_st : bin_ed, bin_st:bin_ed]
    return mat_target, mat_target_norm_z, mat_ori


def GetDomainValue(bin1, bin2, up_bound_use, down_bound_use, mat_target, diag_cut = True):
    """
    Calculate the domain value for selected bin pair.

    Parameters
    ----------
    bin1 : int
        Index of bin1.
    bin2 : int
        Index of bin2.
    up_bound_use : int
        Up bound to calculate up value.
    down_bound_use : int
        Down bound to calculate down value.
    mat_target : numpy.array
        Hi-C matrix used. Need normalization, eg. z-score
    diag_cut : bool, optional
        Whether filter the main diagonal. The default is True.

    Returns
    -------
    domain_value : float
        Mean value within domain.
    up_value : float
        Mean value within up-stream range.
    down_value : float
        Mean value within down-stream range.
    fold_v : float
        Difference between domain value and up or down value.
    pvalue_vec : TYPE
        DESCRIPTION.

    """
    domain_l = bin2 - bin1 + 1
    domain_mat = mat_target[bin1:bin2+1,bin1:bin2+1]
    if diag_cut == True:
        if domain_l <= 5:
            domain_value = np.mean(domain_mat)
            domain_vec = domain_mat.flatten()
        else:
            mat_extract = np.zeros([domain_l, domain_l])
            for j in range(domain_l-1):
                index_l = np.array([1 for i in range(domain_l - j -2)])
                mat_extract += np.diag(index_l, k = j+2)        
            mat_extract += mat_extract.T
            domain_value_all = domain_mat[(mat_extract + mat_extract.T) > 0]
            domain_value = np.mean(domain_value_all)
            domain_vec = domain_value_all
    else:
        domain_value = np.mean(domain_mat)
        domain_vec = domain_mat.flatten()

    #up_range = np.min([bin1 - up_bound, domain_l])
    #down_range = np.min([down_bound - bin2, domain_l])
    up_range = up_bound_use
    down_range = down_bound_use
    ## get up and down-stream regions' value
    if bin1 <= up_range + 1:
        up_value = 0
        up_vec = []
        down_mat = mat_target[bin1:bin2+1, bin2+1:bin2 + down_range + 1]
        down_vec = list(down_mat.flatten())
        down_value = np.mean(down_mat)
    elif bin2 >= len(mat_target) - down_range - 1:
        down_value = 0
        down_vec = []
        up_mat = mat_target[bin1 - up_range:bin1, bin1:bin2+1]
        up_vec = list(up_mat.flatten())
        up_value = np.mean(up_mat)
    else:
        up_mat = mat_target[bin1 - up_range:bin1, bin1:bin2+1]
        down_mat = mat_target[bin1:bin2+1, bin2+1:bin2 + down_range + 1]
        up_vec = list(up_mat.flatten())
        down_vec = list(down_mat.flatten())
        up_value = np.mean(up_mat)
        down_value = np.mean(down_mat)
    if domain_value == 0:
        fold_v = -1
        pvalue_vec = 1
    else:
        bk_vec = np.array(up_vec + down_vec)
        sta_vec, pvalue_vec = scipy.stats.mannwhitneyu(domain_vec, bk_vec, alternative = 'greater')
        #sta_vec, pvalue_vec =  scipy.stats.ttest_ind_from_stats(np.mean(domain_vec), np.std(domain_vec), len(domain_vec), np.mean(bk_vec), np.std(bk_vec), len(bk_vec), equal_var=False)                
                
        #fold_v = (domain_value - np.max([up_value, down_value])) / (domain_value + np.max([up_value, down_value]))
        #fold_v = domain_value - np.max([up_value, down_value])
        if up_value == 0:
            fold_v = domain_value - down_value
        elif down_value == 0:
            fold_v = domain_value - up_value
        else:
            fold_v = domain_value - np.max([up_value, down_value])
        return domain_value, up_value, down_value, fold_v, pvalue_vec


def GetBestBinPair(region_st, score_st, region_ed, score_ed, up_bound_use, down_bound_use, st_cut, mat_target, resolution, weight = 0.5):
    """
    Select the best bin pair in start and end boundary regions.
    Parameters
    ----------
    region_st : list
        Index of bins within start boundary region
    score_st : list
        Boundary score of bins within start boundary region
    region_ed : list
        Index of bins within end boundary region
    score_ed : TYPE
        Boundary score of bins within end boundary region
    up_bound_use : int
        Up bound to calculate up value.
    down_bound_use : int
        Down bound to calculate down value.
    st_cut : int
        Index of bin for end of last bin pair, used for 
        avoiding overlap of domain.
    mat_target : numpy.array
        Hi-C matrix
    resolution : int
        Resolution of Hi-C matrix
    weight : float
        weight to calculate best bin pair. 
        Balance affect of boundary score and domain value. The default is 0.5.

    Returns
    -------
    df_res : pandas.DataFrame
        Record of results for all possible bin pairs between start 
        and end boundary regions.

    """
    df_res = pd.DataFrame(columns = ['region_pair', 'score_pair', 'ave_score', 'domain_v', 'up_v', 'down_v', 'fold_v', 'pvalue_vec'])
    region_il = []
    score_vl = []
    ave_score_vl = []
    domain_vl = []
    up_vl = []
    down_vl = []
    fold_vl = []
    pvec_l = []          
    for i in range(len(region_st)):
        bin1 = region_st[i]
        score1 = score_st[i]
        up_bound_use = up_bound_use + (i + 1)
        if bin1 < st_cut and st_cut != 0:
            continue
        for j in range(len(region_ed)):
            bin2 = region_ed[j]
            score2 = score_ed[j]
            down_bound_use = down_bound_use + (len(region_ed) - j - 1)
            region_il.append([bin1, bin2])
            score_vl.append([score1, score2])
            ave_score_vl.append(np.mean([score1, score2]))
            domain_value, up_value, down_value, fold_v, pvalue_vec = GetDomainValue(bin1, bin2, up_bound_use, down_bound_use, mat_target, diag_cut = True)
            domain_vl.append(domain_value)
            up_vl.append(up_value)
            down_vl.append(down_value)
            fold_vl.append(fold_v)
            pvec_l.append(pvalue_vec)
    df_res['region_pair'] = region_il
    df_res['score_pair'] = score_vl
    df_res['ave_score'] = ave_score_vl
    df_res['domain_v'] = domain_vl
    df_res['up_v'] = up_vl
    df_res['down_v'] = down_vl
    df_res['fold_v'] = fold_vl
    df_res['pvalue_vec'] = pvec_l
    
    score_index = np.argsort(np.array(df_res['ave_score']))
    domain_v_index = np.argsort(np.array(df_res['domain_v']))
    score_rank = np.argsort(score_index)
    domain_v_rank = np.argsort(domain_v_index)
    df_res['score_rank'] = score_rank
    df_res['domain_rank'] = domain_v_rank
    df_res['judge'] = score_rank * weight + (1-weight)*domain_v_rank
    return df_res
    
    
def BoundaryMatch(df_bd_region_type, mat_use, resolution, weight):
    """
    Get best bin pairs between adjacent boundary regions.
    Parameters
    ----------
    df_bd_region_type : pandas.DataFrame
        Dataframe contains boundary region and boundary type.
    mat_use : numpy.array
        DESCRIPTION.
    resolution : int
        Resolution of Hi-C data
    weight : float
        weight to calculate best bin pair. 
        Balance affect of boundary score and domain value. 

    Returns
    -------
    df_record : pandas.DataFrame
        Dataframe to record selected bin pair between adjacent boundary regions

    """
    up_bound_l = list(df_bd_region_type['up_dist'])
    low_bound_l = list(df_bd_region_type['down_dist'])
    record = []
    st_cut = -1
    region_index = []
    for i in range(len(df_bd_region_type)-1):
        region_st = df_bd_region_type['region'][i]
        score_st = df_bd_region_type['score'][i]
        region_ed = df_bd_region_type['region'][i+1]
        score_ed = df_bd_region_type['score'][i+1]        
        up_bound_use = up_bound_l[i]
        down_bound_use = low_bound_l[i]
        
        domain_max = region_ed[-1] - region_st[0] + 1        
        expand_region = np.max([up_bound_use, down_bound_use, domain_max])        
        norm_st = np.max([0, region_st[0] - expand_region])
        norm_ed = np.min([region_ed[-1] + expand_region + 1, len(mat_use)])
        mat_target, mat_target_norm_z, mat_ori = BuildTargetMat(norm_st, norm_ed, mat_use, resolution, cut_dist = 5000000)

        df_res = GetBestBinPair(region_st, score_st, region_ed, score_ed, up_bound_use, down_bound_use, st_cut, mat_target, resolution, weight)
        index_max = np.argmax(np.array(df_res['judge']))       
        region_index.append([i, i+1])
        target = list(df_res.iloc[index_max])
        #if target[0][0] == st_cut:
           #target[0][0] = st_cut + 1 
        record.append(target)
        up_bound_l[i+1] = df_res['region_pair'][index_max][-1] - df_res['region_pair'][index_max][0]
        st_cut = df_res['region_pair'][index_max][-1]
    df_record = pd.DataFrame(record)
    df_record.columns = ['region_pair', 'score_pair', 'ave_score', 'domain_v', 'up_v', 'down_v', 'fold_v', 'pvalue_vec', 'score_rank', 'domain_rank', 'judge']
    df_record['region_record'] = region_index
    return df_record


def GetConsTadBoundary(df_record, df_bd_region_type, Chr, resolution):
    """
    Get ConsTAD domain and boundary
    Parameters
    ----------
    df_record : pandas.DataFrame
        Dataframe to record selected bin pair between adjacent boundary regions
    df_bd_region_type : pandas.DataFrame
        Dataframe contains boundary region and boundary type.
    Chr : str
        Symbol of chromosome
    resolution : int
        Resolution of Hi-C data

    Returns
    -------
    df_tad_cons : pandas.DataFrame
        DataFrame of ConsTADs
    df_boundary_cons : pandas.DataFrame
        DataFrame of ConsTADs boundary

    """
    df_tad_cons = pd.DataFrame(columns = ['chr', 'start', 'end', 'name', 'boundary_st', 'boundary_ed', 'bd_st', 'bd_region_st', 'bd_ed', 'bd_region_ed', 'st_region_type', 'ed_region_type'])
    df_boundary_cons = pd.DataFrame(columns = ['chr', 'start', 'end', 'TAD_name', 'boundary_region', 'region_type'])
    st_l = []
    ed_l = []
    st_use_l = []
    ed_use_l = []
    name_l = []
    bd_st_l = []
    bd_ed_l = []
    region_st_l = []
    region_ed_l = []
    st_type_l = []
    ed_type_l = []
    st_l_bd = []
    ed_l_bd = []
    tad_name_l = []
    region_bd_l = []
    type_bd_l = []
    for i in range(len(df_record)):
        st = df_record['region_pair'][i][0] * resolution
        ed = df_record['region_pair'][i][-1] * resolution + resolution
        st_use = df_record['region_pair'][i][0]
        ed_use = df_record['region_pair'][i][-1]
        name = 'Consensus_' + Chr + '_TAD_' + str(i) 
        bd_st = Chr + ':' + str(st) + '-' + str(st + resolution)
        bd_ed = Chr + ':' + str(ed-resolution) + '-' + str(ed)
        st_index = df_record['region_record'][i][0]
        ed_index = df_record['region_record'][i][-1]
        region_st = df_bd_region_type['region'][st_index]
        region_ed = df_bd_region_type['region'][ed_index]
        st_type = df_bd_region_type['region_type_adjust'][st_index]
        ed_type = df_bd_region_type['region_type_adjust'][ed_index]
        # tad related
        st_l.append(st)
        ed_l.append(ed)
        st_use_l.append(st_use)
        ed_use_l.append(ed_use)
        name_l.append(name)
        bd_st_l.append(bd_st)
        bd_ed_l.append(bd_ed)
        region_st_l.append(region_st)
        region_ed_l.append(region_ed)
        st_type_l.append(st_type)
        ed_type_l.append(ed_type)
        # bd related
        st_l_bd.append(st)
        ed_l_bd.append(st + resolution)
        tad_name_l.append(name)
        region_bd_l.append(region_st)
        type_bd_l.append(st_type)
        
        st_l_bd.append(ed - resolution)
        ed_l_bd.append(ed)
        tad_name_l.append(name)
        region_bd_l.append(region_ed)
        type_bd_l.append(ed_type)
       
    df_tad_cons['start'] = st_l
    df_tad_cons['end'] = ed_l
    df_tad_cons['name'] = name_l
    df_tad_cons['boundary_st'] = bd_st_l
    df_tad_cons['boundary_ed'] = bd_ed_l
    df_tad_cons['bd_region_st'] = region_st_l
    df_tad_cons['bd_st'] = st_use_l
    df_tad_cons['bd_ed'] = ed_use_l
    df_tad_cons['bd_region_ed'] = region_ed_l
    df_tad_cons['st_region_type'] = st_type_l
    df_tad_cons['ed_region_type'] = ed_type_l
    df_tad_cons['chr'] = [Chr for i in range(len(df_tad_cons))]

    df_boundary_cons['start'] = st_l_bd
    df_boundary_cons['end'] = ed_l_bd
    df_boundary_cons['TAD_name'] = tad_name_l
    df_boundary_cons['boundary_region'] = region_bd_l
    df_boundary_cons['region_type'] = type_bd_l
    df_boundary_cons['chr'] = [Chr for i in range(len(df_boundary_cons))]
    return df_tad_cons, df_boundary_cons


def BuildConsTADs(Chr, resolution, mat_use, df_bd_region_type, weight):
    """
    Build ConsTADs from boundary regions.
    Parameters
    ----------
    Chr : str
        Symbol of chromosome
    resolution : int
        Resolution of Hi-C data
    mat_use : numpy.array
        Hi-C matrix
    df_bd_region_type : pandas.DataFrame
        
    weight : float
        weight to calculate best bin pair. 
        Balance affect of boundary score and domain value.  

    Returns
    -------
    df_tad_cons : pandas.DataFrame
        DataFrame of ConsTADs
    df_boundary_cons : pandas.DataFrame
        DataFrame of ConsTADs boundary

    """
    df_record = BoundaryMatch(df_bd_region_type, mat_use, resolution, weight)
    df_tad_cons, df_boundary_cons = GetConsTadBoundary(df_record, df_bd_region_type, Chr, resolution)
    print('Get ' + str(len(df_tad_cons)) + ' ConsTADs')
    return df_tad_cons, df_boundary_cons







