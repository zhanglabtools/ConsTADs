# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 21:39:46 2022

@author: dcdang
"""

import os
import pandas as pd
import numpy as np
import time
import scipy
import scipy.sparse
import scipy.stats 
import copy
import random
from . import source


def __get_bin_name_list_for_chr(chr_length, chr_symbol, resolution):
     start_pos = 0
     start = []
     end = []
     bin_name_chr = []
     while (start_pos + resolution) <= chr_length:
          start.append(start_pos)
          end.append(start_pos + resolution)
          start_pos += resolution
     start.append(start_pos)
     end.append(chr_length)
     for i in range(len(start)):
          bin_name_chr.append(chr_symbol + ':' + str(start[i]) + '-' + str(end[i]))
     return bin_name_chr


def __CollectBdScore(TAD_result, method_list, bin_name_chr, expand_bin = 0):
    """
    Collect scores contributed by each TAD caller.

    Parameters
    ----------
    TAD_result : dict
        Dictionary contains the the uniform format of TAD domain and TAD boundary
    method_list : list
        List of TAD caller
    bin_num : int
        Number of bins along the chromosome
    expand_bin : int, optional
        Window size of boundary voting. The default is 0.

    Returns
    -------
    boundary_score_record : dict
        Dictionary contains the boundary scores for each bin contributed by each TAD caller

    """
    boundary_score_record = {}
    for method in method_list:
        boundary_score_record[method] = {}
        df_domain = TAD_result[method]['TAD_domain']
        method_start_bin_score = np.zeros(len(bin_name_chr))
        method_end_bin_score = np.zeros(len(bin_name_chr))       
        bd_st_record = []
        bd_ed_record = []
        for i in range(len(df_domain)):
            boundary_st = df_domain['boundary_st'][i]
            boundary_ed = df_domain['boundary_ed'][i]
            if boundary_st not in bd_st_record:
                st_ind = bin_name_chr.index(boundary_st)
                method_start_bin_score[st_ind - expand_bin : st_ind + expand_bin + 1] += 1
                bd_st_record.append(boundary_st)
            if boundary_ed not in bd_ed_record:
                ed_ind = bin_name_chr.index(boundary_ed)
                method_end_bin_score[ed_ind - expand_bin : ed_ind + expand_bin + 1] += 1
                bd_ed_record.append(boundary_ed)
        if np.max(method_start_bin_score) > 1 or np.max(method_end_bin_score) > 1:
            print('Wrong bd score contribute!')
        boundary_score_record[method]['start_bin_score'] = method_start_bin_score
        boundary_score_record[method]['end_bin_score'] = method_end_bin_score
    return boundary_score_record


def __GetBdScore(boundary_score_record, method_list):
    """
    Get the boundary scores for each bin along the chromosome 
    based on the score contributed by all TAD callers 

    Parameters
    ----------
    boundary_score_record : dict
        Dictionary contains the boundary scores for each bin contributed by each TAD caller
    method_list : list
        List of TAD caller

    Returns
    -------
    bd_score_final : pandas.DataFrame
        Boundary scores for each bin along the chromosome.

    """
    df_method_bd_score_st_ed = pd.DataFrame(columns = method_list)
    for method in method_list:
        bd_score_st_ed = []
        bd_score_method_st = boundary_score_record[method]['start_bin_score']
        bd_score_method_ed = boundary_score_record[method]['end_bin_score']
        bd_score_st_ed.append(list(bd_score_method_st))
        bd_score_st_ed.append(list(bd_score_method_ed))
        bd_score_st_ed_max = np.max(bd_score_st_ed, axis = 0)
        df_method_bd_score_st_ed[method] = bd_score_st_ed_max
    bd_score_final = pd.DataFrame(np.sum(df_method_bd_score_st_ed, axis = 1))
    bd_score_final.columns = ['bd_score']
    return bd_score_final


def BoundaryVoting(TAD_result, method_list, bin_num, chr_length, chr_symbol, resolution, expand_bin = 0):
    """
    Process of boundary voting among different TAD callers.

    Parameters
    ----------
    TAD_result : dict
        Dictionary contains the the uniform format of TAD domain and TAD boundary
    method_list : list
        List of TAD caller
    bin_num : int
        Number of bins along the chromosome
    expand_bin : int, optional
        Window size of boundary voting. The default is 0.

    Returns
    -------
     bd_score_final : pandas.DataFrame
        Boundary scores for each bin along the chromosome.

    """
    bin_name_chr = __get_bin_name_list_for_chr(chr_length, chr_symbol, resolution)
    boundary_score_record = __CollectBdScore(TAD_result, method_list, bin_name_chr, expand_bin)
    bd_score_primary = __GetBdScore(boundary_score_record, method_list)
    return bd_score_primary



def GetMultiScaleContrastPvalue(bd_score_primary, mat_use, window_list, resolution):    
    """
    Calculate the multi-scale contrast P-value according to the input window_list.

    Parameters
    ----------
    bd_score_final : pandas.DataFrame
        Primary boundary score for each bin along the chromosome.
    mat_use : numpy.array
        Hi-C matrix used for contrast P-value calculation, need normalized (eg. z-score).
    window_list : list
        List of window size for multi-scale contrast P-value, in base pair.
    resolution : int
        Hi-C matrix resolution.

    Returns
    -------
    df_pvalue_result : pandas.DataFrame
    DataFrame contains the multi-scale contrast P-value for each bin along the chromosome    

    """
    df_pvalue_result = pd.DataFrame()
    for square_size in window_list:
        scale_label = str(int(square_size / 1000)) + 'kb-window'
        #print('Dealing with ' + scale_label)
        square_size = int(square_size / resolution)
        Sta_value_list = []
        mat_extract = np.zeros([square_size, square_size])
        for j in range(square_size):
            index_l = np.array([1 for i in range(square_size - j -1)])
            mat_extract += np.diag(index_l, k = j+1)        
        for i in range(len(mat_use)):
            if (i <= (square_size-1)) or (i >= (len(mat_use)-square_size)):
                Sta_value_list.append(-1)
            else:
                #y1 = x.ravel() modify can change original matrix
                #y2 = x.flatten() modify not change original matrix
                Cross_value = mat_use[i - square_size:i, i+1:i+square_size+1].flatten()
                up_mat = np.triu(mat_use[i - square_size:i, i - square_size:i], k = 1)
                down_mat = np.tril(mat_use[i+1:i+square_size+1, i+1:i+square_size+1], k =-1)                                
                #diag_ave_value = (np.diag(mat_use)[i - square_size:i] + np.diag(mat_use)[i+1:i+square_size+1]) /2
                #diag_ave_mat = np.diag(diag_ave_value)                
                #intra_mat = up_mat + down_mat + diag_ave_mat
                #Intra_value = intra_mat.flatten()
                
                intra_mat = up_mat + down_mat
                Intra_value = intra_mat[(mat_extract + mat_extract.T) > 0]
                if np.sum(Cross_value==0) >= len(Cross_value) or np.sum(Intra_value==0) >= len(Intra_value):
                    Sta_value_list.append(1)
                    continue
                sta, pvalue = scipy.stats.mannwhitneyu(Cross_value, Intra_value, alternative = 'less')
                #sta, pvalue =  scipy.stats.ttest_ind_from_stats(np.mean(Cross_value), np.std(Cross_value), len(Cross_value), np.mean(Intra_value), np.std(Intra_value), len(Intra_value), equal_var=False)                
                Sta_value_list.append(pvalue)
        df_pvalue_result[scale_label] = Sta_value_list
    df_pvalue_result['bd_score'] = bd_score_primary['bd_score']
    return df_pvalue_result


def GetMultiScaleContrastIndex(bd_score_primary, mat_use, window_list, resolution):    
    
    """
    Calculate the multi-scale contrast index according to the input window_list.

    Parameters
    ----------
    bd_score_primary : pandas.DataFrame
        Primary boundary score for each bin along the chromosome.
    mat_use : numpy.array
        Hi-C matrix used for contrast index calculation, need normalization (eg. z-score).
    window_list : list
        List of window size for multi-scale contrast index, in base pair.
    resolution : int
        Hi-C matrix resolution.

    Returns
    -------
    df_pvalue_result : pandas.DataFrame
    DataFrame contains the multi-scale contrast P-value for each bin along the chromosome    

    """
    df_Insvalue_result = pd.DataFrame()
    for square_size in window_list:
        scale_label = str(int(square_size / 1000)) + 'kb-window'
        print('Dealing with ' + scale_label)
        square_size = int(square_size / resolution)
        Ins_value_list = []
        mat_extract = np.zeros([square_size, square_size])
        for j in range(square_size):
            index_l = np.array([1 for i in range(square_size - j -1)])
            mat_extract += np.diag(index_l, k = j+1)        
        for i in range(len(mat_use)):
            if (i <= (square_size-1)) or (i >= (len(mat_use)-square_size)):
                Ins_value_list.append(-1)
            else:                
                #y1 = x.ravel() modify can change original matrix
                #y2 = x.flatten() modify not change original matrix
                Cross_value = mat_use[i - square_size:i, i+1:i+square_size+1].flatten()
                up_mat = np.triu(mat_use[i - square_size:i, i - square_size:i], k = 1)
                down_mat = np.tril(mat_use[i+1:i+square_size+1, i+1:i+square_size+1], k =-1)                                
                #diag_ave_value = (np.diag(mat_use)[i - square_size:i] + np.diag(mat_dense)[i+1:i+square_size+1]) /2
                #diag_ave_mat = np.diag(diag_ave_value)                
                #intra_mat = up_mat + down_mat + diag_ave_mat
                #Intra_value = intra_mat.flatten()
                
                intra_mat = up_mat + down_mat
                Intra_value = intra_mat[(mat_extract + mat_extract.T) > 0]
                if np.mean(Intra_value) == 0:
                    Ins_value_list.append(0)
                else:
                    CI_value = (np.mean(Intra_value) - np.mean(Cross_value)) / (np.mean(Intra_value) + np.mean(Cross_value))
                    Ins_value_list.append(CI_value)
        df_Insvalue_result[scale_label] = Ins_value_list
    df_Insvalue_result['bd_score'] = bd_score_primary['bd_score']
    return df_Insvalue_result


def GetBestWindowSizeForContrastPvalue(bd_score_primary, mat_use, window_list_multi, resolution):    
    """
    Select the best window size of the contrast P-value.

    Parameters
    ----------
    bd_score_primary : pandas.DataFrame
        Primary boundary score for each bin along the chromosome.
    mat_use : numpy.array
        Hi-C matrix used for contrast index calculation, need normalization (eg. z-score).
    window_list : list
        List of window size for multi-scale contrast index, in base pair.
    resolution : int
        Hi-C matrix resolution.

    Returns
    -------
    w_best : str
        Best window size for contrast P-value.
    df_pvalue_score_cor : pandas.DataFrame
        DataFrame contains pearson correlation of boundary score profile 
        and contrast P-value with different window size .

    """
    df_pvalue_score_cor = pd.DataFrame()
    w_list_multi = []
    cor_result_cell = []
    for w_use in window_list_multi:
       w_list_multi.append(str(int(w_use / 1000)) + 'kb-window')
    df_bd_insul_pvalue_multi =  GetMultiScaleContrastPvalue(bd_score_primary, mat_use, window_list_multi, resolution)
    for i in range(len(w_list_multi)):                
        w_use = w_list_multi[i]
        window_size = window_list_multi[i]
        w_cut = int(window_size / resolution)                
        cor_pvalue_score = scipy.stats.pearsonr(np.array(bd_score_primary['bd_score'].iloc[w_cut:len(bd_score_primary)-w_cut+1]), -np.array(df_bd_insul_pvalue_multi[w_use].iloc[w_cut:len(bd_score_primary)-w_cut+1]))[0]
        #cor_pvalue_score = scipy.stats.pearsonr(np.array(bd_score_primary['bd_score']), -np.array(df_bd_insul_pvalue_multi[w_use]))[0]
        #print('For ' + str(w_use))
        #print(cor_pvalue_score)
        cor_result_cell.append(cor_pvalue_score)            
    w_best = w_list_multi[np.argmax(cor_result_cell)]
    print('Best window size:' + w_best)
    df_pvalue_score_cor = pd.DataFrame(cor_result_cell)
    df_pvalue_score_cor.columns = ['PCC']
    df_pvalue_score_cor['window'] = w_list_multi
    return w_best, df_bd_insul_pvalue_multi, df_pvalue_score_cor


def GetBoundaryRegion(bd_score_primary, Chr):
    """
    Get boundary region from boundary score profile.

    Parameters
    ----------
    bd_score_primary : pandas.DataFrame.
        Primary boundary score for each bin along the chromosome.
    Chr : str
        Symbol of chromosome.

    Returns
    -------
    df_boundary_region : pandas.DataFrame.
        DataFrame contains the boundary region defined from the boundary score profile.

    """
    df_boundary_region = pd.DataFrame(columns = ['chr', 'start', 'end', 'length', 'region', 'score', 'ave_score', 'max_score', 'up_dist', 'down_dist'])
    st_list = []
    ed_list = []
    length_list = []
    region_l = []
    region_score_l = []
    ave_score_l = []
    max_score_l = []
    up_d_l = []
    down_d_l = []
    tad_seperation_score_new = copy.deepcopy(bd_score_primary)
    region_on = False
    for i in range(len(tad_seperation_score_new)):
        bd_score = tad_seperation_score_new['bd_score'][i]
        if bd_score != 0 and region_on == False:
            target = []
            score_r = []
            target.append(i)
            score_r.append(bd_score)
            region_on = True
            continue
        elif bd_score != 0 and region_on == True:
            target.append(i)
            score_r.append(bd_score)
        elif bd_score == 0 and region_on == True:
            region_on = False
            st_list.append(target[0])
            ed_list.append(target[-1])
            length_list.append(len(target))
            region_l.append(target)
            region_score_l.append(score_r)
            ave_score_l.append(np.mean(score_r))                    
            max_score_l.append(np.max(score_r))
            continue
        else:
            continue
    if region_on == True:
        st_list.append(target[0])
        ed_list.append(target[-1])
        length_list.append(len(target))
        region_l.append(target)
        region_score_l.append(score_r)
        ave_score_l.append(np.mean(score_r))                    
        max_score_l.append(np.max(score_r))
        region_on = False      
    df_boundary_region['start'] = st_list
    df_boundary_region['end'] = ed_list
    df_boundary_region['length'] = length_list
    df_boundary_region['region'] = region_l
    df_boundary_region['score'] = region_score_l
    df_boundary_region['ave_score'] = ave_score_l
    df_boundary_region['max_score'] = max_score_l                        
    df_boundary_region['chr'] = [Chr for j in range(len(df_boundary_region))]
    for i in range(len(df_boundary_region)):
        if i == 0:
            up_d_l.append(0)
            down_d_l.append(df_boundary_region['start'][i+1] - df_boundary_region['end'][i])
        elif i == len(df_boundary_region) - 1:
            down_d_l.append(0)
            up_d_l.append(df_boundary_region['start'][i] - df_boundary_region['end'][i-1])
        else:
            up_d_l.append(df_boundary_region['start'][i] - df_boundary_region['end'][i-1])            
            down_d_l.append(df_boundary_region['start'][i+1] - df_boundary_region['end'][i])
    df_boundary_region['up_dist'] = up_d_l
    df_boundary_region['down_dist'] = down_d_l
    return df_boundary_region


def AddOperation(bd_score_cell, df_bd_insul_pvalue, Chr, w_best, p_cut):
    """
    Add operation to refine the boundary score profile 
    by add 1 score to bins with good contrast P-value.

    Parameters
    ----------
    bd_score_cell : pandas.DataFrame
        boundary score for each bin along the chromosome.  
    df_bd_insul_pvalue : pandas.DataFrame
    DataFrame contains the multi-scale contrast P-value for each bin along the chromosome
    Chr : str
        Symbol of chromosome.
    w_best : str
        Best window size for contrast P-value.
    p_cut : float
        cut-off of contrast P-value

    Returns
    -------
    df_boundary_region_add : pandas.DataFrame.
        DataFrame contains the boundary region defined from the boundary score profile.
    bd_score_cell_add : pandas.DataFrame.
        boundary score for each bin along the chromosome after Add operation

    """
    num = 0
    bd_score_cell_add = copy.deepcopy(bd_score_cell)
    for i in range(len(bd_score_cell)):
        bd_score = bd_score_cell['bd_score'][i]
        bd_pvalue = df_bd_insul_pvalue[w_best][i]
        if bd_pvalue == -1:
            continue
        if bd_score == 0 and bd_pvalue <= p_cut:
            bd_score_cell_add['bd_score'][i] = 1
            num += 1
    print('Add score for ' + str(num) + ' bins')
    df_boundary_region_add = GetBoundaryRegion(bd_score_cell_add, Chr)
    return df_boundary_region_add, bd_score_cell_add
    

def GetLocalMinInPvalue(df_bd_insul_pvalue, w_best, p_cut):   
    local_min_judge = [0]
    for i in range(1, len(df_bd_insul_pvalue)-1):
        if df_bd_insul_pvalue[w_best][i] == -1 or df_bd_insul_pvalue[w_best][i] <= p_cut:
            local_min_judge.append(0)
        else:
            p_h = df_bd_insul_pvalue[w_best][i]
            p_up = df_bd_insul_pvalue[w_best][i-1]
            p_down = df_bd_insul_pvalue[w_best][i+1]
            p_dif = np.max([p_up - p_h, p_down - p_h])
            if p_h <= p_up and p_h <= p_down and p_dif > 0.01 and p_h < 0.9:
                local_min_judge.append(1)
            else:
                local_min_judge.append(0)
    local_min_judge.append(0)
    expand_l = []
    for i in range(2, len(local_min_judge)-2):
        if local_min_judge[i] == 0:
            continue
        else:
            up2 = df_bd_insul_pvalue[w_best][i-2]
            up1 = df_bd_insul_pvalue[w_best][i-1]
            down1 = df_bd_insul_pvalue[w_best][i+1]
            down2 = df_bd_insul_pvalue[w_best][i+2]
            if up1 < up2 and up1 < down2:
                expand_l.append(i-1)
            if down1 < up2 and down1 < down2:
                expand_l.append(i+1)
    for i in range(len(expand_l)):
        local_min_judge[expand_l[i]] = 1
    return local_min_judge


def FilterOperation(bd_score_cell, df_bd_insul_pvalue, Chr, w_best, p_cut, high_score_cut = 5):
    """
    Filter operation to refine the boundary score profile 
    by filter out bins with bad contrast P-values.

    Parameters
    ----------
    bd_score_cell : pandas.DataFrame
        boundary score for each bin along the chromosome.  
    df_bd_insul_pvalue : pandas.DataFrame
     DataFrame contains the multi-scale contrast P-value for each bin along the chromosome
    Chr : str
        Symbol of chromosome.
    w_best : str
        Best window size for contrast P-value.
    p_cut : float
        cut-off of contrast P-value
    high_score_cut : int, optional
        Cut off boundary score. The default is 5.
        Bins with score abover the cut off will not be filtered.

    Returns
    -------
    df_boundary_region_adjust : pandas.DataFrame.
        DataFrame contains the boundary region defined from the boundary score profile.
    bd_score_cell_adjust : TYPE
        boundary score for each bin along the chromosome after Filter operation

    """
    local_min_judge = GetLocalMinInPvalue(df_bd_insul_pvalue, w_best, p_cut)      
    bd_score_cell_adjust = copy.deepcopy(bd_score_cell)
    bd_score_cell_adjust[(bd_score_cell_adjust['bd_score'] < high_score_cut) & (df_bd_insul_pvalue[w_best] > p_cut)] = 0
    for i in range(len(local_min_judge)):
        if local_min_judge[i] == 0:
            continue
        else:
            score_hold = copy.deepcopy(bd_score_cell['bd_score'][i])
            if bd_score_cell_adjust['bd_score'][i] == 0:
                bd_score_cell_adjust['bd_score'][i] = score_hold
    df_boundary_region_adjust = GetBoundaryRegion(bd_score_cell_adjust, Chr)
    return df_boundary_region_adjust, bd_score_cell_adjust
 

def CombineOperation(df_boundary_region, bd_score_cell, Chr, combine_dist = 2):
    """
    Combine operation to refine the boundary score profile 
    by combine two close boundary regions.

    Parameters
    ----------
    df_boundary_region : pandas.DataFrame.
        DataFrame contains the boundary region defined from the boundary score profile.
    bd_score_cell : pandas.DataFrame
        boundary score for each bin along the chromosome.
    Chr : str
        Symbol of chromosome.
    combine_dist : int, optional
        Distance cut off to combine two boundary regions. The default is 2.

    Returns
    -------
    df_boundary_region_combine : pandas.DataFrame.
        DataFrame contains the boundary region defined from the boundary score profile.
    bd_score_cell_combine : pandas.DataFrame.
        boundary score for each bin along the chromosome after Add operation
    """
    bd_score_cell_combine = copy.deepcopy(bd_score_cell)
    num = 0
    for i in range(len(df_boundary_region)-1):
        ed = df_boundary_region['end'][i]
        down_dist = df_boundary_region['down_dist'][i]
        if down_dist <= combine_dist:
            st_next = df_boundary_region['start'][i+1]
            fill_score = (bd_score_cell['bd_score'][ed] + bd_score_cell['bd_score'][st_next]) / 2
            bd_score_cell_combine[ed + 1: ed + down_dist] = fill_score
            num += 1
    #print('There are ' + str(num) + ' times combination.')
    df_boundary_region_combine = GetBoundaryRegion(bd_score_cell_combine, Chr)
    return df_boundary_region_combine, bd_score_cell_combine 


def BuildTadSeperationLandscape(Chr, resolution, mat_file, bin_num, mat_type, mat_norm_check, bd_score_cell, window_list, 
                                   p_cut, high_score_cut, combine_dist, norm_type = 'z-score', cut_dist = 12000000):
    """
    Function to build TAD separation landscape.

    Parameters
    ----------
    Chr : str
        Symbol of chromosome
    resolution : int
        Resolution of Hi-C data
    mat_file : str
        File path of Hi-C matrix
    bin_num : int
        Number of bins for target chromosome
    mat_type : str
        Type of Hi-C matrix, dense or sparse
    mat_norm_check : bool
        For distance-dependent normalized Hi-C matrix: True;
        otherwise: False
    bd_score_cell : pandas.DataFrame
        boundary score for each bin along the chromosome.
    window_list : list
        Multi-scale window size for contrast P-value computation (base pair)
    p_cut : float
        Cut-off for contrast P-value
    high_score_cut : int
        Cut-off for boundary score filter.
    combine_dist : int
        Cut-off for combination of nearby boundary regions
    norm_type : str, optional
        Type of distance-dependent normalization. The default is 'z-score', 
        other like 'min-max', 'obs_exp'. 
    cut_dist : int, optional
        Up-bound for distance-dependent normalization. The default is 12000000.

    Returns
    -------
    mat_dense : numpy.array
        Dense Hi-C matrix
    mat_norm : numpy.array
        Distance-dependent normalized Hi-C matrix
    result_record : dict
        Boundary score profile and correspondig boundary regions 
        obtained by three operations
    w_best : str
        Best window for contrast P-value
    df_bd_insul_pvalue : pandas.DataFrame
        Multi-scale contrast P-value for bins along the chromosome
    df_pvalue_score_cor : TYPE
        Pearson correlation between Multi-scale contrast P-value and primary boundary score profile

    """
    result_record = {}
    print('Load Hi-C matrix...')
    mat_dense = source.LoadHicMat(mat_file, bin_num, mat_type)
    if mat_norm_check == False:
        print('Normalize the Hi-C matrix...')
        mat_norm = source.DistanceNormalizedMatrix(mat_dense, resolution, norm_type, cut_dist)
    else:
        print('Normalization done')
        mat_norm = mat_dense
    
    print('Calculate multi-scale contrast pvalue...')    
    w_best, df_bd_insul_pvalue, df_pvalue_score_cor = GetBestWindowSizeForContrastPvalue(bd_score_cell, mat_norm, window_list, resolution)
    
    print('Building TAD seperation landscape...')
    df_boundary_region = GetBoundaryRegion(bd_score_cell, Chr)    
    result_record['Original'] = {}
    result_record['Original']['bd_region'] = df_boundary_region
    result_record['Original']['TAD_score'] = bd_score_cell
    
    print('Operation 1: Add')
    df_boundary_region_add, bd_score_cell_add = AddOperation(bd_score_cell, df_bd_insul_pvalue, Chr, w_best, p_cut)
    result_record['Add'] = {}
    result_record['Add']['bd_region'] = df_boundary_region_add
    result_record['Add']['TAD_score'] = bd_score_cell_add
    
    print('Operation 2: Filter')
    df_boundary_region_adjust, bd_score_cell_adjust = FilterOperation(bd_score_cell_add, df_bd_insul_pvalue, Chr, w_best, p_cut, high_score_cut)
    result_record['Filter'] = {}
    result_record['Filter']['bd_region'] = df_boundary_region_adjust
    result_record['Filter']['TAD_score'] = bd_score_cell_adjust
       
    print('Operation 3: Combine')
    df_boundary_region_combine, bd_score_cell_combine = CombineOperation(df_boundary_region_adjust, bd_score_cell_adjust, Chr, combine_dist)
    result_record['Combine'] = {}
    result_record['Combine']['bd_region'] = df_boundary_region_combine
    result_record['Combine']['TAD_score'] = bd_score_cell_combine
    
    return mat_dense, mat_norm, result_record, w_best, df_bd_insul_pvalue, df_pvalue_score_cor

































