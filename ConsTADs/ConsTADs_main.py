# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:37:13 2022

@author: dcdang
"""

import sys
import copy
import numpy as np

from . import Preprocess as pre
from . import TadSeparationLandscape as TSL
from . import IdentifyBoundaryRegions as IBR
from . import GetConsTADs as GCT

#src_path = 'E:/Users/dcdang/project/monkey project/TAD_intergare/ConsTADs_script/scripts'
#if src_path not in sys.path:
    #sys.path.append(src_path)

def ConsTADs(TAD_caller_result_add, target_chr, resolution, chr_size, method_list,
             mat_file, mat_type, window_list, mat_norm_check = False, p_cut = 0.05, high_score_cut = 5, combine_dist = 2,
             K = 3, color_bd = ['#D65F4D', '#459457', '#4392C3'], 
             weight = 0.5):
    """
    

    Parameters
    ----------
    TAD_caller_result_add : str
        Path of fold containing results of multiple TAD callers.
    target_chr : str
        Symbol of target chromosome, eg: chr2.
    resolution : int
        Resolution of Hi-C contact map.
    chr_size : int
        Length of target chromosome in base pair.
    method_list : list
        List of TAD caller names.
    mat_file : str
        Path of file for Hi-C contact map of target chr.
    mat_type : str
        Type of Hi-C contact map file, dense or sparse.
    window_list : list
        List of window size for calculating multi-scale contrast p-value. These sizes are lengths in base pair. 
    mat_norm_check : bool, optional
        Dose the Hi-C contact map are distance-dependent normalized. The default is False. 
    p_cut : float, optional
        Cut off of contrast p-value. The default is 0.05.
    high_score_cut : int, optional
        Cut off of confident boundary score. The default is 5.
    combine_dist : int, optional
        The cut off of distance for boundary regions' combination. The default is 2.
    K : int, optional
        Number of boundary region types. The default is 3.
    color_bd : list, optional
        List of color for different boundary region types. The default is ['#D65F4D', '#459457', '#4392C3'].
    weight : float, optional
        weight to calculate best bin pair. 
        Balance affect of boundary score and domain value. The default is 0.5 for boundary score.

    Returns
    -------
    TAD_caller_result_all : dict
        Dictionary containing the results of multiple TAD callers for the Hi-C contact map of target chromosme.
    mat_dense : numpy.array
        Hi-C contact map in dense format.
    mat_norm : numpy.array
        Distance-dependent normalized Hi-C contact map in dense format.
    result_record : dict
        Result of each oreration when building the TAD separation landscape.
    w_best : int
        Best window size for contrast p-value.
    df_bd_insul_pvalue : pandas.DataFrame
        Multi-scale contrast p-value for each bin along the chromosome.
    df_pvalue_score_cor : pandas.DataFrame
        Pearson correlation of multi-scale contrast p-value and boundary score profile.
    df_boundary_region_with_types : pandas.DataFrame
        DataFrame of Boundary regions and their types.
    df_tad_cons : pandas.DataFrame
        DataFrame of ConsTADs.
    df_boundary_cons : pandas.DataFrame
        DataFrame of ConsTADs boundaries.

    """
    bin_num = int(np.ceil(chr_size / resolution))
    print( '\033[1mStep 1: Preprocess of TAD caller results...\033[0m' )
    TAD_caller_result_all = pre.PreprocessTADs(TAD_caller_result_add, method_list, resolution, target_chr, chr_size)
    print('Done!')
    
    print('\033[1mStep 2: Boundary voting...\033[0m')
    bd_score_primary = TSL.BoundaryVoting(TAD_caller_result_all, method_list, bin_num, chr_size, target_chr, resolution, expand_bin = 0)
    print('Done!')
    
    print('\033[1mStep 3: Build TAD separation landscape...\033[0m')
    mat_dense, mat_norm, result_record, w_best, df_bd_insul_pvalue, df_pvalue_score_cor = TSL.BuildTadSeperationLandscape(target_chr, resolution, mat_file, bin_num, mat_type, mat_norm_check, bd_score_primary, window_list, 
                                   p_cut, high_score_cut, combine_dist, norm_type = 'z-score', cut_dist = 12000000)
    print('Done!')
   
    print('\033[1mStep 4: Identify three types of boundary regions...\033[0m')
    df_boundary_region_combine = copy.deepcopy(result_record['Combine']['bd_region'])
    df_boundary_region_with_types = IBR.IdentifyBoundaryRegions(df_boundary_region_combine, K, color_bd, save_name = '', permute=True)
    print('Done!')
    
    print('\033[1mStep 5: Get ConsTADs based on TAD separation landscape...\033[0m')
    df_tad_cons, df_boundary_cons = GCT.BuildConsTADs(target_chr, resolution, mat_dense, df_boundary_region_with_types, weight)
    print('Done!')
    
    return TAD_caller_result_all, mat_dense, mat_norm, result_record, w_best, df_bd_insul_pvalue, df_pvalue_score_cor, df_boundary_region_with_types, df_tad_cons, df_boundary_cons
















