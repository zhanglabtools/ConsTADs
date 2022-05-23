# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 15:33:14 2022

@author: dcdang
"""

import os
import pandas as pd
import numpy as np
import pickle
import scipy
import copy



def SaveData(file, objects):
    save_file = open(file, 'wb')
    pickle.dump(objects, save_file, 2)


def ReadSaveData(file):
    read_file = open(file, 'rb')
    objects = pickle.load(read_file)
    read_file.close()
    return objects


def SparseMatrixToDense(df_mat_sparse, bin_num):
    df_mat_sparse.columns = ['bin1', 'bin2', 'value']
    row = np.array(df_mat_sparse['bin1'])
    col = np.array(df_mat_sparse['bin2'])
    val = np.array(df_mat_sparse['value'])
    mat_hic_sparse = scipy.sparse.csr_matrix((val, (row,col)), shape = (bin_num, bin_num))
    mat_dense_up = mat_hic_sparse.toarray()
    mat_dense_low = mat_dense_up.T
    mat_dense_diag = np.diag(np.diag(mat_dense_up))
    mat_dense = mat_dense_up + mat_dense_low - mat_dense_diag
    return mat_dense
    

def LoadHicMat(mat_file, bin_num, mat_type = 'dense'):
    """
    Load Hi-C data dense matrix. The default matrix file is in .csv format
    with \t separation.

    Parameters
    ----------
    mat_file : str
        File path of Hi-C data matrix.
    mat_type : str, optional
        Type of matrix file, dense or sparse. The default is 'dense'.

    Returns
    -------
    mat_hic : numpy.array
        Hi-C data dense matrix
    """
    if os.path.exists(mat_file) == False:
        print('Hi-C matrix do not exit!')
    if mat_type == 'dense':
        df_mat_dense = pd.read_csv(mat_file, sep = '\t', header = None)
        mat_hic = np.array(df_mat_dense.values)
    if mat_type == 'sparse':
        df_mat_sparse = pd.read_csv(mat_file, sep = '\t', header = None)
        mat_hic = SparseMatrixToDense(df_mat_sparse, bin_num) 
    return mat_hic


def DistanceNormalizedMatrix(mat_target, resolution, norm_type = 'z-score', cut_dist = 5000000):
    """
    Normalize Hi-C data matrix by each diagonal.

    Parameters
    ----------
    mat_target : numpy.array
        Input Hi-C data matrix.
    resolution : int
        Hi-C data resolution.
    norm_type : str, optional
        Type of normalization, including z-score, min-max, obs_exp. 
        The default is 'z-score'.
    cut_dist : int, optional
        Range of Hi-C data to normalize, represent bins within the distance.
        The default is 5000000.

    Returns
    -------
    mat_zscore : numpy.array
        Normalized Hi-C data matrix.

    """
    mat_normalize = mat_target
    mat_zscore = np.zeros([len(mat_normalize), len(mat_normalize)])
    cut = int(cut_dist / resolution)
    cut = np.min([cut, len(mat_zscore)])
    for i in range(cut):
        diag_value = np.diag(mat_normalize, i)
        if norm_type == 'z-score':
            if np.std(diag_value) == 0:
                diag_z_score = diag_value - np.mean(diag_value)
            else:
                diag_z_score = (diag_value - np.mean(diag_value)) / np.std(diag_value)
        elif norm_type == 'min-max':
            if np.max(diag_value) == np.min(diag_value):
                diag_z_score = diag_value - np.min(diag_value)
            else:
                diag_z_score = (diag_value - np.min(diag_value)) / (np.max(diag_value) - np.min(diag_value))
        elif norm_type == 'obs_exp':
            diag_value_copy = copy.deepcopy(diag_value)
            diag_value_copy[diag_value == 0] = np.mean(diag_value)
            diag_z_score = np.log2((diag_value_copy / np.mean(diag_value)))
        mat_zscore += np.diag(diag_z_score, k = i)
    mat_zscore = mat_zscore + mat_zscore.T - np.diag(np.diag(mat_zscore, 0), k = 0)   
    return mat_zscore




















