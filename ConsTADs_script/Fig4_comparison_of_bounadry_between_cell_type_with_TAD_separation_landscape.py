# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 00:06:09 2022

@author: dcdang
"""

import os
import pandas as pd
import numpy as np
import scipy.sparse 
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import seaborn as sns
import pickle
import scipy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
from itertools import combinations

#################################  load TAD caller results

def save_data(file, objects):
    save_file = open(file, 'wb')
    pickle.dump(objects, save_file, 2)

def read_save_data(file):
    read_file = open(file, 'rb')
    objects = pickle.load(read_file)
    read_file.close()
    return objects

def get_method_color(method_list, color_list11):
    method_color = {}
    for i in range(len(method_list)):
        method = method_list[i]
        color = color_list11[i]
        method_color[method] = color
    return method_color


add = 'E:/Users/dcdang/TAD_intergate/final_run/compare'
enzyme_list = ['DpnII', 'MboI']
target_chr = 'chr2'
resolution = 50000
chr_size = 243199373

method_list = ['3DNetMod', 'CaTCH', 'CHDF', 'ClusterTAD', 
               'deDoc', 'DI', 'GMAP', 'HiCDB', 'HiCseg', 
               'HiTAD', 'ICFinder', 'IS', 'MSTD', 'OnTAD',
               'Spectral','TopDom']

color_list16 = ['#317FBA', '#CB2027', '#E5AB22', '#A65627',
              '#3B8C41', '#E48BBB', '#984F9F', '#F26B4D',
              '#76AD75', '#996666', '#E9DC6C', '#7570B3', 
              '#00C0F6', '#333396', '#676733', '#ED5B64']

#sns.palplot(color_list16)

method_color = get_method_color(method_list, color_list16)

tad_result_add = 'E:/Users/dcdang/TAD_intergate/final_run/cell_type_result'
cell_type_list = ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'K562', 'KBM7', 'NHEK']
cell_color_all = ['#467FB4', '#D33127', '#6EAF5E', '#835A91', '#E27134', '#52A7A6', '#A7A03E']
cell_color = {}
for cell_type in cell_type_list:
    cell_ind = cell_type_list.index(cell_type)
    cell_color[cell_type] = cell_color_all[cell_ind]        

TAD_result_all_cell_type = read_save_data('E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape' + '/' + 'TAD_result_all_cell_type.pkl')



######################  load hic mat matrix
 
def get_chr_ref(species):
     if species == 'monkey':
          chr_ref = 'rheMac8'
     elif species == 'human':
          chr_ref = 'hg19'
     elif species == 'mouse':
          chr_ref = 'mm10'
     elif species == 'dog':
          chr_ref = 'canFam3'
     elif species == 'macaque':
          chr_ref = 'rheMac2'
     elif species == 'rabbit':
          chr_ref = 'oryCun2'
     else:
          print('Something wrong!')
     return chr_ref 

def chr_cut(chr_length, chr_symbol, resolution, chr_ref):
     start_pos = 0
     start = []
     end = []
     name_list = []
     while (start_pos + resolution) <= chr_length:
          start.append(start_pos)
          end.append(start_pos + resolution)
          start_pos += resolution
     start.append(start_pos)
     end.append(chr_length)
     for i in range(len(start)):
          name_list.append(chr_symbol + ':' + str(start[i]) + '-' + str(end[i]))
     return name_list

def get_bin_name_list_for_chr(species, resolution, chr_ref, ref_add):
     chr_size = {}
     ref_file = ref_add + '/' + 'chrom_' + chr_ref + '_sizes.txt'
     df_ref = pd.read_csv(ref_file, sep = '\t', header = None)
     for i in range(len(df_ref)):
          Chr = df_ref[0][i]
          chr_size[Chr] = df_ref[1][i]
     chr_list = list(chr_size.keys())
     bin_name_list = {}
     for Chr in chr_list:
          chr_length = chr_size[Chr]
          name_list = chr_cut(chr_length, Chr, resolution, chr_ref)
          bin_name_list[Chr] = name_list     
     return bin_name_list

def save_bin_name_ord_file(bin_name_Chr, save_name):
    df_bin_name = pd.DataFrame(columns = ['chr', 'start', 'end', 'index'])
    chr_list = []
    st_list = []
    ed_list = []
    for i in range(len(bin_name_Chr)):
        target = bin_name_Chr[i]
        Chr = target.split(':')[0]
        st = int(target.split(':')[-1].split('-')[0])
        ed = int(target.split(':')[-1].split('-')[-1])
        chr_list.append(Chr)
        st_list.append(st)
        ed_list.append(ed)
    df_bin_name['chr'] = chr_list
    df_bin_name['start'] = st_list
    df_bin_name['end'] = ed_list
    df_bin_name['index'] = np.array(range(len(df_bin_name))) + 1
    df_bin_name.to_csv(save_name, sep = '\t', header = None, index = None)

def dist_normalize_matrix(mat_target, resolution, norm_type = 'z-score', cut_dist = 5000000, e_0 = 10**(-5)):
    mat_normalize = copy.deepcopy(mat_target)
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
   
resolution = 50000
chr_symbol = 'chr2'
species = 'human'
sparse_mat_add = 'E:/Users/dcdang/share/TAD_integrate/HiC_data_matrix'
ref_add = 'E:/Users/dcdang/multi_sepcies_project/multi_species_hic_results/reference_size' 
res_symbol = str(int(resolution / 1000))

hic_mat_all_cell_replicate = {}
for cell_type in cell_type_list:
    if cell_type == 'GM12878':
        enzyme_list = ['DpnII', 'MboI']
    else:
        enzyme_list = ['MboI']
    hic_mat_all_cell_replicate[cell_type] = {}    
    for enzyme in enzyme_list:
         hic_mat_all_cell_replicate[cell_type][enzyme] = {}

for cell_type in cell_type_list:
    print('This is ' + cell_type)
    if cell_type == 'GM12878':
        enzyme_list = ['DpnII', 'MboI']
    else:
        enzyme_list = ['MboI']
    for enzyme in enzyme_list:
        print('We are dealing with ' + enzyme)
        chr_ref = get_chr_ref(species)
        bin_name_list = get_bin_name_list_for_chr(species, resolution, chr_ref, ref_add)
        for data_type in ['raw', 'iced']:
            print('Loading ' + data_type + ' Hi-C data...')
            length = len(bin_name_list[chr_symbol])
            df_sparse = pd.read_csv(sparse_mat_add + '/' + 'Rao2014_' + cell_type + 
                                    '_' + enzyme + '_' + res_symbol + 'kb_' + 
                                    chr_symbol + '_' + data_type + '_sparse.txt', sep = '\t', header = None)
            df_sparse.columns = ['bin1', 'bin2', 'value']
            row = np.array(df_sparse['bin1'])
            col = np.array(df_sparse['bin2'])
            val = np.array(df_sparse['value'])
            
            mat_hic_sparse = scipy.sparse.csr_matrix((val, (row,col)), shape = (length, length))
            mat_dense_up = mat_hic_sparse.toarray()
            mat_dense_low = mat_dense_up.T
            mat_dense_diag = np.diag(np.diag(mat_dense_up))
            mat_dense = mat_dense_up + mat_dense_low - mat_dense_diag              
            
            hic_mat_all_cell_replicate[cell_type][enzyme][data_type] = mat_dense
            print('Done!')

Chr = 'chr2'
cut_dist = 12000000
resolution = 50000
filter_length = 1000000
norm_type = 'z-score'
p_cut = 0.05

hic_mat_all_cell_replicate_znorm = {}
for cell_type in cell_type_list:
    print('This is ' + cell_type)
    hic_mat_all_cell_replicate_znorm[cell_type] = {}
    if cell_type == 'GM12878':
        enzyme_list = ['DpnII', 'MboI']
    else:
        enzyme_list = ['MboI']
    for enzyme in enzyme_list:
        print('For ' + enzyme)
        mat_dense = hic_mat_all_cell_replicate[cell_type][enzyme]['iced']
        mat_norm_z = dist_normalize_matrix(mat_dense, resolution, norm_type, cut_dist)
        hic_mat_all_cell_replicate_znorm[cell_type][enzyme] = mat_norm_z   



########################### get three kinds of 1D indicator (DI, IS, CI)

### IS (not match with the output IS, use the IS of method instead)
def get_IS_value(mat_dense, filter_size, square_size, resolution):
    filter_size = int(filter_size / resolution)
    square_size = int(square_size / resolution)
    IS_list = []
    for i in range(len(mat_dense)):
        if (i <= (filter_size-1)) or (i >= (len(mat_dense)-filter_size)):
            IS_list.append(-1)
        else:
            C_value = np.mean(mat_dense[i - square_size:i, i+1:i+square_size+1])
            IS_list.append(C_value)
    IS_value_all = pd.DataFrame(IS_list)
    IS_value_all_part = IS_value_all[IS_value_all[0] != -1]
    mean_IS = np.mean(IS_value_all_part[0])
    IS_value_list = []
    for i in range(len(IS_list)):
        if IS_list[i] == -1:
            IS_value_list.append(-100)
        else:
            IS_value_list.append(np.log2(IS_list[i] / mean_IS))
    return IS_value_list

### DI (match with the output of method)
def get_DI_value(mat_dense, window, resolution):
    window_size = int(window / resolution)
    DI_list = []
    for i in range(len(mat_dense)):
        if i < window_size:
            sum_A = np.sum(mat_dense[i:i+1, 0:i])
            sum_B = np.sum(mat_dense[i:i+1, i+1:i+window_size+1])
        elif i >= len(mat_dense) - window_size:
            sum_A = np.sum(mat_dense[i:i+1, i-window_size:i])
            sum_B = np.sum(mat_dense[i:i+1, i+1:len(mat_dense)])
        else:
            sum_A = np.sum(mat_dense[i:i+1, i-window_size:i])
            sum_B = np.sum(mat_dense[i:i+1, i+1:i+window_size+1])
        E_value = (sum_A + sum_B) / 2
        DI_value = (sum_B-sum_A) / np.abs(sum_B-sum_A) * ((sum_A - E_value)**2 / E_value + (sum_B - E_value)**2 / E_value)
        DI_list.append(DI_value)
    DI_value_all = pd.DataFrame(DI_list)
    DI_value_all = DI_value_all.fillna(0)
    DI_value_list = list(DI_value_all[0])
    DI_value_all = pd.DataFrame(DI_value_list)
    DI_value_all.columns = ['DI']
    return DI_value_all

### CI   
def get_CI_value(mat_dense, filter_size, square_size, resolution):
    filter_size = int(filter_size / resolution)
    square_size = int(square_size / resolution)
    CI_list = []
    for i in range(len(mat_dense)):
        if (i <= (filter_size-1)) or (i >= (len(mat_dense)-filter_size)):
            CI_list.append(0)
        else:
            C_value = np.sum(mat_dense[i - square_size:i, i+1:i+square_size+1])
            sum_up = (np.sum(mat_dense[i - square_size:i, i - square_size:i]) - np.sum(np.diag(mat_dense)[i - square_size:i])) / 2
            sum_down = (np.sum(mat_dense[i+1:i+square_size+1, i+1:i+square_size+1]) - np.sum(np.diag(mat_dense)[i+1:i+square_size+1])) / 2
            sum_diag = (np.sum(np.diag(mat_dense)[i - square_size:i]) + np.sum(np.diag(mat_dense)[i+1:i+square_size+1])) /2
            
            I_intra = sum_up + sum_down + sum_diag
            I_inter = C_value
            CI_value = (I_intra - I_inter) / (I_intra + I_inter)
            #CI_value = (sum_up + sum_down + sum_diag) / C_value
            CI_list.append(CI_value)
    CI_value_all = pd.DataFrame(CI_list)
    CI_value_all = CI_value_all.fillna(0)
    CI_value_all.columns = ['CI']
    return CI_value_all
    

resolution = 50000
filter_size_indic = 1000000
square_size_indic = 200000
window_indic = 2500000

indictor_record_all_cell = {}
indictor_list = ['DI', 'IS', 'CI']
for cell_type in cell_type_list:
    print('This is ' + cell_type)
    indictor_record_all_cell[cell_type] = {}
    if cell_type == 'GM12878':
        enzyme_list = ['DpnII', 'MboI']
    else:
        enzyme_list = ['MboI']
    for enzyme in enzyme_list:
        indictor_record_all_cell[cell_type][enzyme] = {}
        mat_cell = hic_mat_all_cell_replicate[cell_type][enzyme]['iced']
        for indic in indictor_list:
            if indic == 'IS':
                IS_file = 'E:/Users/dcdang/TAD_intergate/final_run/cell_type_result' + '/' + 'Rao2014-' + cell_type + '-' + enzyme + '-allreps-filtered-50kb' + '/' + 'chr2_IS.bed'
                IS_value_all = pd.read_csv(IS_file, header = None, sep = '\t')
                IS_value_all.columns = ['chr', 'st', 'ed', 'insulationScore']
                IS_value_all = IS_value_all['insulationScore']
                IS_value_all = pd.DataFrame(IS_value_all.fillna(0))
                IS_value_all.columns = ['IS']
                indic_value = IS_value_all            
            if indic == 'DI':
                DI_value_all = get_DI_value(mat_cell, window_indic, resolution)
                indic_value = DI_value_all
            if indic == 'CI':
                CI_value_all = get_CI_value(mat_cell, filter_size_indic, square_size_indic, resolution)
                indic_value = CI_value_all
            indictor_record_all_cell[cell_type][enzyme][indic] = indic_value


################################ calculate boundary score by boundary voting
            
def get_boundary_score_new(bin_name_Chr, TAD_result, method_list, expand_bin):
    boundary_score_record = {}
    for method in method_list:
        boundary_score_record[method] = {}
        df_domain = TAD_result[method]['TAD_domain']
        method_start_bin_score = np.zeros(len(bin_name_Chr))
        method_end_bin_score = np.zeros(len(bin_name_Chr))       
        bd_st_record = []
        bd_ed_record = []
        for i in range(len(df_domain)):
            boundary_st = df_domain['boundary_st'][i]
            boundary_ed = df_domain['boundary_ed'][i]
            if boundary_st not in bd_st_record:
                st_ind = bin_name_Chr.index(boundary_st)
                method_start_bin_score[st_ind - expand_bin : st_ind + expand_bin + 1] += 1
                bd_st_record.append(boundary_st)
            if boundary_ed not in bd_ed_record:
                ed_ind = bin_name_Chr.index(boundary_ed)
                method_end_bin_score[ed_ind - expand_bin : ed_ind + expand_bin + 1] += 1
                bd_ed_record.append(boundary_ed)
        if np.max(method_start_bin_score) > 1 or np.max(method_end_bin_score) > 1:
            print('Wrong bd score contribute!')
        boundary_score_record[method]['start_bin_score'] = method_start_bin_score
        boundary_score_record[method]['end_bin_score'] = method_end_bin_score
    return boundary_score_record


Chr = 'chr2'
bin_name_Chr = bin_name_list[Chr]
expand_bin = 0
method_score_cell_all = {}
boundary_score_cell_all = {}
for cell_type in cell_type_list:
    print('This is ' + cell_type)
    boundary_score_cell_all[cell_type] = {}
    method_score_cell_all[cell_type] = {}
    if cell_type == 'GM12878':
        enzyme_list = ['DpnII', 'MboI']
    else:
        enzyme_list = ['MboI']
    for enzyme in enzyme_list:
        TAD_result = TAD_result_all_cell_type[cell_type][enzyme]
        boundary_score_record = get_boundary_score_new(bin_name_Chr, TAD_result, method_list, expand_bin)

        df_method_bd_score_st_ed = pd.DataFrame(columns = method_list)
        for method in method_list:
            bd_score_st_ed = []
            bd_score_method_st = boundary_score_record[method]['start_bin_score']
            bd_score_method_ed = boundary_score_record[method]['end_bin_score']
            bd_score_st_ed.append(list(bd_score_method_st))
            bd_score_st_ed.append(list(bd_score_method_ed))
            bd_score_st_ed_max = np.max(bd_score_st_ed, axis = 0)
            df_method_bd_score_st_ed[method] = bd_score_st_ed_max
        method_score_cell_all[cell_type][enzyme] = df_method_bd_score_st_ed   
        bd_score_final = pd.DataFrame(np.sum(df_method_bd_score_st_ed, axis = 1))
        bd_score_final.columns = ['bd_score']
        boundary_score_cell_all[cell_type][enzyme] = bd_score_final
        
        
### loading TAD separation landscape

result_record_all = read_save_data(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape' + '/' + 'TAD_separation_landscape_for_all_cell_type.pkl')

####### Analysis process....
### 1. clustering of cell line with bd score and other indictors
        
def get_correlation_of_score_across_cell(result_record_all, score_type, cor_type = 'pearson'):
    cell_enzyme_l = result_record_all.keys()
    cor_all = []
    for cell1 in cell_enzyme_l:
        print('cell1 is ' + cell1)
        cor_cell1 = []
        bd_score_cell1 = result_record_all[cell1]['BD_region'][score_type]['TAD_score']
        for cell2 in cell_enzyme_l:
            #print('cell2 is ' + cell2)
            bd_score_cell2 = result_record_all[cell2]['BD_region'][score_type]['TAD_score']            
            score_1 = np.array(bd_score_cell1['bd_score'])
            score_2 = np.array(bd_score_cell2['bd_score'])            
            if cor_type == 'pearson':
                cell_1_2_cor = scipy.stats.pearsonr(score_1, score_2)[0]
            elif cor_type == 'spearman':           
                cell_1_2_cor = scipy.stats.spearmanr(score_1, score_2)[0]
            cor_cell1.append(cell_1_2_cor)
        cor_all.append(cor_cell1)
    df_cell_cor = pd.DataFrame(np.array(cor_all))
    df_cell_cor.columns = cell_enzyme_l
    df_cell_cor.index = cell_enzyme_l    
    sns.clustermap(df_cell_cor, cmap="coolwarm", method = 'average', metric='euclidean', figsize=(6, 6))
    return df_cell_cor

def get_correlation_of_indic_across_cell(indictor_record_all_cell, indic_type, cor_type = 'pearson'):
    cor_all = []
    for cell1 in cell_enzyme_l:
        print('cell1 is ' + cell1)
        cor_cell1 = []
        cell_type1 = cell1.split('_')[0]
        enzyme1 = cell1.split('_')[-1]
        bd_indic_cell1 = indictor_record_all_cell[cell_type1][enzyme1][indic_type]     
        for cell2 in tqdm(cell_enzyme_l):
            #print('cell2 is ' + cell2)
            cell_type2 = cell2.split('_')[0]
            enzyme2 = cell2.split('_')[-1]
            bd_indic_cell2 = indictor_record_all_cell[cell_type2][enzyme2][indic_type]     
            
            score_1 = np.array(bd_indic_cell1[indic_type])
            score_2 = np.array(bd_indic_cell2[indic_type])
            if cor_type == 'pearson':
                cell_1_2_cor = scipy.stats.pearsonr(score_1, score_2)[0]
            elif cor_type == 'spearman':           
                cell_1_2_cor = scipy.stats.spearmanr(score_1, score_2)[0]
            cor_cell1.append(cell_1_2_cor)
        cor_all.append(cor_cell1)
    df_cell_cor = pd.DataFrame(np.array(cor_all))
    df_cell_cor.columns = cell_enzyme_l
    df_cell_cor.index = cell_enzyme_l    
    sns.clustermap(df_cell_cor, cmap="coolwarm", method = 'average', figsize=(6, 6))
    return df_cell_cor
    

save_add = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis'

cell_enzyme_l = result_record_all.keys()
data_type_l = ['Mesoderm', 'Mesoderm', 'Ectoderm', 'Mesoderm', 'Endoderm', 'Mesoderm', 'Mesoderm', 'Ectoderm']


score_type = 'Original'
df_cell_cor_bd_score_ori = get_correlation_of_score_across_cell(result_record_all, score_type, cor_type = 'pearson')
df_cell_cor_bd_score_ori['data_type'] = data_type_l
df_cell_cor_bd_score_ori.to_csv(save_add + '/' + 'bd_score_ori_correlation_cell.bed', sep = '\t', header = True, index = True)


score_type = 'Combine'
df_cell_cor_bd_score_ref = get_correlation_of_score_across_cell(result_record_all, score_type, cor_type = 'pearson')
df_cell_cor_bd_score_ref['data_type'] = data_type_l
df_cell_cor_bd_score_ref.to_csv(save_add + '/' + 'bd_score_ref_correlation_cell.bed', sep = '\t', header = True, index = True)


indic_type = 'DI'
df_cell_cor_DI = get_correlation_of_indic_across_cell(indictor_record_all_cell, indic_type, cor_type = 'pearson')
df_cell_cor_DI['data_type'] = data_type_l 
df_cell_cor_DI.to_csv(save_add + '/' + 'DI_correlation_cell.bed', sep = '\t', header = True, index = True)

indic_type = 'IS'
df_cell_cor_IS = get_correlation_of_indic_across_cell(indictor_record_all_cell, indic_type, cor_type = 'pearson')
df_cell_cor_IS['data_type'] = data_type_l 
df_cell_cor_IS.to_csv(save_add + '/' + 'IS_correlation_cell.bed', sep = '\t', header = True, index = True)

indic_type = 'CI'
df_cell_cor_CI = get_correlation_of_indic_across_cell(indictor_record_all_cell, indic_type, cor_type = 'pearson')
df_cell_cor_CI['data_type'] = data_type_l 
df_cell_cor_CI.to_csv(save_add + '/' + 'CI_correlation_cell.bed', sep = '\t', header = True, index = True)

'''
cell_enzyme_l = list(result_record_all.keys())
for cell_enzyme in cell_enzyme_l:
    print('This is ' + cell_enzyme)
    cell = cell_enzyme.split('_')[0]
    enzyme = cell_enzyme.split('_')[1]
    mat_raw = hic_mat_all_cell_replicate[cell][enzyme]['raw']
    print(np.sum(mat_raw))
    print(np.sum(mat_raw == mat_raw.T)==len(mat_raw)**2)
'''


### conservation across cell type and bd score

def draw_mean_bd_score_cell_line_show_number(df_bd_score_cell_cons, save_name = ''):
    median_value = []
    for i in range(8):
        print('This is ' + str(i))
        bd_value_cell_num = df_bd_score_cell_cons[df_bd_score_cell_cons['cons_cell'] == i]
        median_value.append(np.median(bd_value_cell_num['bd_score_mean']))
        print('The number of bins: ' + str(len(bd_value_cell_num)))
        if i != 7:
            bd_value_cell_num_next = df_bd_score_cell_cons[df_bd_score_cell_cons['cons_cell'] == i+1]   
            sta, pvalue = scipy.stats.mannwhitneyu(bd_value_cell_num['bd_score_mean'], bd_value_cell_num_next['bd_score_mean'], alternative = 'less')
            print(pvalue)            
    plt.figure(figsize=(6,5))
    sns.violinplot(x = 'cons_cell', y = 'bd_score_mean', data = df_bd_score_cell_cons, scale="width", cut=0, palette = ['#7F7F7F', '#E377C2', '#8C564B', '#9467BD', '#D62728', '#2CA02C', '#FF7F0E', '#1F77B4'], saturation=1)
    #sns.boxplot(x = 'cons_cell', y = 'bd_score_mean', data = df_bd_score_cell_cons, fliersize=0, palette = ['#7F7F7F', '#E377C2', '#8C564B', '#9467BD', '#D62728', '#2CA02C', '#FF7F0E', '#1F77B4'], saturation=1)                                                                                                                        
    #plt.plot(median_value, c = 'black', linestyle = '--', linewidth=2)                                                                                                                           
    #plt.legend(fontsize = 12)
    plt.xlabel('Occurrence number in multiple cell lines',  FontSize = 12)
    plt.ylabel('Average boundary score across cell lines',  FontSize = 12)
    plt.xticks(list(range(8)), list(range(8)), FontSize = 12)
    plt.yticks(FontSize = 12)
    plt.ylim([-0.7,14.7])
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)   
    plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

    
def draw_mean_bd_score_std_cell_line_show_number(df_bd_score_cell_cons, save_name = ''):
    median_value = [0]
    for i in range(1,8):
        print('This is ' + str(i))
        bd_value_cell_num = df_bd_score_cell_cons[df_bd_score_cell_cons['cons_cell'] == i]
        median_value.append(np.median(bd_value_cell_num['bd_score_std']))
        print('The number of bins: ' + str(len(bd_value_cell_num)))
        if i != 7:
            bd_value_cell_num_next = df_bd_score_cell_cons[df_bd_score_cell_cons['cons_cell'] == i+1]   
            sta, pvalue = scipy.stats.mannwhitneyu(bd_value_cell_num['bd_score_std'], bd_value_cell_num_next['bd_score_std'], alternative = 'less')
            print(pvalue)            
    plt.figure(figsize=(6,5))
    #sns.violinplot(x = 'cons_cell', y = 'bd_score_mean', data = df_bd_score_cell_cons, scale="width", cut=0, palette = ['#7F7F7F', '#E377C2', '#8C564B', '#9467BD', '#D62728', '#2CA02C', '#FF7F0E', '#1F77B4'], saturation=1)
    sns.boxplot(x = 'cons_cell', y = 'bd_score_std', data = df_bd_score_cell_cons, fliersize=0, palette = ['#7F7F7F', '#E377C2', '#8C564B', '#9467BD', '#D62728', '#2CA02C', '#FF7F0E', '#1F77B4'], saturation=1)                                                                                                                        
    #plt.plot(median_value, c = 'black', linestyle = '--', linewidth=2)                                                                                                                           
    #plt.legend(fontsize = 12)
    plt.xlabel('Occurrence number in multiple cell lines',  FontSize = 12)
    plt.ylabel('Standard deviation of boundary score across cell lines',  FontSize = 12)
    plt.xticks(list(range(8)), list(range(8)), FontSize = 12)
    plt.yticks(FontSize = 12)
    #plt.ylim([-0.7,14.7])
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)   
    plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

def check_bd_score_mean_and_phastCons_score(df_bd_score_cell_cons, df_cons_score_chr2, save_name):
    df_bd_score_cell_cons['phastCons_score'] = df_cons_score_chr2[0]
    bd_score_l = np.array(df_bd_score_cell_cons['bd_score_mean'])
    phastCons_score_l = np.array(df_bd_score_cell_cons['phastCons_score'])    
    Pearson_cor = scipy.stats.pearsonr(bd_score_l, phastCons_score_l)[0]    
    plt.figure(figsize=(6,5))
    plt.scatter(bd_score_l, phastCons_score_l,s = 3)                                                                                                                        
    plt.xlabel('Average bounary score across cell lines',  FontSize = 12)
    plt.ylabel('phastCons score',  FontSize = 12)
    plt.xticks(FontSize = 12)
    plt.yticks(FontSize = 12)
    plt.ylim([0,0.4])
    plt.text(7,0.36, 'Pearson correlation = ' + str(np.round(Pearson_cor,2)), fontsize=12)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)   
    plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()


def check_bd_score_mean_and_DNase_phastCons_score(df_bd_score_cell_cons, phconst_score_mean_DNase, save_name):
    df_bd_score_cell_cons['phastCons_score'] = phconst_score_mean_DNase
    bd_score_l = np.array(df_bd_score_cell_cons['bd_score_mean'])
    phastCons_score_l = np.array(df_bd_score_cell_cons['phastCons_score'])    
    Pearson_cor = scipy.stats.pearsonr(bd_score_l, phastCons_score_l)[0]    
    plt.figure(figsize=(6,5))
    plt.scatter(bd_score_l, phastCons_score_l,s = 3)                                                                                                                        
    plt.xlabel('Average bounary score across cell lines',  FontSize = 12)
    plt.ylabel('phastCons score (DNase peak region)',  FontSize = 12)
    plt.xticks(FontSize = 12)
    plt.yticks(FontSize = 12)
    #plt.ylim([0,0.4])
    plt.text(7,0.7, 'Pearson correlation = ' + str(np.round(Pearson_cor,2)), fontsize=12)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)   
    plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
 
def check_bd_score_mean_and_CTCF_motif(df_bd_score_cell_cons, df_cons_score_chr2, save_name):
    df_bd_score_cell_cons['CTCF_motif_num'] = df_cons_score_chr2[4]
    bd_score_l = np.array(df_bd_score_cell_cons['bd_score_mean'])
    phastCons_score_l = np.array(df_bd_score_cell_cons['CTCF_motif_num'])    
    Pearson_cor = scipy.stats.pearsonr(bd_score_l, phastCons_score_l)[0]    
    plt.figure(figsize=(6,5))
    plt.scatter(bd_score_l, phastCons_score_l,s = 3)                                                                                                                        
    plt.xlabel('Average bounary score across cell lines',  FontSize = 12)
    plt.ylabel('CTCF motif number',  FontSize = 12)
    plt.xticks(FontSize = 12)
    plt.yticks(FontSize = 12)
    plt.ylim([-1,80])
    plt.text(6,70, 'Pearson correlation = ' + str(np.round(Pearson_cor,2)), fontsize=12)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)   
    plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()

def check_bd_score_mean_and_CTCF_peak(df_bd_score_cell_cons, df_mean_CTCF, save_name):
    df_bd_score_cell_cons['CTCF_peak_num'] = df_mean_CTCF['mean_CTCF_score']
    bd_score_l = np.array(df_bd_score_cell_cons['bd_score_mean'])
    phastCons_score_l = np.array(df_bd_score_cell_cons['CTCF_peak_num'])    
    Pearson_cor = scipy.stats.pearsonr(bd_score_l, phastCons_score_l)[0]    
    plt.figure(figsize=(6,5))
    plt.scatter(bd_score_l, phastCons_score_l,s = 3)                                                                                                                        
    plt.xlabel('Average bounary score across cell lines',  FontSize = 12)
    plt.ylabel('Average CTCF peak number',  FontSize = 12)
    plt.xticks(FontSize = 12)
    plt.yticks(FontSize = 12)
    #plt.ylim([-1,80])
    plt.text(6,7.5, 'Pearson correlation = ' + str(np.round(Pearson_cor,2)), fontsize=12)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)   
    plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()


def draw_ratio_with_CTCF(df_mean_CTCF, df_bd_score_cell_cons, save_name):
    color_all = ['#BDBDBD', '#B85B4D', '#4392C3']
    ratio_all_l = []
    for i in range(8):
        ratio_l = []
        print('This is ' + str(i))
        df_mean_CTCF_part = df_mean_CTCF[df_bd_score_cell_cons['cons_cell'] == i]
        num_0 = len(df_mean_CTCF_part[df_mean_CTCF_part['mean_CTCF_score'] == 0])
        num_01 = len(df_mean_CTCF_part[(df_mean_CTCF_part['mean_CTCF_score'] > 0) & (df_mean_CTCF_part['mean_CTCF_score'] <= 1)])
        num_1 = len(df_mean_CTCF_part[df_mean_CTCF_part['mean_CTCF_score'] > 1])
        ratio_0 = num_0 / len(df_mean_CTCF_part)
        ratio_01 = num_01 / len(df_mean_CTCF_part)
        ratio_1 = num_1 / len(df_mean_CTCF_part)           
        if (ratio_0 + ratio_01 + ratio_1) == 1:
            print('Check done!')
        ratio_l = [ratio_0, ratio_01, ratio_1]
        ratio_all_l.append(ratio_l)
    df_bin_with_CTCF_ratio = pd.DataFrame(ratio_all_l)
    df_bin_with_CTCF_ratio['type'] = list(range(8))    
    df_bin_with_CTCF_ratio = df_bin_with_CTCF_ratio.sort_values(by = [0,1,2], ascending = False)
    df_bin_with_CTCF_ratio = df_bin_with_CTCF_ratio.reset_index(drop = True)
    df_bin_with_CTCF_ratio[-1] = [0 for i in range(len(df_bin_with_CTCF_ratio))]
    method_ord = list(df_bin_with_CTCF_ratio['type'])
    plt.figure(figsize=(6,5))
    bottom_list = np.zeros(len(method_ord))
    for i in range(len([1,2,3])):
        plt.bar(range(len(method_ord)), list(df_bin_with_CTCF_ratio[i]), align="center", bottom=list(bottom_list), color=color_all[i], edgecolor = 'black', linewidth = 1.5)
        bottom_list += np.array(df_bin_with_CTCF_ratio[i])   
    plt.xticks(list(range(len(method_ord))), method_ord, rotation= 0, FontSize = 12)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'], FontSize = 12)
    plt.ylim([0,1])
    plt.ylabel('Proportion',  FontSize = 12)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)    
    plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)

        
def dnase_peak_phconst_score_fill_with_bin_average(df_cons_score_chr2, df_cons_score_DNase_peak_chr2):
    target_col = list(df_cons_score_DNase_peak_chr2.columns)[4:]
    df_cons_score_DNase_peak_chr2_fill = copy.deepcopy(df_cons_score_DNase_peak_chr2)
    for col in target_col:
        ind = (df_cons_score_DNase_peak_chr2_fill[col] == -1)
        df_cons_score_DNase_peak_chr2_fill[col][ind] = df_cons_score_chr2[0][ind]
    return df_cons_score_DNase_peak_chr2_fill



cell_type_list = ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'K562', 'KBM7', 'NHEK']
enzyme = 'MboI'
score_type = 'Combine'
df_bd_binary_all_cell = pd.DataFrame(columns = cell_type_list)
df_bd_score_all_cell = pd.DataFrame(columns = cell_type_list)
for cell in cell_type_list:
    bd_score_cell_binary = copy.deepcopy(result_record_all[cell+'_'+enzyme]['BD_region'][score_type]['TAD_score'])
    df_bd_score_all_cell[cell] = list(bd_score_cell_binary['bd_score'])
    bd_score_cell_binary[bd_score_cell_binary != 0] = 1
    df_bd_binary_all_cell[cell] = list(bd_score_cell_binary['bd_score'])



### get mean for vector with zero

df_bd_score_cell_cons_with0 = pd.DataFrame(columns = ['cons_cell', 'bd_score_mean', 'bd_score_std'])
cons_cell_l = []
bd_score_mean_l = []
bd_score_std_l = []

for i in range(len(df_bd_binary_all_cell)):
    binary_line = df_bd_binary_all_cell.loc[i]
    score_line = df_bd_score_all_cell.loc[i]
    cons_cell_l.append(np.sum(binary_line))
    bd_score_mean_l.append(np.mean(score_line))
    bd_score_std_l.append(np.std(score_line))
df_bd_score_cell_cons_with0['cons_cell'] = cons_cell_l
df_bd_score_cell_cons_with0['bd_score_mean'] = bd_score_mean_l
df_bd_score_cell_cons_with0['bd_score_std'] = bd_score_std_l
df_bd_score_cell_cons_with0 = df_bd_score_cell_cons_with0.fillna(0)

### get mean for non-zero vector
    
df_bd_score_cell_cons = pd.DataFrame(columns = ['cons_cell', 'bd_score_mean', 'bd_score_std'])
cons_cell_l = []
bd_score_mean_l = []
bd_score_std_l = []

for i in range(len(df_bd_binary_all_cell)):
    binary_line = df_bd_binary_all_cell.loc[i]
    score_line = df_bd_score_all_cell.loc[i]
    binary_line_part = binary_line[binary_line != 0]
    score_line_part = score_line[binary_line != 0]
    cons_cell_l.append(np.sum(binary_line))
    bd_score_mean_l.append(np.mean(score_line_part))
    bd_score_std_l.append(np.std(score_line_part))
df_bd_score_cell_cons['cons_cell'] = cons_cell_l
df_bd_score_cell_cons['bd_score_mean'] = bd_score_mean_l
df_bd_score_cell_cons['bd_score_std'] = bd_score_std_l
df_bd_score_cell_cons = df_bd_score_cell_cons.fillna(0)


### cell show up number and bd socre mean std
save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/cell_number_bd_score_mean_violin.svg'
draw_mean_bd_score_cell_line_show_number(df_bd_score_cell_cons, save_name)

save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/cell_number_bd_score_std.svg'
draw_mean_bd_score_std_cell_line_show_number(df_bd_score_cell_cons, save_name)


save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/cell_number_bd_score_mean_with0_violin.svg'
draw_mean_bd_score_cell_line_show_number(df_bd_score_cell_cons_with0, save_name)

save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/cell_number_bd_score_std_with0.svg'
draw_mean_bd_score_std_cell_line_show_number(df_bd_score_cell_cons_with0, save_name)


### bd score mean and phastCons_score
df_cons_score_chr2 = pd.read_csv('E:/Users/dcdang/share/TAD_integrate/phastCons46way/chr2.phastCons46way.bed', sep = '\t', header = None)
save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/phastCons_score_bd_score_mean.svg'
check_bd_score_mean_and_phastCons_score(df_bd_score_cell_cons, df_cons_score_chr2, save_name)


df_cons_score_DNase_peak_chr2 = pd.read_csv('E:/Users/dcdang/share/TAD_integrate/DNase_peak/DNase_peak_with_phastCon_score/human_chr2_50000_bin_with_DNase_peak_phastCon_score.bed', sep = '\t', header = 0)
df_cons_score_DNase_peak_chr2_fill = dnase_peak_phconst_score_fill_with_bin_average(df_cons_score_chr2, df_cons_score_DNase_peak_chr2)

target_col = list(df_cons_score_DNase_peak_chr2.columns)[4:]
phconst_score_mean_DNase = np.mean(df_cons_score_DNase_peak_chr2_fill[target_col], axis = 1)


def get_dnase_phconst_score_only_peak_region(df_cons_score_DNase_peak_chr2, df_cons_score_chr2):
    phscore_dnase_l = []
    target_col = list(df_cons_score_DNase_peak_chr2.columns)[4:]
    for i in range(len(df_cons_score_DNase_peak_chr2)):
        target_row = df_cons_score_DNase_peak_chr2[target_col].iloc[i]
        if np.sum(target_row==-1) == len(target_col):
            phscore_dnase_l.append(df_cons_score_chr2[0][i])
        else:
            target_row_only_peak = target_row[target_row != -1]
            phscore_dnase_l.append(np.mean(target_row_only_peak))
    return phscore_dnase_l
            
        
phconst_score_mean_DNase_only_peak = get_dnase_phconst_score_only_peak_region(df_cons_score_DNase_peak_chr2, df_cons_score_chr2)

save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/phastCons_score_DNase_bd_score_mean.svg'
check_bd_score_mean_and_DNase_phastCons_score(df_bd_score_cell_cons, phconst_score_mean_DNase, save_name)


### CTCF motif and mean score
df_CTCF_motif_chr2 = pd.read_csv('E:/Users/dcdang/share/TAD_integrate/CTCF_motif_overlap/chrom_with_CTCF.bed', sep = '\t', header = None)
save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/CTCF_motif_bd_score_mean.svg'
check_bd_score_mean_and_CTCF_motif(df_bd_score_cell_cons, df_CTCF_motif_chr2, save_name)

   
### CTCF peak and mean score
df_CTCF_peak_num_bin_chr2 = pd.read_csv('E:/Users/dcdang/share/TAD_integrate/CTCF_peak_overlap_bin/human_chr2_50000_bin_with_CTCF_peak_multiple_cells.bed', sep = '\t', header = 0)

cell_with_CTCF_peak = ['GM12878_CTCF_peak_num', 'K562_CTCF_peak_num',
       'IMR90_CTCF_peak_num', 'HUVEC_CTCF_peak_num', 'HMEC_CTCF_peak_num',
       'NHEK_CTCF_peak_num']

df_CTCF_peak_num_bin_chr2_part = df_CTCF_peak_num_bin_chr2[cell_with_CTCF_peak]

df_mean_CTCF = pd.DataFrame(np.mean(df_CTCF_peak_num_bin_chr2_part, axis = 1))
df_mean_CTCF.columns = ['mean_CTCF_score']


save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/CTCF_peak_num_bd_score_cell_num.svg'
check_bd_score_mean_and_CTCF_peak(df_bd_score_cell_cons, df_mean_CTCF, save_name)

save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/CTCF_peak_bd_score_cell_num.svg'
draw_ratio_with_CTCF(df_mean_CTCF, df_bd_score_cell_cons, save_name)


### CTCF peak phConst score 
df_cons_score_CTCF_peak_chr2 = pd.read_csv('E:/Users/dcdang/share/TAD_integrate/CTCF_peak_overlap_bin/CTCF_phConst_score/human_chr2_50000_bin_with_CTCF_peak_phastCon_score.bed', sep = '\t', header = 0)
df_cons_score_CTCF_peak_chr2_fill = dnase_peak_phconst_score_fill_with_bin_average(df_cons_score_chr2, df_cons_score_CTCF_peak_chr2)

target_col = list(df_cons_score_CTCF_peak_chr2.columns)[4:]
phconst_score_mean_CTCF = np.mean(df_cons_score_CTCF_peak_chr2_fill[target_col], axis = 1)


phconst_score_mean_CTCF_only_peak = get_dnase_phconst_score_only_peak_region(df_cons_score_CTCF_peak_chr2, df_cons_score_chr2)


### Housekeeping gene
df_chrom_region_with_hk_gene = pd.read_csv('E:/Users/dcdang/share/TAD_integrate/housekeeping_gene/human_chr2_50000_bin_hk_genes.bed', sep = '\t', header = None)


### combine all the information and draw pics
def get_percent_pos(score_cut, bd_score):
    for i in range(1, len(score_cut)):
        if bd_score > score_cut[i]:
            continue
        else:
            return i    

def draw_bd_score_level_signal_barplot(x_label, y_label, df_bin_score_CTCF_celln_phConst, y_tick_label, save_name = ''):  
    record_pvalue = {}
    level_l = list(np.unique(df_bin_score_CTCF_celln_phConst['bd_score_mean_percent_label']))
    for l1 in level_l:
        df_data_part1 = df_bin_score_CTCF_celln_phConst[df_bin_score_CTCF_celln_phConst['bd_score_mean_percent_label'] == l1]
        vec_1 = np.array(df_data_part1[y_label])
        for l2 in level_l:
            if l2 <= l1:
                continue
            df_data_part2 = df_bin_score_CTCF_celln_phConst[df_bin_score_CTCF_celln_phConst['bd_score_mean_percent_label'] == l2]
            vec_2 = np.array(df_data_part2[y_label])
            sta, pvalue = scipy.stats.mannwhitneyu(vec_1, vec_2)
            record_pvalue[str(l1) + '_' + str(l2)] = pvalue  
    plt.figure(figsize=(5,5))
    sns.barplot(x = x_label, y = y_label, data = df_bin_score_CTCF_celln_phConst, capsize = 0.15,             
            saturation = 1,             
            errcolor = 'black', errwidth = 2,  
            ci = 95, facecolor=(1, 1, 1, 0), linewidth = 2, edgecolor=['#7F7F7F', '#9467BD', '#D62728', '#2CA02C', '#FF7F0E', '#1F77B4']
                               )
    plt.xticks(rotation= 0, FontSize = 12)
    plt.yticks(FontSize = 12)
    #plt.ylim([0, 0.285])
    #plt.ylim([0.06, 0.16])  
    #plt.ylim([0, 6.9])
    #plt.ylim([0, 1.85])
    plt.ylabel(y_tick_label,  FontSize = 12)
    plt.xlabel('Boundary score level',  FontSize = 12)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=3, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)    
    plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)
    return record_pvalue

df_bin_score_CTCF_celln_phConst = pd.DataFrame()
df_bin_score_CTCF_celln_phConst['Cell_num'] = np.array(df_bd_score_cell_cons['cons_cell'])
df_bin_score_CTCF_celln_phConst['bd_score_mean'] = np.array(df_bd_score_cell_cons['bd_score_mean'])
#df_bin_score_CTCF_celln_phConst['phConst'] = phconst_score_mean_DNase
df_bin_score_CTCF_celln_phConst['phConst'] = phconst_score_mean_DNase_only_peak
df_bin_score_CTCF_celln_phConst['CTCF_peaks'] = df_mean_CTCF['mean_CTCF_score'] 
df_bin_score_CTCF_celln_phConst['CTCF_phConst'] = phconst_score_mean_CTCF_only_peak 
df_bin_score_CTCF_celln_phConst['HK_gene'] = df_chrom_region_with_hk_gene[4]


#sns.clustermap(df_bin_score_CTCF_celln_phConst, cmap = 'coolwarm', standard_scale=1,figsize=(7,7))

#percent_cut = [0, 20, 40, 60, 80, 100]
#percent_cut = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
percent_cut = [0, 30, 44, 58, 72, 86, 100]

score_cut = []
for per_cut in percent_cut:
    score_cut.append(np.percentile(df_bin_score_CTCF_celln_phConst['bd_score_mean'], per_cut))

cut_label = []
for i in range(len(df_bin_score_CTCF_celln_phConst)):
    bd_score = df_bin_score_CTCF_celln_phConst['bd_score_mean'][i]
    per_pos = get_percent_pos(score_cut, bd_score)
    cut_label.append(per_pos)
    
df_bin_score_CTCF_celln_phConst['bd_score_mean_percent_label'] = cut_label


plt.figure(figsize=(5,5))
sns.barplot(x = 'bd_score_mean_percent_label', y = 'Cell_num', data = df_bin_score_CTCF_celln_phConst)

plt.figure(figsize=(5,5))
#sns.violinplot(x = 'bd_score_mean_percent_label', y = 'phConst', data = df_bin_score_CTCF_celln_phConst, cut=0)
sns.boxplot(x = 'bd_score_mean_percent_label', y = 'phConst', data = df_bin_score_CTCF_celln_phConst, fliersize=0)
plt.ylim([-0.05,0.5])

plt.figure(figsize=(5,5))
sns.barplot(x = 'bd_score_mean_percent_label', y = 'CTCF_peaks', data = df_bin_score_CTCF_celln_phConst)

plt.figure(figsize=(5,5))
sns.barplot(x = 'bd_score_mean_percent_label', y = 'CTCF_phConst', data = df_bin_score_CTCF_celln_phConst)

plt.figure(figsize=(5,5))
sns.barplot(x = 'bd_score_mean_percent_label', y = 'HK_gene', data = df_bin_score_CTCF_celln_phConst)

### draw pictures for use
x_label = 'bd_score_mean_percent_label'
y_label = 'HK_gene'
y_tick_label = 'Number of housekeeping genes'  
save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/bd_score_level_HK_gene_num.svg'
draw_bd_score_level_signal_barplot(x_label, y_label, df_bin_score_CTCF_celln_phConst, y_tick_label, save_name = save_name)   
    
         
x_label = 'bd_score_mean_percent_label'
y_label = 'CTCF_phConst'  
y_tick_label = 'phConst score of CTCF binding sites'  
save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/bd_score_level_CTCF_phConst_score.svg'
draw_bd_score_level_signal_barplot(x_label, y_label, df_bin_score_CTCF_celln_phConst, y_tick_label, save_name = save_name)   
    
    
x_label = 'bd_score_mean_percent_label'
y_label = 'Cell_num'  
y_tick_label = 'Number of cell lines with non-zero score'  
save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/bd_score_level_cell_number.svg'
draw_bd_score_level_signal_barplot(x_label, y_label, df_bin_score_CTCF_celln_phConst, y_tick_label, save_name = save_name)   
    
   
x_label = 'bd_score_mean_percent_label'
y_label = 'CTCF_peaks'  
y_tick_label = 'Average number of CTCF peaks across cell lines'  
save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/bd_score_level_CTCF_peaks.svg'
draw_bd_score_level_signal_barplot(x_label, y_label, df_bin_score_CTCF_celln_phConst, y_tick_label, save_name = save_name)   
   

################################# pair-wise cell type comparision

def matrix_part_max_norm(mat_region):
    vec_diag = np.diag(mat_region) 
    mat_diag = np.diag(vec_diag)
    mat_region -= mat_diag
    mat_region = mat_region / np.max(mat_region)
    return mat_region

def draw_pair_wise_map_compare(mat_dense1, mat_dense2, st, ed, bd_cell_score1, bd_cell_score2, resolution, target_site = [], save_name = '', bin_size = 10): 
    x_axis_range = range(len(bd_cell_score1['bd_score'][st:ed]))
    start_ =  st * resolution 
    end_ = ed * resolution
    start = int(start_ / resolution)
    end = int(end_ / resolution)
    cord_list = []
    pos_list = []
    pos_start = start_ 
    x_ticks_l = []
    y_ticks_l = []
    for i in range(ed - st):
        if i % bin_size == 0:
            cord_list.append(i)
            pos = pos_start + i*resolution
            pos_list.append(pos)
            if i + bin_size < ed - st:
                pos_label = str(pos / 1000000)
            else:
                #pos_label = str(pos / 1000000) + '(Mb)'
                pos_label = str(pos / 1000000) 
            x_ticks_l.append(pos_label)
            y_ticks_l.append(str(pos / 1000000))
    region_name = Chr + ':' + str(start_ / 1000000) + '-' + str(end_ / 1000000) + ' Mb'
    
    dense_matrix_part1 = copy.deepcopy(mat_dense1[start:end, start:end])
    dense_matrix_norm1 = matrix_part_max_norm(dense_matrix_part1)
    dense_matrix_part2 = copy.deepcopy(mat_dense2[start:end, start:end])
    dense_matrix_norm2 = matrix_part_max_norm(dense_matrix_part2)
    
    dense_matrix_combine = np.triu(dense_matrix_norm1) + np.tril(-dense_matrix_norm2)
    #vmax = np.percentile(dense_matrix_combine, 90)
    #vmin = -vmax     
    vmax = np.percentile(dense_matrix_norm1, 90)
    vmin = -np.percentile(dense_matrix_norm2, 90)
    norm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

    #plt.figure(figsize=(6,6))
    #plt.imshow(dense_matrix_combine, cmap = 'seismic', vmin = vmin, vmax =vmax, norm=norm) 
    #plt.imshow(dense_matrix_combine, cmap = 'coolwarm', vmin = vmin, vmax =vmax, norm=norm) 
    #plt.colorbar()

    plt.figure(figsize=(6,8))
    ax1 = plt.subplot2grid((9, 7), (0, 0), rowspan=6,colspan=6)
    img = ax1.imshow(dense_matrix_combine, cmap = 'coolwarm', vmin = vmin, vmax =vmax, norm=norm) 
    ax1.set_xticks([])
    #ax1.set_yticks([])
    ax1.spines['bottom'].set_linewidth(0)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(0)
    ax1.spines['top'].set_linewidth(0)
    ax1.tick_params(axis = 'y', length=5, width = 1.6)
    ax1.tick_params(axis = 'x', length=5, width = 1.6)
    plt.xticks(cord_list, x_ticks_l, FontSize = 10)
    plt.yticks(cord_list, y_ticks_l, FontSize = 10)
    ax1.set_title('TAD landscape of region:' + region_name, fontsize=12, pad = 15.0)
    
    cax = plt.subplot2grid((9, 7), (0, 6), rowspan=6,colspan=1)
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes("right", size="1.5%", pad= 0.2)
    #cbar = plt.colorbar(img, cax=cax, ticks=MultipleLocator(2.0), format="%.1f",orientation='vertical',extendfrac='auto',spacing='uniform')
    cbaxes = inset_axes(cax, width="30%", height="70%", loc=5) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])

    ax2 = plt.subplot2grid((9, 7), (6, 0), rowspan=1,colspan=6,sharex=ax1)
    ax2.plot(list(bd_cell_score1['bd_score'][st:ed]), color='black')
    ax2.bar(x_axis_range, list(bd_cell_score1['bd_score'][st:ed]), label='score1', color='#B70D28')
    ax2.spines['bottom'].set_linewidth(1.6)
    ax2.spines['left'].set_linewidth(1.6)
    ax2.spines['right'].set_linewidth(1.6)
    ax2.spines['top'].set_linewidth(1.6)
    ax2.tick_params(axis = 'y', length=5, width = 1.6)
    ax2.tick_params(axis = 'x', length=5, width = 1.6)
    ax2.set_ylabel('BD score', FontSize = 10)

    ax3 = plt.subplot2grid((9, 7), (7, 0), rowspan=1,colspan=6,sharex=ax1)
    ax3.plot(list(bd_cell_score2['bd_score'][st:ed]), color='black')
    ax3.bar(x_axis_range, list(bd_cell_score2['bd_score'][st:ed]), label='score2', color='#3D50C3')
    ax3.spines['bottom'].set_linewidth(1.6)
    ax3.spines['left'].set_linewidth(1.6)
    ax3.spines['right'].set_linewidth(1.6)
    ax3.spines['top'].set_linewidth(1.6)
    ax3.tick_params(axis = 'y', length=5, width = 1.6)
    ax3.tick_params(axis = 'x', length=5, width = 1.6)
    ax3.set_ylabel('BD score', FontSize = 10)
    if len(target_site) != 0:
        site_use = []
        for i in range(len(target_site)):
            if target_site[i] < st:
                pass
            elif target_site[i] <= end:
                site_use.append(target_site[i])
            else:
                break
        plt.vlines(np.array(site_use) - st, 0, 14, linestyles='--')
    if save_name != '':
        plt.savefig(save_name, format = 'svg') 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

def draw_contact_map_and_score_square_2_compare(contact_map, bd_score_show, cell_line, cell_color, save_name = '', type_ = 'single', pos_draw = '', resolution = 50000):
    if len(contact_map) == 41:
        x_range = [0,10,20,30,40]
    elif len(contact_map) == 21:
        x_range = [0,5,10,15,20]
    elif len(contact_map) == 15:
        x_range = [0, 4, 7, 10, 14]
    plt.figure(figsize=(5,5.5))    
    ax1 = plt.subplot2grid((5, 5), (0, 0), rowspan=4,colspan=4)    
    if type_ == 'single':
        img = ax1.imshow(contact_map, cmap='seismic', vmin = np.percentile(contact_map, 10), vmax = np.percentile(contact_map, 85))    
    else:
        vmin = np.percentile(contact_map, 0)
        vmax = np.percentile(contact_map, 100)    
        norm = mcolors.DivergingNorm(vmin=vmin, vcenter = 0, vmax=vmax)
        img = ax1.imshow(contact_map, cmap='seismic', vmin = vmin, vmax = vmax, norm = norm)      
    #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0, vmax = 0.27)    
        #if cell_line in ['HMEC', 'HUVEC','NHEK']:
        #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0, vmax = 0.18)
    #elif cell_line == 'IMR90':
        #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0, vmax = 0.12)
    #else:
        #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0, vmax = 0.25)
    ax1.set_xticks([])
    #ax1.set_yticks([])
    ax1.spines['bottom'].set_linewidth(0)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(0)
    ax1.spines['top'].set_linewidth(0)
    ax1.tick_params(axis = 'y', length=5, width = 1.6)
    ax1.tick_params(axis = 'x', length=5, width = 1.6)
    if pos_draw == '':
        plt.xticks(x_range, x_range, FontSize = 10)
        plt.yticks(x_range, x_range, FontSize = 10)
    else:
        ticks = []
        for x in x_range:
            pos = (pos_draw[0]+x) * resolution / 1000000
            ticks.append(pos)
        plt.xticks(x_range, ticks, FontSize = 10)
        plt.yticks(x_range, ticks, FontSize = 10)
    ax1.set_title(cell_line, fontsize=12, pad = 15.0)
    cax = plt.subplot2grid((5, 5), (0, 4), rowspan=4,colspan=1)
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes("right", size="1.5%", pad= 0.2)
    #cbar = plt.colorbar(img, cax=cax, ticks=MultipleLocator(2.0), format="%.1f",orientation='vertical',extendfrac='auto',spacing='uniform')
    cbaxes = inset_axes(cax, width="30%", height="60%", loc=5) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical', )
    plt.rcParams['font.size'] = 5
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])

    ax5 = plt.subplot2grid((5, 5), (4, 0), rowspan=1,colspan=4,)
    ax5.plot(bd_score_show, color = 'black')
    ax5.fill_between(list(range(len(bd_score_show))), 0, bd_score_show, color = cell_color[cell_line])
    #ax5.bar(list(range(len(bd_score_show))), bd_score_show)
    if np.max(bd_score_show) <= 1 and np.min(bd_score_show) >= -1:
        plt.ylim([-1,1])
    if np.max(bd_score_show) >= 5:
        plt.hlines(5, np.min(x_range), np.max(x_range), linestyles = '--', linewidth=1.6)
    if np.min(bd_score_show) <= -5:
        plt.hlines(-5, np.min(x_range), np.max(x_range), linestyles = '--', linewidth=1.6)
    ax5.spines['bottom'].set_linewidth(1.6)
    ax5.spines['left'].set_linewidth(1.6)
    ax5.spines['right'].set_linewidth(1.6)
    ax5.spines['top'].set_linewidth(1.6)
    ax5.tick_params(axis = 'y', length=5, width = 1.6)
    ax5.tick_params(axis = 'x', length=5, width = 1.6)
    ax5.set_ylabel('Bd score', FontSize = 10)
    plt.xticks(x_range, x_range)
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)

def draw_contact_map_and_score_square_2_compare_plus(contact_map, bd_score_show, CTCF_peak_show, cell_line, cell_color, save_name = '', type_ = 'single', pos_draw = '', resolution = 50000):
    if len(contact_map) == 41:
        x_range = [0,10,20,30,40]
    elif len(contact_map) == 21:
        x_range = [0,5,10,15,20]
    elif len(contact_map) == 15:
        x_range = [0, 4, 7, 10, 14]
    plt.figure(figsize=(5,5.5))    
    ax1 = plt.subplot2grid((6, 5), (0, 0), rowspan=4,colspan=4)    
    if type_ == 'single':
        img = ax1.imshow(contact_map, cmap='seismic', vmin = np.percentile(contact_map, 10), vmax = np.percentile(contact_map, 85))    
    else:
        img = ax1.imshow(contact_map, cmap='seismic')      
    #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0, vmax = 0.27)    
        #if cell_line in ['HMEC', 'HUVEC','NHEK']:
        #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0, vmax = 0.18)
    #elif cell_line == 'IMR90':
        #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0, vmax = 0.12)
    #else:
        #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0, vmax = 0.25)
    ax1.set_xticks([])
    #ax1.set_yticks([])
    ax1.spines['bottom'].set_linewidth(0)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(0)
    ax1.spines['top'].set_linewidth(0)
    ax1.tick_params(axis = 'y', length=5, width = 1.6)
    ax1.tick_params(axis = 'x', length=5, width = 1.6)
    if pos_draw == '':
        plt.xticks(x_range, x_range, FontSize = 10)
        plt.yticks(x_range, x_range, FontSize = 10)
    else:
        ticks = []
        for x in x_range:
            pos = (pos_draw[0]+x) * resolution / 1000000
            ticks.append(pos)
        plt.xticks(x_range, ticks, FontSize = 10)
        plt.yticks(x_range, ticks, FontSize = 10)
    ax1.set_title(cell_line, fontsize=12, pad = 15.0)
    cax = plt.subplot2grid((6, 5), (0, 4), rowspan=4,colspan=1)
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes("right", size="1.5%", pad= 0.2)
    #cbar = plt.colorbar(img, cax=cax, ticks=MultipleLocator(2.0), format="%.1f",orientation='vertical',extendfrac='auto',spacing='uniform')
    cbaxes = inset_axes(cax, width="30%", height="60%", loc=5) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])
    
    ax4 = plt.subplot2grid((6, 5), (4, 0), rowspan=1,colspan=4,)
    ax4.plot(CTCF_peak_show, color = 'black')
    ax4.fill_between(list(range(len(CTCF_peak_show))), 0, CTCF_peak_show, color = cell_color[cell_line])
    #ax4.bar(list(range(len(bd_score_show))), bd_score_show)
    ax4.spines['bottom'].set_linewidth(1.6)
    ax4.spines['left'].set_linewidth(1.6)
    ax4.spines['right'].set_linewidth(1.6)
    ax4.spines['top'].set_linewidth(1.6)
    ax4.tick_params(axis = 'y', length=5, width = 1.6)
    ax4.tick_params(axis = 'x', length=5, width = 1.6)
    ax4.set_ylabel('CTCF peaks', FontSize = 10)
    plt.xticks(x_range, x_range)
    
    ax5 = plt.subplot2grid((6, 5), (5, 0), rowspan=1,colspan=4,)
    ax5.plot(bd_score_show, color = 'black')
    ax5.fill_between(list(range(len(bd_score_show))), 0, bd_score_show, color = cell_color[cell_line])
    #ax5.bar(list(range(len(bd_score_show))), bd_score_show)
    if np.max(bd_score_show) <= 1 and np.min(bd_score_show) >= -1:
        plt.ylim([-1,1])
    if np.max(bd_score_show) >= 5:
        plt.hlines(5, np.min(x_range), np.max(x_range), linestyles = '--', linewidth=1.6)
    if np.min(bd_score_show) <= -5:
        plt.hlines(-5, np.min(x_range), np.max(x_range), linestyles = '--', linewidth=1.6)
    ax5.spines['bottom'].set_linewidth(1.6)
    ax5.spines['left'].set_linewidth(1.6)
    ax5.spines['right'].set_linewidth(1.6)
    ax5.spines['top'].set_linewidth(1.6)
    ax5.tick_params(axis = 'y', length=5, width = 1.6)
    ax5.tick_params(axis = 'x', length=5, width = 1.6)
    ax5.set_ylabel('Bd score', FontSize = 10)
    plt.xticks(x_range, x_range)
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)
    
def deal_with_tad_seperation_score(tad_seperation_score, Chr):
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
    tad_seperation_score_new = copy.deepcopy(tad_seperation_score)
    #tad_seperation_score_new[tad_seperation_score_new['bd_score'] <= 1]=0
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

def get_uniun_bd_region(bd_cell_score1, bd_cell_score2, cell1, cell2):
    df_score_2compare = pd.DataFrame(columns = [cell1, cell2, 'sum'])
    df_score_2compare[cell1] = bd_cell_score1['bd_score']
    df_score_2compare[cell2] = bd_cell_score2['bd_score']
    df_score_2compare['sum'] = bd_cell_score1['bd_score'] + bd_cell_score2['bd_score']   
    adj_index = list(df_score_2compare[df_score_2compare['sum'] < 2].index)
    df_score_2compare['sum'].iloc[adj_index] = 0     
    core_score = copy.deepcopy(df_score_2compare['sum'])
    core_score = pd.DataFrame(core_score)
    core_score.columns = ['bd_score']
    df_bd_core_region_pair = deal_with_tad_seperation_score(core_score, Chr = 'chr2')
    
    cell1_judge_l = []
    cell2_judge_l = []
    score1_l = []
    score2_l = []
    max1_l = []
    max2_l = []
    score_diff_l = []
    for i in range(len(df_bd_core_region_pair)):
        region = df_bd_core_region_pair['region'][i]
        score1 = bd_cell_score1['bd_score'].iloc[region]
        score2 = bd_cell_score2['bd_score'].iloc[region]
        score1_l.append(list(score1))
        score2_l.append(list(score2))
        score_diff = np.max(score1) - np.max(score2)
        score_diff_l.append(score_diff)
        max1_l.append(np.max(score1))
        max2_l.append(np.max(score2))
        
        if np.max(score1) >= 5:
            cell1_judge_l.append(1)
        elif (np.max(score1) > 2) and (np.max(score1) < 5):
            cell1_judge_l.append(0)
        else:
            cell1_judge_l.append(-1)
        if np.max(score2) >= 5:
            cell2_judge_l.append(1)
        elif (np.max(score2) > 2) and (np.max(score2) < 5):
            cell2_judge_l.append(0)
        else:
            cell2_judge_l.append(-1)
    df_bd_core_region_pair = df_bd_core_region_pair[['chr', 'start', 'end', 'length', 'region', 'score']]
    df_bd_core_region_pair['score1_' + cell1] = score1_l
    df_bd_core_region_pair['score2_' + cell2] = score2_l
    df_bd_core_region_pair['max1_' + cell1] = max1_l
    df_bd_core_region_pair['max2_' + cell2] = max2_l
    df_bd_core_region_pair['score_diff'] = score_diff_l
    df_bd_core_region_pair[cell1] = cell1_judge_l
    df_bd_core_region_pair[cell2] = cell2_judge_l
    
    df_bd_region_1_large_2 = copy.deepcopy(df_bd_core_region_pair[(df_bd_core_region_pair[cell1] == 1) & (df_bd_core_region_pair[cell2] == 1) & (df_bd_core_region_pair['score_diff'] >= 3)])
    df_bd_region_1_large_2['index'] = list(df_bd_region_1_large_2.index)
    df_bd_region_1_large_2 = df_bd_region_1_large_2.reset_index(drop = True)
    print('There are ' + str(len(df_bd_region_1_large_2)) + ' stronger bd region from ' + cell1 + ' to ' + cell2 )
    print('Ratio is ' + str(len(df_bd_region_1_large_2) / len(df_bd_core_region_pair)))

    df_bd_region_1_less_2 = copy.deepcopy(df_bd_core_region_pair[(df_bd_core_region_pair[cell1] == 1) & (df_bd_core_region_pair[cell2] == 1) & (df_bd_core_region_pair['score_diff'] <= -3)])
    df_bd_region_1_less_2['index'] = list(df_bd_region_1_less_2.index)
    df_bd_region_1_less_2 = df_bd_region_1_less_2.reset_index(drop = True)
    print('There are ' + str(len(df_bd_region_1_less_2)) + ' weaker bd region from ' + cell1 + ' to ' + cell2 )
    print('Ratio is ' + str(len(df_bd_region_1_less_2) / len(df_bd_core_region_pair)))
   
    #df_bd_region_1_cons_2 = copy.deepcopy(df_bd_core_region_pair[(df_bd_core_region_pair[cell1] == 1) & (df_bd_core_region_pair[cell2] == 1) & (df_bd_core_region_pair['score_diff'] < 3) & (df_bd_core_region_pair['score_diff'] > -3)])
    #df_bd_region_1_cons_2 = copy.deepcopy(df_bd_core_region_pair[(df_bd_core_region_pair[cell1] == 1) & (df_bd_core_region_pair[cell2] == 1) & (df_bd_core_region_pair['score_diff'] > 3)])
    df_bd_region_1_cons_2 = copy.deepcopy(df_bd_core_region_pair[(df_bd_core_region_pair[cell1] == 1) & (df_bd_core_region_pair[cell2] == 1)])
    df_bd_region_1_cons_2['index'] = list(df_bd_region_1_cons_2.index)
    df_bd_region_1_cons_2 = df_bd_region_1_cons_2.reset_index(drop = True)
    print('There are ' + str(len(df_bd_region_1_cons_2)) + ' conserved bd region between ' + cell1 + ' and ' + cell2 )
    print('Ratio is ' + str(len(df_bd_region_1_cons_2) / len(df_bd_core_region_pair)))
    df_bd_region_1_over_2 = copy.deepcopy(df_bd_core_region_pair[(df_bd_core_region_pair[cell1] == 1) & (df_bd_core_region_pair[cell2] == -1) & (df_bd_core_region_pair['score_diff'] >= 5)])
    #df_bd_region_1_over_2 = copy.deepcopy(df_bd_core_region_pair[(df_bd_core_region_pair[cell1] == 1) & (df_bd_core_region_pair[cell2] == -1)])
    df_bd_region_1_over_2['index'] = list(df_bd_region_1_over_2.index)
    df_bd_region_1_over_2 = df_bd_region_1_over_2.reset_index(drop = True)
    print('There are ' + str(len(df_bd_region_1_over_2)) + ' gain bd region from ' + cell1 + ' to ' + cell2 )
    print('Ratio is ' + str(len(df_bd_region_1_over_2) / len(df_bd_core_region_pair)))
    df_bd_region_2_over_1 = copy.deepcopy(df_bd_core_region_pair[(df_bd_core_region_pair[cell1] == -1) & (df_bd_core_region_pair[cell2] == 1)& (df_bd_core_region_pair['score_diff'] <= -5)])
    #df_bd_region_2_over_1 = copy.deepcopy(df_bd_core_region_pair[(df_bd_core_region_pair[cell1] == -1) & (df_bd_core_region_pair[cell2] == 1)])
    df_bd_region_2_over_1['index'] = list(df_bd_region_2_over_1.index)
    df_bd_region_2_over_1 = df_bd_region_2_over_1.reset_index(drop =True)
    print('There are ' + str(len(df_bd_region_2_over_1)) + ' lose bd region from ' + cell1 + ' to ' + cell2 )
    print('Ratio is ' + str(len(df_bd_region_2_over_1) / len(df_bd_core_region_pair)))
    return df_bd_core_region_pair, df_bd_region_1_large_2, df_bd_region_1_less_2, df_bd_region_1_cons_2, df_bd_region_1_over_2, df_bd_region_2_over_1


def get_random_mat(df_random_bd, mat_dense1, mat_dense2, mat_len = 7):
    mat_dense_11 = np.zeros([2*mat_len+1, 2*mat_len+1])
    mat_dense_12 = np.zeros([2*mat_len+1, 2*mat_len+1])
    for i in range(len(df_random_bd)):
        region = df_random_bd['region'][i]
        mid_ind = int((region[0] + region[-1]) / 2)
        if mid_ind < mat_len + 1 or mid_ind > len(mat_dense1) - mat_len -1:
            continue    
        mat_region1 = copy.deepcopy(mat_dense1[mid_ind - mat_len : mid_ind + mat_len+1, mid_ind - mat_len : mid_ind + mat_len+1])        
        mat_region2 = copy.deepcopy(mat_dense2[mid_ind - mat_len : mid_ind + mat_len+1, mid_ind - mat_len : mid_ind + mat_len+1])       
        if np.sum(mat_region1) == 0 or np.sum(mat_region2) == 0:
            continue        
        mat_region1_norm = mat_region1 / np.sum(mat_region1)
        mat_region2_norm = mat_region2 / np.sum(mat_region2)        
        mat_dense_11 += mat_region1_norm
        mat_dense_12 += mat_region2_norm

    mat_dense_11 = mat_dense_11 / np.sum(mat_dense_11)   
    mat_dense_12 = mat_dense_12 / np.sum(mat_dense_12)
    return mat_dense_11, mat_dense_12
    
def get_CI_value_for_region_mat(mat_region1_norm):   
    size = int(len(mat_region1_norm) / 2)
    C_value = np.sum(mat_region1_norm[:size, size+1:])
    sum_up = (np.sum(mat_region1_norm[:size, :size]) - np.sum(np.diag(mat_region1_norm)[:size])) / 2
    sum_down = (np.sum(mat_region1_norm[size+1:, size+1:]) - np.sum(np.diag(mat_region1_norm)[size+1:])) / 2
    sum_diag = (np.sum(np.diag(mat_region1_norm)[:size]) + np.sum(np.diag(mat_region1_norm)[size+1:])) /2    
    I_intra = sum_up + sum_down + sum_diag
    I_inter = C_value
    CI_value = (I_intra - I_inter) / (I_intra + I_inter)
    return CI_value

def get_all_CI_for_chr(mat_dense1, bd_cell_score1, mat_len = 7):
    CI_value_all_chr = []
    bd_score_all_chr = []
    for i in range(len(mat_dense1)):
        if i <= mat_len or i >= len(mat_dense1) - mat_len:
            continue
        else:
            mat_region = mat_dense1[i-mat_len:i+mat_len+1, i-mat_len:i+mat_len+1]
            mat_region_norm = mat_region / np.sum(mat_region)
            CI_value = get_CI_value_for_region_mat(mat_region_norm)
            CI_value_all_chr.append(CI_value)
            bd_score_all_chr.append(bd_cell_score1['bd_score'][i])
    return CI_value_all_chr, bd_score_all_chr

def judge_score(score1, score_bin = [[0], [1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14,15,16]]):
    judge = -1
    for i in range(len(score_bin)):
        score_cut = score_bin[i]
        if score1 >= score_cut[0] and score1 <= score_cut[-1]:
            judge = i
        else:
            continue
    return judge

def compare_CI_group_by_bd_score(df_CI_value_2_cell, score_bin, save_name = ''):   
    df_stats_CI_score = pd.DataFrame(columns = ['CI', 'score', 'cell'])
    CI_l = []
    score_l = []
    cell_l = []
    for i in range(len(df_CI_value_2_cell)):
        score1 = df_CI_value_2_cell['bd_score1'][i]
        score2 = df_CI_value_2_cell['bd_score2'][i]
        CI_1 = df_CI_value_2_cell['CI_1'][i]
        CI_2 = df_CI_value_2_cell['CI_2'][i]
        
        judge1 = judge_score(score1, score_bin)
        judge2 = judge_score(score2, score_bin)
        if judge1 != -1:
            score_l.append(judge1)
            CI_l.append(CI_1)
            cell_l.append('cell1')
        if judge2 != -1:
            score_l.append(judge2)
            CI_l.append(CI_2)
            cell_l.append('cell2')

    df_stats_CI_score['CI'] = CI_l
    df_stats_CI_score['score'] = score_l
    df_stats_CI_score['cell'] = cell_l
    df_stats_CI_score = df_stats_CI_score.fillna(0.5)
    plt.figure(figsize=(10,5))
    sns.violinplot(x = 'score', y = 'CI', hue = 'cell', data = df_stats_CI_score)
    plt.xticks(FontSize = 10)
    plt.yticks(FontSize = 10)
    #plt.ylim([0.3, 0.95])
    plt.ylabel('CI value',  FontSize = 10)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    #plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)  
    plt.legend(loc = 'upper left')
    if save_name != '':    
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)      
    return df_stats_CI_score
        
def max_norm_region_mat(mat_region, mat_random):
    vec_diag = np.diag(mat_region) 
    mat_diag = np.diag(vec_diag)
    vec_diag_u1 = np.diag(mat_region, k=1)
    mat_diag_u1 = np.diag(vec_diag_u1, k=1)
    vec_diag_l1 = np.diag(mat_region, k=-1)
    mat_diag_l1 = np.diag(vec_diag_l1, k=-1)
    #mat_region = mat_region - mat_diag  
    #mat_region = mat_region - mat_diag_u1  
    #mat_region = mat_region - mat_diag_l1  
    mat_region = mat_region / np.sum(mat_region)
    if np.sum(mat_random) != 0:
        mat_region = mat_region / mat_random
    #mat_region_new = np.zeros([len(mat_region), len(mat_region)])
    #for k in range(len(mat_region)):
        #diag_vec = np.diag(mat_region, k=k)
        #diag_vec = diag_vec / np.mean(diag_vec)
        #mat_region_new += np.diag(diag_vec, k = k)
        #if k != 0:
            #mat_region_new += np.diag(diag_vec, k = -k)        
    return mat_region


def aggregate_plot_heatmap(df_bd_region_1_over_2, df_random_bd, mat_dense1, mat_dense2, bd_cell_score1, bd_cell_score2, cell1, cell2, cell_color, descr, mat_len = 10):    
    mat_dense_11 = np.zeros([2*mat_len+1, 2*mat_len+1])
    mat_dense_12 = np.zeros([2*mat_len+1, 2*mat_len+1])
    score_vec_11 = []
    score_vec_12 = []
    CI_value_11 = []
    CI_value_12 = []
    mat_random1, mat_random2 = get_random_mat(df_random_bd, mat_dense1, mat_dense2, mat_len)
    mat_random1 = 0
    mat_random2 = 0
    for i in range(len(df_bd_region_1_over_2)):
        region = df_bd_region_1_over_2['region'][i]
        mid_ind = int((region[0] + region[-1]) / 2)
        if mid_ind < mat_len + 1 or mid_ind > len(mat_dense1) - mat_len -1:
            continue    
        mat_region1 = copy.deepcopy(mat_dense1[mid_ind - mat_len : mid_ind + mat_len+1, mid_ind - mat_len : mid_ind + mat_len+1])        
        mat_region2 = copy.deepcopy(mat_dense2[mid_ind - mat_len : mid_ind + mat_len+1, mid_ind - mat_len : mid_ind + mat_len+1])       
        score_11 = bd_cell_score1['bd_score'].iloc[mid_ind - mat_len : mid_ind + mat_len+1]
        score_12 = bd_cell_score2['bd_score'].iloc[mid_ind - mat_len : mid_ind + mat_len+1]
        if np.sum(mat_region1) == 0 or np.sum(mat_region2) == 0:
            continue        
        score_vec_11.append(score_11)
        score_vec_12.append(score_12)
        mat_region1_norm = max_norm_region_mat(mat_region1, mat_random1)
        mat_region2_norm = max_norm_region_mat(mat_region2, mat_random2)
                       
        CI_value1 = get_CI_value_for_region_mat(mat_region1_norm)
        CI_value2 = get_CI_value_for_region_mat(mat_region2_norm)
        CI_value_11.append(CI_value1)
        CI_value_12.append(CI_value2)
        
        mat_dense_11 += mat_region1_norm
        mat_dense_12 += mat_region2_norm

    mat_dense_11 = mat_dense_11 / np.sum(mat_dense_11)   
    mat_dense_12 = mat_dense_12 / np.sum(mat_dense_12)
    
    score_show_11 = np.array(score_vec_11)
    score_show_11 = np.mean(score_show_11, axis = 0)
    score_show_12 = np.array(score_vec_12)
    score_show_12 = np.mean(score_show_12, axis = 0)
    
    df_CI_value_region = pd.DataFrame(columns = ['CI_1', 'CI_2'])
    df_CI_value_region['CI_1'] = CI_value_11
    df_CI_value_region['CI_2'] = CI_value_12
    df_CI_value_region = df_CI_value_region.fillna(0.5)
    
    #save_name = ''
    save_name = descr + '_1_map.svg'
    draw_contact_map_and_score_square_2_compare(mat_dense_11, score_show_11, cell1, cell_color, save_name, type_ = 'single')
    save_name = descr + '_2_map.svg'
    draw_contact_map_and_score_square_2_compare(mat_dense_12, score_show_12, cell2, cell_color, save_name, type_ = 'single')

    mat_dense_11 = mat_dense_11 + np.diag([1 for j in range(len(mat_dense_11))], k=0) 
    mat_dense_12 = mat_dense_12 + np.diag([1 for j in range(len(mat_dense_12))], k=0)       
    mat_compare12 = mat_dense_11 - mat_dense_12
    score_show_compare = score_show_11 - score_show_12
    save_name = descr + '_1_2_map.svg'
    draw_contact_map_and_score_square_2_compare(mat_compare12, score_show_compare, cell1, cell_color, save_name, type_ = 'compare')
    return df_CI_value_region
    

def aggregate_plot_heatmap_with_CTCF(df_bd_region_1_over_2, mat_dense1, mat_dense2, bd_cell_score1, bd_cell_score2, df_CTCF_peak_num_bin_chr2, cell1, cell2, cell_color, mat_len = 10):    
    mat_dense_11 = np.zeros([2*mat_len+1, 2*mat_len+1])
    mat_dense_12 = np.zeros([2*mat_len+1, 2*mat_len+1])
    score_vec_11 = []
    score_vec_12 = []
    ctcf_vec_11 = []
    ctcf_vec_12 = []
    for i in range(len(df_bd_region_1_over_2)):
        region = df_bd_region_1_over_2['region'][i]
        mid_ind = int((region[0] + region[-1]) / 2)
        if mid_ind < mat_len + 1 or mid_ind > len(mat_dense1) - mat_len -1:
            continue    
        mat_region1 = copy.deepcopy(mat_dense1[mid_ind - mat_len : mid_ind + mat_len+1, mid_ind - mat_len : mid_ind + mat_len+1])        
        mat_region2 = copy.deepcopy(mat_dense2[mid_ind - mat_len : mid_ind + mat_len+1, mid_ind - mat_len : mid_ind + mat_len+1])       
        score_11 = bd_cell_score1['bd_score'].iloc[mid_ind - mat_len : mid_ind + mat_len+1]
        score_12 = bd_cell_score2['bd_score'].iloc[mid_ind - mat_len : mid_ind + mat_len+1]
        ctcf_11 = df_CTCF_peak_num_bin_chr2[cell1 + '_CTCF_peak_num'].iloc[mid_ind - mat_len : mid_ind + mat_len+1]
        ctcf_12 = df_CTCF_peak_num_bin_chr2[cell2 + '_CTCF_peak_num'].iloc[mid_ind - mat_len : mid_ind + mat_len+1]
               
        if np.sum(mat_region1) == 0 or np.sum(mat_region2) == 0:
            continue        
        score_vec_11.append(score_11)
        score_vec_12.append(score_12)
        ctcf_vec_11.append(ctcf_11)
        ctcf_vec_12.append(ctcf_12)

        mat_region1_norm = max_norm_region_mat(mat_region1)
        mat_region2_norm = max_norm_region_mat(mat_region2)
        mat_dense_11 += mat_region1_norm
        mat_dense_12 += mat_region2_norm
    mat_dense_11 = mat_dense_11 / np.sum(mat_dense_11)
    mat_dense_12 = mat_dense_12 / np.sum(mat_dense_12)

    score_show_11 = np.array(score_vec_11)
    score_show_11 = np.mean(score_show_11, axis = 0)
    score_show_12 = np.array(score_vec_12)
    score_show_12 = np.mean(score_show_12, axis = 0)
    
    ctcf_show_11 = np.array(ctcf_vec_11)
    ctcf_show_11 = np.mean(ctcf_show_11, axis = 0)
    ctcf_show_12 = np.array(ctcf_vec_12)
    ctcf_show_12 = np.mean(ctcf_show_12, axis = 0)
    
    draw_contact_map_and_score_square_2_compare_plus(mat_dense_11, score_show_11, ctcf_show_11, cell1, cell_color, save_name='', type_ = 'single')
    draw_contact_map_and_score_square_2_compare_plus(mat_dense_12, score_show_12, ctcf_show_12, cell2, cell_color, save_name='', type_ = 'single')

    mat_dense_11 = mat_dense_11 + np.diag([1 for j in range(len(mat_dense_11))], k=0) 
    mat_dense_12 = mat_dense_12 + np.diag([1 for j in range(len(mat_dense_12))], k=0)       
    mat_compare12 = mat_dense_11 / mat_dense_12
    score_show_compare = score_show_11 - score_show_12
    draw_contact_map_and_score_square_2_compare(mat_compare12, score_show_compare, cell1, cell_color, save_name='', type_ = 'compare')

def check_overlap(region_l, region):
    check = False
    for i in range(len(region_l)):
        if len(set(region_l[i]).intersection(set(region))) != 0:
            check = True
            break
    return check
                  
def get_random_pos_for_compare(df_bd_core_region_pair, mat_dense1, rand_num, Chr = 'chr2'):
    df_random_bd = pd.DataFrame(columns = ['chr', 'start', 'end', 'region'])
    chr_l = []
    st_l = []
    ed_l = []
    region_l = []
    count = 0
    while count < rand_num:
        overlap = True
        pos = random.randint(0, len(mat_dense1)-1)
        length_ind = random.randint(0, len(df_bd_core_region_pair)-1)
        length = df_bd_core_region_pair['length'][length_ind]
        if pos + length >= len(mat_dense1):
            length = 1
        region = list(range(pos, pos + length))
        region_check = check_overlap(region_l, region)
        if region_check == False:
            chr_l.append(Chr)
            st_l.append(pos)
            ed_l.append(pos + length - 1)
            region_l.append(region)
            count += 1
    df_random_bd['chr'] = chr_l
    df_random_bd['start'] = st_l
    df_random_bd['end'] = ed_l
    df_random_bd['region'] = region_l
    df_random_bd = df_random_bd.sort_values(by = ['start'])
    df_random_bd = df_random_bd.reset_index(drop = True)
    return df_random_bd
        

def stats_bd_region_type(df_bd_core_region_pair, df_bd_region_1_large_2, df_bd_region_1_less_2, df_bd_region_1_cons_2, df_bd_region_1_over_2, df_bd_region_2_over_1, type_l = ['All', 'Stronger', 'Weaker', 'Cons', 'Gain', 'Lose']):
    number_l = [] 
    ratio_l = []
    stats_info = {}
    number_l.append(len(df_bd_core_region_pair))
    number_l.append(len(df_bd_region_1_large_2))
    number_l.append(len(df_bd_region_1_less_2))
    number_l.append(len(df_bd_region_1_cons_2))
    number_l.append(len(df_bd_region_1_over_2))
    number_l.append(len(df_bd_region_2_over_1))
    ratio_l.append(len(df_bd_core_region_pair) / len(df_bd_core_region_pair))
    ratio_l.append(len(df_bd_region_1_large_2) / len(df_bd_core_region_pair))
    ratio_l.append(len(df_bd_region_1_less_2) / len(df_bd_core_region_pair))
    ratio_l.append(len(df_bd_region_1_cons_2) / len(df_bd_core_region_pair))
    ratio_l.append(len(df_bd_region_1_over_2) / len(df_bd_core_region_pair))
    ratio_l.append(len(df_bd_region_2_over_1) / len(df_bd_core_region_pair))    
    stats_info['type'] = type_l 
    stats_info['number'] = number_l
    stats_info['ratio'] = ratio_l
    return stats_info


def get_CI_value_combine(CI_pair_record, type_target_l, save_name = ''):
    df_CI_value_region_all = pd.DataFrame(columns = ['CI', 'cell', 'type'])
    CI_l = []
    cell_label_l = []
    type_l = []
    df_random_ci = CI_pair_record['Random']
    random_ci_1 = np.median(df_random_ci['CI_1'])
    random_ci_2 = np.median(df_random_ci['CI_2'])     
    for type_ in list(CI_pair_record.keys()):
        #if type_ not in ['All', 'Random', 'Gain', 'Lose']:
        #if type_ not in ['Cons', 'Gain', 'Lose']:
        if type_ not in type_target_l:            
            continue
        df_CI_record = CI_pair_record[type_]
        #CI_l += list(np.array(df_CI_record['CI_1']) / random_ci_1)
        CI_l += list(np.array(df_CI_record['CI_1']) / 1)
        cell_label_l += ['cell1' for k in range(len(df_CI_record))]
        type_l += [type_ for k in range(len(df_CI_record))]
        #CI_l += list(np.array(df_CI_record['CI_2']) / random_ci_2)
        CI_l += list(np.array(df_CI_record['CI_2']) / 1)
        cell_label_l += ['cell2' for k in range(len(df_CI_record))]
        type_l += [type_ for k in range(len(df_CI_record))]
    df_CI_value_region_all['CI'] = CI_l
    df_CI_value_region_all['type'] = type_l
    df_CI_value_region_all['cell'] = cell_label_l
    plt.figure(figsize=(5,5))
    #sns.boxplot(x = 'type', y = 'CI', hue = 'cell', data = df_CI_value_region_all) 
    sns.violinplot(x = 'type', y = 'CI', hue = 'cell', data = df_CI_value_region_all) 
    #plt.title()
    plt.xticks(FontSize = 10)
    plt.yticks(FontSize = 10)
    #plt.ylim([0.3, 0.95])
    plt.ylabel('CI value',  FontSize = 10)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    #plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)  
    plt.legend(loc = 'upper left')
    if save_name != '':    
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)   
    return df_CI_value_region_all
    
def draw_score_distribution_of_pairwise_cell(df_bd_core_region_pair_part, cell1, cell2, save_name = ''):
    plt.figure(figsize=(6,5))
    sns.distplot(np.array(df_bd_core_region_pair_part['max1_'+cell1]), bins=10, hist = False,  color = 'blue', label = cell1, kde_kws={"lw": 3, 'linestyle':'-', 'shade':True})
    sns.distplot(np.array(df_bd_core_region_pair_part['max2_'+cell2]), bins=10, hist = False,  color = 'red', label = cell2, kde_kws={"lw": 3, 'linestyle':'-', 'shade':True})
    sns.distplot(np.array(df_bd_core_region_pair_part['score_diff']), bins=10, hist = False, label = 'difference', kde_kws={"lw": 3, 'linestyle':'-', 'shade':True})
    plt.xticks(FontSize = 10)
    plt.yticks(FontSize = 10)
    #plt.ylim([0.3, 0.95])
    plt.ylabel('Density',  FontSize = 10)
    plt.ylabel('Boundary score',  FontSize = 10)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    #plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)  
    plt.legend(loc = 'upper right')
    if save_name != '':    
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)   
    
def boostrap_for_CI_pvalue(df_CI_value_2_cell, df_CI_value_gain, method, direc = 'up', boost_n = 10000):    
    median_CI_diff_l = []
    mean_CI_diff_l = []
    for i in range(boost_n):
        if method == 'boostrap':
            index_use = np.random.choice(list(range(len(df_CI_value_2_cell))), size = len(df_CI_value_2_cell), replace=True, p=None)        
        elif method == 'random':
            index_use = np.random.choice(list(range(len(df_CI_value_2_cell))), size = len(df_CI_value_gain), replace=False, p=None)        
        index_use = sorted(index_use)
        df_CI_value_2_cell_boost = df_CI_value_2_cell.iloc[index_use]

        CI_diff = np.array(df_CI_value_2_cell_boost['CI_1'] - df_CI_value_2_cell_boost['CI_2'])
        median_CI_diff_l.append(np.median(CI_diff))
        mean_CI_diff_l.append(np.mean(CI_diff))    
    diff_mean = np.mean(df_CI_value_gain['CI_1'] - df_CI_value_gain['CI_2'])
    diff_median = np.median(df_CI_value_gain['CI_1'] - df_CI_value_gain['CI_2'])    
    #plt.figure(figsize=(6,5))
    #plt.hist(median_CI_diff_l, bins=100)    
    #plt.figure(figsize=(6,5))
    #plt.hist(mean_CI_diff_l, bins=100)    
    if direc == 'up':
        p_median = np.sum(np.array(median_CI_diff_l) >= diff_median) / len(median_CI_diff_l)
        p_mean = np.sum(np.array(mean_CI_diff_l) >= diff_mean) / len(mean_CI_diff_l)
    else: 
        p_median = np.sum(np.array(median_CI_diff_l) <= diff_median) / len(median_CI_diff_l)
        p_mean = np.sum(np.array(mean_CI_diff_l) <= diff_mean) / len(mean_CI_diff_l)
    print('Mean of difference p value: ' + str(p_mean))
    print('Median of difference p value: ' + str(p_median))


def compare_CTCF_peak_numbers(df_bd_region_1_cons_2, df_CTCF_peak_num_bin_chr2_part_norm, cell1, cell2, cell_color, save_name = '', expand_len = 15):   
    df_ctcf_pos = pd.DataFrame(columns = ['ctcf_l', 'pos_l', 'cell_l'])
    ctcf_l = []
    pos_l = []
    cell_l = []
    for i in range(len(df_bd_region_1_cons_2)):
        region = df_bd_region_1_cons_2['region'][i]
        score1 = df_bd_region_1_cons_2['score1_' + cell1][i]
        score2 = df_bd_region_1_cons_2['score2_' + cell2][i]
        mid1 = region[np.argmax(score1)]
        mid2 = region[np.argmax(score2)]
        if mid1 < expand_len + 1 or mid1 > len(df_CTCF_peak_num_bin_chr2_part_norm) - expand_len -1:
            continue
        if mid2 < expand_len + 1 or mid2 > len(df_CTCF_peak_num_bin_chr2_part_norm) - expand_len -1:
            continue
        ctcf_vec1 = df_CTCF_peak_num_bin_chr2_part_norm[cell1 + '_CTCF_peak_num'][mid1 - expand_len : mid1+expand_len+1]
        ctcf_vec2 = df_CTCF_peak_num_bin_chr2_part_norm[cell2 + '_CTCF_peak_num'][mid2 - expand_len : mid2+expand_len+1]
        
        ctcf_l += list(ctcf_vec1)
        pos_l += list(range(len(ctcf_vec1)))
        cell_l += [cell1 for j in range(len(ctcf_vec1)) ]
        
        ctcf_l += list(ctcf_vec2)
        pos_l += list(range(len(ctcf_vec2)))
        cell_l += [cell2 for j in range(len(ctcf_vec2)) ]
    df_ctcf_pos['ctcf_l'] = ctcf_l
    df_ctcf_pos['pos_l'] = pos_l
    df_ctcf_pos['cell_l'] = cell_l
        
    plt.figure(figsize=(6,5))
    sns.lineplot(x = 'pos_l', y = 'ctcf_l', data = df_ctcf_pos, hue = 'cell_l',  style="cell_l", markers=True, linewidth = 3)
    plt.xticks([0, 3, 7, 11, 14], ['-350kb', '-200kb', 'boundary center', '200kb', '350kb'], FontSize = 10)
    plt.yticks(FontSize = 10)
    #plt.ylim([0.3, 0.95])
    plt.ylabel('CTCF peaks fold-change',  fontSize = 12)
    plt.xlabel('',  fontSize = 0)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    #plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)  
    plt.legend(loc = 'best', prop = {'size':12}, fancybox = None, edgecolor = 'white', facecolor = None, title = None, title_fontsize = 0)
    if save_name != '':    
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)   


CTCF_sum = np.sum(df_CTCF_peak_num_bin_chr2_part, axis = 0)
df_CTCF_peak_num_bin_chr2_part_norm = copy.deepcopy(df_CTCF_peak_num_bin_chr2_part)
for i in range(len(df_CTCF_peak_num_bin_chr2_part.columns)):
    col_name = list(df_CTCF_peak_num_bin_chr2_part.columns)[i]
    df_CTCF_peak_num_bin_chr2_part_norm[col_name] = df_CTCF_peak_num_bin_chr2_part_norm[col_name] / (CTCF_sum[i] / len(df_CTCF_peak_num_bin_chr2_part_norm)) 


pair_compare_result = {}
for cell1 in cell_type_list:
    print('Cell 1 is :' + cell1)
    cell_ind = cell_type_list.index(cell1)
    pair_compare_result[cell1] = {}
    for cell2 in cell_type_list[cell_ind+1:]:
        bd_pair_record = {}
        CI_pair_record = {}
        print('Cell 2 is :' + cell2)
        pair_compare_result[cell1][cell2] = {}
        mat_dense1 = hic_mat_all_cell_replicate[cell1]['MboI']['iced']  
        mat_dense2 = hic_mat_all_cell_replicate[cell2]['MboI']['iced']   
        
        bd_cell_score1 = result_record_all[cell1 + '_MboI']['BD_region']['Combine']['TAD_score'] 
        bd_cell_score2 = result_record_all[cell2 + '_MboI']['BD_region']['Combine']['TAD_score']
        df_score_2compare = pd.DataFrame(columns = [cell1, cell2])
        df_score_2compare[cell1] = bd_cell_score1['bd_score']
        df_score_2compare[cell2] = bd_cell_score2['bd_score']
        
        # get bd region class
        df_bd_core_region_pair, df_bd_region_1_large_2, df_bd_region_1_less_2, df_bd_region_1_cons_2, df_bd_region_1_over_2, df_bd_region_2_over_1 = get_uniun_bd_region(bd_cell_score1, bd_cell_score2, cell1, cell2)
        #df_bd_core_region_pair, df_bd_region_1_large_2, df_bd_region_1_less_2, df_bd_region_1_cons_2 = get_uniun_bd_region_test(bd_cell_score1, bd_cell_score2, cell1, cell2)
        bd_pair_record['All'] = df_bd_core_region_pair
        bd_pair_record['Stronger'] = df_bd_region_1_large_2
        bd_pair_record['Weaker'] = df_bd_region_1_less_2
        bd_pair_record['Cons'] = df_bd_region_1_cons_2
        bd_pair_record['Gain'] = df_bd_region_1_over_2
        bd_pair_record['Lose'] = df_bd_region_2_over_1
   
        # get random bd region
        rand_num = 300
        df_random_bd = get_random_pos_for_compare(df_bd_core_region_pair, mat_dense1, rand_num, Chr = 'chr2')
        df_random_bd2 = get_random_pos_for_compare(df_bd_core_region_pair, mat_dense1, rand_num, Chr = 'chr2')
        
        # stats bd region info
        stats_info = stats_bd_region_type(df_bd_core_region_pair, df_bd_region_1_large_2, df_bd_region_1_less_2, df_bd_region_1_cons_2, df_bd_region_1_over_2, df_bd_region_2_over_1, type_l = ['All', 'Stronger', 'Weaker', 'Cons', 'Gain', 'Lose'])        
        pair_compare_result[cell1][cell2]['Stats_info'] = stats_info        
        pair_compare_result[cell1][cell2]['Bd_region'] = bd_pair_record
        
        # make up dir
        pair_com_add = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\cell_type_analysis\pairwise_compare'
        pair_save_add = pair_com_add + '/' + cell1 + '/' + cell2
        if not os.path.exists(pair_save_add):
            os.makedirs(pair_save_add)
         
        # compare CI value group by bd_score for pairwise cell compare
        df_CI_value_2_cell = pd.DataFrame(columns = ['bd_score1', 'bd_score2','CI_1', 'CI_2'])
        CI_value_cell1, bd_score_all_chr1 = get_all_CI_for_chr(mat_dense1, bd_cell_score1, mat_len = 7)
        CI_value_cell2, bd_score_all_chr2 = get_all_CI_for_chr(mat_dense2, bd_cell_score2, mat_len = 7)
        df_CI_value_2_cell['CI_1'] = CI_value_cell1
        df_CI_value_2_cell['CI_2'] = CI_value_cell2
        df_CI_value_2_cell = df_CI_value_2_cell.fillna(0.5)
        df_CI_value_2_cell['bd_score1'] = bd_score_all_chr1
        df_CI_value_2_cell['bd_score2'] = bd_score_all_chr2                
        score_bin = []
        for i in range(15):
            score_bin.append([i])
        #score_bin = [[0], [1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14,15,16]]
        save_name = pair_save_add + '/' + cell1 + '_' + cell2 + '_CI_compare_group_by_score.svg'
        df_stats_CI_score = compare_CI_group_by_bd_score(df_CI_value_2_cell, score_bin, save_name)

        # get CI value, draw heatmap for each group and save fig
        descr = pair_save_add + '/' + cell1 + '_' + cell2 + '_random_bd'
        df_CI_value_random = aggregate_plot_heatmap(df_random_bd, df_random_bd2, mat_dense1, mat_dense2, bd_cell_score1, bd_cell_score2, cell1, cell2, cell_color, descr, mat_len = 7)
        descr = pair_save_add + '/' + cell1 + '_' + cell2 + '_cons_bd'
        df_CI_value_cons = aggregate_plot_heatmap(df_bd_region_1_cons_2, df_random_bd2, mat_dense1, mat_dense2, bd_cell_score1, bd_cell_score2, cell1, cell2, cell_color, descr, mat_len = 7)
        descr = pair_save_add + '/' + cell1 + '_' + cell2 + '_stronger_bd'
        df_CI_value_stronger = aggregate_plot_heatmap(df_bd_region_1_large_2, df_random_bd2, mat_dense1, mat_dense2, bd_cell_score1, bd_cell_score2, cell1, cell2, cell_color, descr, mat_len = 7)
        descr = pair_save_add + '/' + cell1 + '_' + cell2 + '_weaker_bd'
        df_CI_value_weaker = aggregate_plot_heatmap(df_bd_region_1_less_2, df_random_bd2, mat_dense1, mat_dense2, bd_cell_score1, bd_cell_score2, cell1, cell2, cell_color, descr, mat_len = 7)
        descr = pair_save_add + '/' + cell1 + '_' + cell2 + '_gain_bd'
        df_CI_value_gain = aggregate_plot_heatmap(df_bd_region_1_over_2, df_random_bd2, mat_dense1, mat_dense2, bd_cell_score1, bd_cell_score2, cell1, cell2, cell_color, descr, mat_len = 7)
        descr = pair_save_add + '/' + cell1 + '_' + cell2 + '_lose_bd'
        df_CI_value_lose = aggregate_plot_heatmap(df_bd_region_2_over_1, df_random_bd2, mat_dense1, mat_dense2, bd_cell_score1, bd_cell_score2, cell1, cell2, cell_color, descr, mat_len = 7)       
        #aggregate_plot_heatmap_with_CTCF(df_bd_region_1_over_2, mat_dense1, mat_dense2, bd_cell_score1, bd_cell_score2, df_CTCF_peak_num_bin_chr2, cell1, cell2, cell_color, mat_len = 10)
        
        # compare score distribution of certain type bd region
        gain_score_save_name = pair_save_add + '/' + cell1 + '_' + cell2 + '_gain_score_distribution_compare.svg'
        draw_score_distribution_of_pairwise_cell(df_bd_region_1_over_2, cell1, cell2, gain_score_save_name)
        lose_score_save_name = pair_save_add + '/' + cell1 + '_' + cell2 + '_lose_score_distribution_compare.svg'
        draw_score_distribution_of_pairwise_cell(df_bd_region_2_over_1, cell1, cell2, lose_score_save_name)
        #cons_score_save_name = ''
        #draw_score_distribution_of_pairwise_cell(df_bd_region_1_cons_2, cell1, cell2, cons_score_save_name)

        # compare CI value for each group
        CI_pair_record['All'] = df_CI_value_2_cell
        CI_pair_record['Random'] = df_CI_value_random
        CI_pair_record['Stronger'] = df_CI_value_stronger
        CI_pair_record['Weaker'] = df_CI_value_weaker
        CI_pair_record['Cons'] = df_CI_value_cons
        CI_pair_record['Gain'] = df_CI_value_gain
        CI_pair_record['Lose'] = df_CI_value_lose
        CI_save_name = pair_save_add + '/' + cell1 + '_' + cell2 + '_CI_compare_for_specific_type.svg'
        type_target_l = ['Gain', 'Lose']
        df_CI_value_region_all = get_CI_value_combine(CI_pair_record, type_target_l, CI_save_name)

        # boostrap for CI pvalue
        print('Get p value for some specific bd region compare.......')
        print('For gain bd:')
        #boostrap_for_CI_pvalue(df_CI_value_2_cell, df_CI_value_gain, method = 'boostrap', direc = 'up', boost_n = 10000)
        print('For lose bd:')
        #boostrap_for_CI_pvalue(df_CI_value_2_cell, df_CI_value_lose, method = 'boostrap', direc = 'down', boost_n = 10000)

        # random for pvalue:
        #print('For gain bd:')
        #boostrap_for_CI_pvalue(df_CI_value_2_cell, df_CI_value_gain, method = 'random', direc = 'up', boost_n = 10000)
        #print('For lose bd:')
        #boostrap_for_CI_pvalue(df_CI_value_2_cell, df_CI_value_lose, method = 'random', direc = 'down', boost_n = 10000)

        # check CTCF peak number
                
        if cell1 == 'KBM7' or cell2 == 'KBM7':
            print('No CTCF information....')
        else:
            save_name = pair_save_add + '/' + cell1 + '_' + cell2 + '_cons_CTCF_compare.svg'
            compare_CTCF_peak_numbers(df_bd_region_1_cons_2, df_CTCF_peak_num_bin_chr2_part_norm, cell1, cell2, cell_color, save_name, expand_len = 7)
            save_name = pair_save_add + '/' + cell1 + '_' + cell2 + '_gain_CTCF_compare.svg'
            compare_CTCF_peak_numbers(df_bd_region_1_over_2, df_CTCF_peak_num_bin_chr2_part_norm, cell1, cell2, cell_color, save_name, expand_len = 7)
            save_name = pair_save_add + '/' + cell1 + '_' + cell2 + '_lose_CTCF_compare.svg'
            compare_CTCF_peak_numbers(df_bd_region_2_over_1, df_CTCF_peak_num_bin_chr2_part_norm, cell1, cell2, cell_color, save_name, expand_len = 7)


#### multiple cell type boundary comparison

###### conserved and cell specific boundary region find

##  test for one matrix

####### You will need cv2. If you do not have it, run: pip install opencv-python
import cv2
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid


def resize(im__,scale_percent = 100):
    width = int(im__.shape[1] * scale_percent / 100)
    height = int(im__.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(im__, dim, interpolation = cv2.INTER_NEAREST)
    return resized

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),cv2.INTER_NEAREST)


def interp1dnan(A):
    A_=np.array(A)
    ok = np.isnan(A)==False
    xp = ok.nonzero()[0]
    fp = A[ok]
    x  = np.isnan(A).nonzero()[0]
    A_[np.isnan(A)] = np.interp(x, xp, fp)
    return A_


def interpolate_chr(_chr):
    """linear interpolate chromosome coordinates"""
    _new_chr = np.array(_chr)
    for i in range(_new_chr.shape[-1]):
        _new_chr[:,i]=interp1dnan(_new_chr[:,i])
    return _new_chr


contact_map = hic_mat_all_cell_replicate['GM12878']['MboI']['iced'][200:301,200:301]

mat_ = contact_map
pad=0
# the minimum and maximum distance in nanometers. this sets the threshold of the image
min_val = 0
max_val = None   
if max_val is None: 
    max_val = np.nanmax(mat_)
if min_val is None: 
    min_val = np.nanmin(mat_)
    
#This colors the image
im_ = (np.clip(mat_, min_val, max_val) - min_val) / (max_val - min_val)
im__ = np.array(cm.seismic(im_)[:,:,:3]*255,dtype=np.uint8)
for j in [-1,0,1]:
    diag = np.diagonal(im__[:,:,2], offset=j)
    mat_diag = np.diag(diag, k = j)
    mat_fill = np.diag(np.array([np.max(im__[:,:,2]) for i in range(len(diag))]), k=j)
    im__[:,:,2] = im__[:,:,2] - mat_diag + mat_fill

# resize image 10x to get good resolution
resc = 10
resized = resize(im__, resc*100)

# Rotate 45 degs
resized = rotate_bound(resized,-45)
start = int(pad*np.sqrt(2)*resc)
center = int(resized.shape[1]/2)

#Clip it to the desired size
padup=25 ##### how much of the matrix to keep in the up direction
resized = resized[center-resc*padup:center]
#resized = resized[center-resc*padup:center+resc*padup]


cts = np.array([100 for i in range(len(mat_))])
start = 0
min__ = 0
cts_perc = 1.*cts/len(cts)*100*resc
x_vals = (np.arange(len(cts_perc))-min__)*resc*np.sqrt(2)-start

# create axes
fig = plt.figure(figsize=(7,7))

grid = ImageGrid(fig, 111, nrows_ncols=(2, 1),axes_pad=0.)
contact_ax = grid[0]
contact_ax.imshow(resized[:,:,2], cmap='seismic')

grid[1].plot(x_vals,cts_perc,'ko-')

grid[0].set_yticks([])

#contact_ax.set_xlim([min_*resc*np.sqrt(2),max_*resc*np.sqrt(2)])


'''
plt.figure(figsize=(7,5))
#plt.imshow(np.array(resized), cmap='seismic', vmin = np.percentile(np.array(resized), 5), vmax = np.percentile(np.array(resized), 95))
plt.imshow(resized[:,:,2], cmap='coolwarm')

plt.figure(figsize=(7,5))
#plt.imshow(np.array(resized), cmap='seismic', vmin = np.percentile(np.array(resized), 5), vmax = np.percentile(np.array(resized), 95))
plt.imshow(resized, cmap='seismic')
'''


''' plot triangle contact map

contact_map = hic_mat_all_cell_replicate['GM12878']['MboI']['iced'][200:301,200:301]

dst=ndimage.rotate(contact_map, 45, order=0, reshape=True, prefilter=False, cval=0)


start=0
length = len(dst)
height = length / 6
plt.figure(figsize=(7,3))
#plt.imshow(dst, cmap='coolwarm', vmin = np.percentile(np.unique(dst), 5), vmax = np.percentile(np.unique(dst), 95))
plt.imshow(dst, cmap='coolwarm', vmin = np.percentile(np.array(dst), 5), vmax = np.percentile(np.array(dst), 95))
plt.ylim([start+length/2,start+length/2+height])


start=0
length = len(dst)
height = length / 6
plt.figure(figsize=(7,3))
with np.errstate(divide='ignore'): 
    plt.imshow(dst, origin = 'upper', cmap='coolwarm', interpolation="nearest", extent=(int(start or 1) - 0.5,\
                    int(start or 1) + length - 0.5, int(start or 1) - 0.5,int(start or 1) + length - 0.5), \
            vmin = np.percentile(np.array(dst), 5), vmax = np.percentile(np.array(dst), 95))
plt.ylim([start+length/2,start+length/2+height])
		


plt.figure(figsize=(7,7))
plt.imshow(contact_map, cmap='coolwarm', vmin = np.percentile(contact_map, 5), vmax = np.percentile(contact_map, 95))

'''

## plot for multi-cell line and boundary score


def plot_triangular_contact_map_boundary_score_single_cell_line(st, ed, cell_line, enzyme, cell_color, hic_mat_all_cell_replicate, result_record_all):
    
    mat_dense = copy.deepcopy(hic_mat_all_cell_replicate[cell_line][enzyme]['iced'][st:ed, st:ed])
    bd_score_show = copy.deepcopy(result_record_all[cell_line + '_' + enzyme]['BD_region']['Combine']['TAD_score']['bd_score'].iloc[st:ed])
 
    mat_ = mat_dense
    pad=0
    # the minimum and maximum distance in nanometers. this sets the threshold of the image
    min_val = 0
    max_val = None   
    if max_val is None: 
        max_val = np.nanmax(mat_)
    if min_val is None: 
        min_val = np.nanmin(mat_)
        
    #This colors the image
    im_ = (np.clip(mat_, min_val, max_val) - min_val) / (max_val - min_val)
    im__ = np.array(cm.seismic(im_)[:,:,:3]*255,dtype=np.uint8)
    for j in [-1,0,1]:
        diag = np.diagonal(im__[:,:,2], offset=j)
        mat_diag = np.diag(diag, k = j)
        mat_fill = np.diag(np.array([np.max(im__[:,:,2]) for i in range(len(diag))]), k=j)
        im__[:,:,2] = im__[:,:,2] - mat_diag + mat_fill
    
    # resize image 10x to get good resolution
    resc = 10
    resized = resize(im__, resc*100)
    
    # Rotate 45 degs
    resized = rotate_bound(resized,-45)
    start = int(pad*np.sqrt(2)*resc)
    center = int(resized.shape[1]/2)
    
    #Clip it to the desired size
    padup=25 ##### how much of the matrix to keep in the up direction
    resized = resized[center-resc*padup:center]
    #resized = resized[center-resc*padup:center+resc*padup]
       
    cts = np.array(bd_score_show)
    start = 0
    min__ = 0
    cts_perc = 1.*cts/len(cts)*100*resc
    x_vals = (np.arange(len(cts_perc))-min__)*resc*np.sqrt(2)-start
    
    # create axes
    fig = plt.figure(figsize=(10,10))
    
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0)
    contact_ax = grid[0]
    contact_ax.imshow(resized[:,:,2], cmap='seismic')
    
    grid[1].plot(x_vals, cts_perc, color='black')
    #grid[1].bar(x_vals, cts_perc, width=2, color = cell_color[cell_line])
    grid[1].fill_between(x_vals, 0, cts_perc, color = cell_color[cell_line])
    
    grid[0].set_yticks([])
    grid[0].set_ylabel(cell_line, fontdict = {'size': 3, 'rotation':90})

    plt.yticks([0,50], [0,5], FontSize = 5)
    grid[1].set_ylabel('Boundary score', fontdict = {'size': 3, 'rotation':0})
    
    



st = 800
ed = 950
cell_line = 'GM12878'
enzyme = 'MboI'
plot_triangular_contact_map_boundary_score_single_cell_line(st, ed, cell_line, enzyme, cell_color, hic_mat_all_cell_replicate, result_record_all)






def plot_triangular_contact_map_boundary_score_multiple_cell_line(st, ed, cell_type_list, enzyme, cell_color, hic_mat_all_cell_replicate, result_record_all):

    fig = plt.figure(figsize=(10,10))    
    for i in range(len(cell_type_list)):
        cell_line = cell_type_list[i]
        print('This is ' + cell_line)
        mat_dense = copy.deepcopy(hic_mat_all_cell_replicate[cell_line][enzyme]['iced'][st:ed, st:ed])
        bd_score_show = copy.deepcopy(result_record_all[cell_line + '_' + enzyme]['BD_region']['Combine']['TAD_score']['bd_score'].iloc[st:ed])
           
        mat_ = mat_dense
        pad=0
        # the minimum and maximum distance in nanometers. this sets the threshold of the image
        min_val = 0
        max_val = None   
        if max_val is None: 
            max_val = np.nanmax(mat_)
        if min_val is None: 
            min_val = np.nanmin(mat_)
            
        #This colors the image
        im_ = (np.clip(mat_, min_val, max_val) - min_val) / (max_val - min_val)
        im__ = np.array(cm.seismic(im_)[:,:,:3]*255,dtype=np.uint8)
        for j in [-1,0,1]:
            diag = np.diagonal(im__[:,:,2], offset=j)
            mat_diag = np.diag(diag, k = j)
            mat_fill = np.diag(np.array([np.max(im__[:,:,2]) for i in range(len(diag))]), k=j)
            im__[:,:,2] = im__[:,:,2] - mat_diag + mat_fill
        
        # resize image 10x to get good resolution
        resc = 10
        resized = resize(im__, resc*100)
        
        # Rotate 45 degs
        resized = rotate_bound(resized,-45)
        start = int(pad*np.sqrt(2)*resc)
        center = int(resized.shape[1]/2)
        
        #Clip it to the desired size
        padup=25 ##### how much of the matrix to keep in the up direction
        resized = resized[center-resc*padup:center]
        #resized = resized[center-resc*padup:center+resc*padup]
           
        cts = np.array(bd_score_show)
        start = 0
        min__ = 0
        cts_perc = 1.*cts/len(cts)*100*resc
        fold = np.max(cts_perc) / np.max(cts)
        print(fold)
        x_vals = (np.arange(len(cts_perc))-min__)*resc*np.sqrt(2)-start
    
        pos = '71' + str(i+1)
        pos = int(pos)
        #grid = ImageGrid(fig, pos, nrows_ncols=(2, 1), axes_pad=0, share_all=True, add_all = True)
        grid = ImageGrid(fig, pos, nrows_ncols=(2, 1), axes_pad=0, share_all = False, cbar_mode= None, cbar_location='right', cbar_pad=0.5, cbar_set_cax = False)
        contact_ax = grid[0]
        cax = grid.cbar_axes[0]
        #cbaxes = inset_axes(cax, width="3%", height="100%", loc = 5) 
        data = resized[:,:,2] / np.max(resized[:,:,2])
        im = contact_ax.imshow(data, cmap='seismic')
        #cb = plt.colorbar(im, cax = cbaxes)
        #tick_locator = ticker.MaxNLocator(nbins=5)
        #cb.locator = tick_locator
        #cb.set_ticks([np.min(data), 0.25,0.5, 0.75, np.max(data)])
        #cb.update_ticks()
        
        grid[1].plot(x_vals, cts_perc, color='black')
        #grid[1].bar(x_vals, cts_perc, width=2, color = cell_color[cell_line])
        grid[1].fill_between(x_vals, 0, cts_perc, color = cell_color[cell_line])
        #grid[1].set_xticks([0, 50*np.sqrt(2)*resc, 100*np.sqrt(2)*resc, 150*np.sqrt(2)*resc, 200*np.sqrt(2)*resc], [0, 50, 100, 150, 200])
        grid[0].set_yticks([])
        grid[0].set_ylabel(cell_line, fontdict = {'size': 5, 'rotation':90})
    
        #grid[1].set_ylabel('Boundary score', fontdict = {'size': 0, 'rotation':0})

  
st = 800
ed = 950

#st = 0
#ed = 80

plot_triangular_contact_map_boundary_score_multiple_cell_line(st, ed, cell_type_list, enzyme, cell_color, hic_mat_all_cell_replicate, result_record_all)

plot_triangular_contact_map_boundary_score_multiple_cell_line(st, ed, ['K562', 'NHEK'], enzyme, cell_color, hic_mat_all_cell_replicate, result_record_all)



#### one case find between K562 and NHEK
st = 780
ed = 860

st = 0
ed = 80
site_use = []
site_target = ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'K562', 'KBM7', 'NHEK']
for cell_line in cell_type_list:
    #cell_line = 'K562'
    enzyme = 'MboI'
    contact_map = copy.deepcopy(hic_mat_all_cell_replicate[cell_line][enzyme]['iced'][st:ed,st:ed])
    contact_map = contact_map / np.max(contact_map)
    plt.figure(figsize=(7,7))
    plt.imshow(contact_map, cmap='coolwarm', vmin = np.percentile(contact_map, 5), vmax = np.percentile(contact_map, 95))
    #plt.imshow(contact_map, cmap='coolwarm', vmin = 0, vmax = 0.25)
    if cell_line in site_target:
        if len(site_use) != 0:
            for j in range(len(site_use)):
                plt.vlines(site_use[j] - st, 0, site_use[j]-st, linestyle = '--')
    plt.colorbar(shrink=0.8)

########### find cell-type conserved and specific

cell_type_list = ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'K562', 'KBM7', 'NHEK']
enzyme = 'MboI'
score_type = 'Combine'
df_bd_binary_all_cell = pd.DataFrame(columns = cell_type_list)
df_bd_score_all_cell = pd.DataFrame(columns = cell_type_list)
for cell in cell_type_list:
    bd_score_cell_binary = copy.deepcopy(result_record_all[cell+'_'+enzyme]['BD_region'][score_type]['TAD_score'])
    df_bd_score_all_cell[cell] = list(bd_score_cell_binary['bd_score'])
    bd_score_cell_binary[bd_score_cell_binary != 0] = 1
    df_bd_binary_all_cell[cell] = list(bd_score_cell_binary['bd_score'])





def get_core_boundary_region(df_bd_binary_all_cell, df_bd_score_all_cell, cell_type_list):
    df_bd_binary_all_cell['Sum_score'] = np.sum(df_bd_score_all_cell, axis = 1)
    df_bd_binary_all_cell_adj = copy.deepcopy(df_bd_binary_all_cell)
    adj_index = list(df_bd_binary_all_cell_adj[df_bd_binary_all_cell['Sum_score'] < 5].index)
    df_bd_binary_all_cell_adj['Sum_score'].iloc[adj_index] = 0
    core_score = copy.deepcopy(df_bd_binary_all_cell_adj['Sum_score'])
    core_score = pd.DataFrame(core_score)
    core_score.columns = ['bd_score']
    df_boundary_core_region = deal_with_tad_seperation_score(core_score, Chr = 'chr2')
    df_boundary_core_region_judge = copy.deepcopy(df_boundary_core_region[['chr', 'start', 'end', 'length', 'region', 'score']])
    for cell in cell_type_list:
        cell_judge = []
        for i in range(len(df_boundary_core_region)):
            region_core = df_boundary_core_region['region'][i]
            #region_score = df_boundary_core_region['score'][i]
            score_region_cell = df_bd_score_all_cell[cell].iloc[region_core]
            if np.max(score_region_cell) >= 5:
                cell_judge.append(1)
            else:
                cell_judge.append(0)
        df_boundary_core_region_judge[cell] = cell_judge

    df_part = df_boundary_core_region_judge[cell_type_list]        
    bd_por = list(np.sum(df_part, axis = 1))
    df_boundary_core_region_judge['sum_cell'] = bd_por
    type_l = []
    num_l = []
    for type_ in list(np.unique(bd_por)):
        type_l.append(type_)
        num_l.append(bd_por.count(type_))
    print(type_l)
    print(num_l)
    return df_boundary_core_region_judge


df_boundary_core_region_judge = get_core_boundary_region(df_bd_binary_all_cell, df_bd_score_all_cell, cell_type_list)


df_boundary_core_region_judge[cell_type_list].to_csv(r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\cell_type_analysis\core_boundary\Core_boundary_cell_score.bed', sep = '\t', header = True, index = None)





df_boundary_core_region_judge_no0 = df_boundary_core_region_judge[df_boundary_core_region_judge['sum_cell'] != 0]
df_boundary_core_region_judge_no0 = df_boundary_core_region_judge_no0.reset_index(drop = True)

index_l = []
type_l = []
for r in range(1,8):
    for i in combinations(range(7), r):
        df_part = copy.deepcopy(df_boundary_core_region_judge_no0)
        i_ = []
        for n in range(7):
            if n not in i:
                i_.append(n)
        for x in i:
            cell = cell_type_list[x]
            df_part = df_part[df_part[cell] == 1]
        for x in i_:
            cell = cell_type_list[x]
            df_part = df_part[df_part[cell] == 0]
        index_l += list(df_part.index)
        type_l += [i for n in range(len(df_part))]
        print(i, len(df_part))
                
df_index_type = pd.DataFrame(columns = ['index', 'type'])
df_index_type['index'] = index_l
df_index_type['type'] = type_l

df_index_type = df_index_type.sort_values(by = ['index'])

 
df_boundary_core_region_judge_no0['type'] = list(df_index_type['type']) 



def get_bd_score_of_core_boundary_region(df_boundary_core_region_judge, df_bd_score_all_cell, cell_type_list):
    score_all = []
    sum_cell_l = []
    rank_value_l = []
    for i in range(len(df_boundary_core_region_judge)):
        cell_num = df_boundary_core_region_judge['sum_cell'][i]
        region = df_boundary_core_region_judge['region'][i]
        core_score_cell = []
        sum_cell_l.append(cell_num)    
        cons_v = []
        spe_v = []
        for cell in cell_type_list:
            max_score = np.max(df_bd_score_all_cell[cell].iloc[region])
            core_score_cell.append(max_score)
            if df_boundary_core_region_judge[cell][i] == 0:
                spe_v.append(max_score)
            else:
                cons_v.append(max_score)
        if len(cons_v) == 0 and len(spe_v) != 0:
            rank_value_l.append(0)
            score_all.append(core_score_cell)
            continue
        if len(cons_v) != 7:
            rank_value = np.mean(cons_v) - np.mean(spe_v)
        else:
            rank_value = np.mean(cons_v) / (np.std(cons_v) + 0.0001)
        rank_value_l.append(rank_value)    
        score_all.append(core_score_cell)
    #score_all = np.log2(np.array(score_all)+1)
    df_bd_score_core_region_mat = pd.DataFrame(score_all)
    df_bd_score_core_region_mat.columns = cell_type_list
    df_bd_score_core_region_mat['sum_score'] = np.sum(df_bd_score_core_region_mat[cell_type_list], axis = 1)
    df_bd_score_core_region_mat['sum_cell'] = sum_cell_l 
    df_bd_score_core_region_mat['rank_value'] = rank_value_l

    DF_record = []
    for cell_num in [7,1]:
        if cell_num == 7:
            df_bd_score_core_region_mat_part = df_bd_score_core_region_mat[df_bd_score_core_region_mat['sum_cell'] == cell_num]
            df_bd_score_core_region_mat_part = df_bd_score_core_region_mat_part.sort_values(by = 'rank_value', ascending=False)
            DF_record.append(df_bd_score_core_region_mat_part)
        elif cell_num == 6:
            DF_record_6 = []
            for cell in cell_type_list:
                cell_index = cell_type_list.index(cell)
                target = []
                for m in range(len(cell_type_list)):
                    if m != cell_index:
                        target.append(m)
                target = tuple(target)
                df_bd_score_core_region_mat_part = df_bd_score_core_region_mat[df_boundary_core_region_judge['type'] == target]
                df_bd_score_core_region_mat_part = df_bd_score_core_region_mat_part.sort_values(by = 'rank_value', ascending=False)
                DF_record_6.append(df_bd_score_core_region_mat_part)
                
            df_record_6 = DF_record_6[0]
            for i in range(1, len(DF_record_6)):
                df_record_6 = np.concatenate([df_record_6, DF_record_6[i]], axis = 0)
            DF_record.append(df_record_6)
            
        elif cell_num == 1:
            DF_record_0 = []
            for cell in cell_type_list:
                df_bd_score_core_region_mat_part = df_bd_score_core_region_mat[(df_bd_score_core_region_mat['sum_cell'] == cell_num) & (df_boundary_core_region_judge[cell]) == 1]
                df_bd_score_core_region_mat_part = df_bd_score_core_region_mat_part.sort_values(by = 'rank_value', ascending=False)
                DF_record_0.append(df_bd_score_core_region_mat_part)
            df_record_0 = DF_record_0[0]
            for i in range(1, len(DF_record_0)):
                df_record_0 = np.concatenate([df_record_0, DF_record_0[i]], axis = 0)
            DF_record.append(df_record_0)
            
    df_record = DF_record[0]
    for i in range(1,len(DF_record)):
        df_record = np.concatenate([df_record, DF_record[i]], axis = 0)
    df_record = pd.DataFrame(df_record)
    df_record.columns = list(df_bd_score_core_region_mat.columns)
      
    plt.figure(figsize=(6,6))
    df_record_draw = np.log2(np.array(df_record[cell_type_list]) + 1)
    df_record_draw = pd.DataFrame(df_record_draw)
    df_record_draw.columns = cell_type_list
    sns.heatmap(df_record_draw, cmap = "GnBu", xticklabels=1, yticklabels=False)

   
get_bd_score_of_core_boundary_region(df_boundary_core_region_judge_no0, df_bd_score_all_cell, cell_type_list)



def get_mat_and_bd_score_specific_region_with_pos(df_boundary_core_region_judge_no0, df_bd_score_all_cell, hic_mat_all_cell_replicate, cell_type_list, region_tp, mat_len = 10, norm = False):
    
    mat_region_cell = {}
    score_region_cell = {}
    pos_region = {}
    for cell in cell_type_list:
        mat_zero = np.zeros([2*mat_len+1, 2*mat_len+1])
        #mat_region_cell[cell] = mat_zero
        mat_region_cell[cell] = []
        score_region_cell[cell] = []
        pos_region[cell] = []
        
    df_boundary_core_region_judge_part = df_boundary_core_region_judge_no0[df_boundary_core_region_judge_no0['type'] == region_tp]
    df_boundary_core_region_judge_part = df_boundary_core_region_judge_part.reset_index(drop = True)

    for i in range(len(df_boundary_core_region_judge_part)):
        if i % 50 == 0 and i != 0:
            print('50 regions done!')
        region = df_boundary_core_region_judge_part['region'][i]
        mid_ind = int((region[0] + region[-1]) / 2)
        if mid_ind < mat_len or mid_ind > len(df_bd_score_all_cell) - mat_len:
            continue
        for cell in cell_type_list:
            mat_target = mat_region_cell[cell]
            if norm == False:
                mat_cell = copy.deepcopy(hic_mat_all_cell_replicate[cell]['MboI']['iced'])
            else:
                mat_cell = copy.deepcopy(hic_mat_all_cell_replicate[cell]['MboI'])
            mat_region = mat_cell[mid_ind - mat_len : mid_ind + mat_len+1, mid_ind - mat_len : mid_ind + mat_len+1]        
            vec_diag = np.diag(mat_region) 
            mat_diag = np.diag(vec_diag)
            mat_region -= mat_diag
            if norm == False:
                mat_region = mat_region / np.max(mat_region)
            #mat_target += mat_region
            mat_target.append(mat_region)
            mat_region_cell[cell] = mat_target
    
            score_target = score_region_cell[cell]        
            score_region = copy.deepcopy(df_bd_score_all_cell[cell].iloc[mid_ind - mat_len : mid_ind + mat_len+1])
            score_target.append(list(score_region))
            score_region_cell[cell] = score_target
            
            pos_target = pos_region[cell]
            pos = (mid_ind - mat_len, mid_ind + mat_len)
            pos_target.append(pos)
            pos_region[cell] = pos_target
    return mat_region_cell, score_region_cell, pos_region



def plot_triangular_contact_map_boundary_score_compare_cell_line(cell_line, mat_dense, bd_score_show, cell_type_list, cell_color, save_name = ''):

    fig = plt.figure(figsize=(5,2))    
    #mat_dense = copy.deepcopy(mat_region_cell[cell_line])
    #bd_score_show_cell = copy.deepcopy(np.array(score_region_cell[cell_line]))
    #bd_score_show = np.mean(bd_score_show_cell, axis = 0)
    
    mat_ = mat_dense
    pad=0
    # the minimum and maximum distance in nanometers. this sets the threshold of the image
    min_val = 0
    max_val = None   
    if max_val is None: 
        max_val = np.nanmax(mat_)
    if min_val is None: 
        min_val = np.nanmin(mat_)
        
    #This colors the image
    im_ = (np.clip(mat_, min_val, max_val) - min_val) / (max_val - min_val)
    im__ = np.array(cm.seismic(im_)[:,:,:3]*255,dtype=np.uint8)
    for j in [-1,0,1]:
        diag = np.diagonal(im__[:,:,2], offset=j)
        mat_diag = np.diag(diag, k = j)
        mat_fill = np.diag(np.array([np.max(im__[:,:,2]) for i in range(len(diag))]), k=j)
        im__[:,:,2] = im__[:,:,2] - mat_diag + mat_fill
    
    # resize image 10x to get good resolution
    resc = 10
    resized = resize(im__, resc*100)
    
    # Rotate 45 degs
    resized = rotate_bound(resized,-45)
    start = int(pad*np.sqrt(2)*resc)
    center = int(resized.shape[1]/2)
    
    #Clip it to the desired size
    padup=25 ##### how much of the matrix to keep in the up direction
    resized = resized[center-resc*padup:center]
    #resized = resized[center-resc*padup:center+resc*padup]
       
    cts = np.array(bd_score_show)
    start = 0
    min__ = 0
    cts_perc = 1.*cts/len(cts)*100*resc
    fold = np.max(cts_perc) / np.max(cts)
    print(fold)
    x_vals = (np.arange(len(cts_perc))-min__)*resc*np.sqrt(2)-start

    pos = '111'
    pos = int(pos)
    #grid = ImageGrid(fig, pos, nrows_ncols=(2, 1), axes_pad=0, share_all=True, add_all = True)
    grid = ImageGrid(fig, pos, nrows_ncols=(2, 1), axes_pad=0, share_all = False, cbar_mode= None, cbar_location='right', cbar_pad=0.5, cbar_set_cax = False)
    contact_ax = grid[0]
    #cax = grid.cbar_axes[0]
    #cbaxes = inset_axes(cax, width="3%", height="100%", loc = 5) 
    data = resized[:,:,2] / np.max(resized[:,:,2])
    im = contact_ax.imshow(data, cmap='seismic')
    #cb = plt.colorbar(im, cax = cbaxes)
    #tick_locator = ticker.MaxNLocator(nbins=5)
    #cb.locator = tick_locator
    #cb.set_ticks([np.min(data), 0.25,0.5, 0.75, np.max(data)])
    #cb.update_ticks()
    
    grid[1].plot(x_vals, cts_perc, color='black')
    #grid[1].bar(x_vals, cts_perc, width=2, color = cell_color[cell_line])
    grid[1].fill_between(x_vals, 0, cts_perc, color = cell_color[cell_line])
    
    grid[0].set_yticks([])
    grid[0].set_ylabel(cell_line, fontdict = {'size': 5, 'rotation':90})
    #grid[1].set_ylabel('Boundary score', fontdict = {'size': 0, 'rotation':0})
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)
   

def draw_contact_map_and_score_square(contact_map, bd_score_show, cell_line, cell_color, save_name = '', pos_draw = '', resolution = 50000):
    if len(contact_map) == 41:
        x_range = [0,10,20,30,40]
    elif len(contact_map == 21):
        x_range = [0,5,10,15,20]
    plt.figure(figsize=(5,5.5))    
    ax1 = plt.subplot2grid((5, 5), (0, 0), rowspan=4,colspan=4)
    img = ax1.imshow(contact_map, cmap='seismic', vmin = np.percentile(contact_map, 10), vmax = np.percentile(contact_map, 85))    
    #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0.1, vmax = 0.5)    
    #img = ax1.imshow(contact_map, cmap='seismic')    
    #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0, vmax = 0.27)    
        #if cell_line in ['HMEC', 'HUVEC','NHEK']:
        #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0, vmax = 0.18)
    #elif cell_line == 'IMR90':
        #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0, vmax = 0.12)
    #else:
        #img = ax1.imshow(contact_map, cmap='seismic', vmin = 0, vmax = 0.25)
    ax1.set_xticks([])
    #ax1.set_yticks([])
    ax1.spines['bottom'].set_linewidth(0)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(0)
    ax1.spines['top'].set_linewidth(0)
    ax1.tick_params(axis = 'y', length=5, width = 1.6)
    ax1.tick_params(axis = 'x', length=5, width = 1.6)
    if pos_draw == '':
        plt.xticks(x_range, x_range, FontSize = 10)
        plt.yticks(x_range, x_range, FontSize = 10)
    else:
        ticks = []
        for x in x_range:
            pos = (pos_draw[0]+x) * resolution / 1000000
            ticks.append(pos)
        plt.xticks(x_range, ticks, FontSize = 10)
        plt.yticks(x_range, ticks, FontSize = 10)
    ax1.set_title(cell_line, fontsize=12, pad = 15.0)
    cax = plt.subplot2grid((5, 5), (0, 4), rowspan=4,colspan=1)
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes("right", size="1.5%", pad= 0.2)
    #cbar = plt.colorbar(img, cax=cax, ticks=MultipleLocator(2.0), format="%.1f",orientation='vertical',extendfrac='auto',spacing='uniform')
    cbaxes = inset_axes(cax, width="30%", height="60%", loc=5) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])

    ax5 = plt.subplot2grid((5, 5), (4, 0), rowspan=1,colspan=4,)
    ax5.plot(bd_score_show, color = 'black')
    ax5.fill_between(list(range(len(bd_score_show))), 0, bd_score_show, color = cell_color[cell_line])
    #ax5.bar(list(range(len(bd_score_show))), bd_score_show)
    ax5.spines['bottom'].set_linewidth(1.6)
    ax5.spines['left'].set_linewidth(1.6)
    ax5.spines['right'].set_linewidth(1.6)
    ax5.spines['top'].set_linewidth(1.6)
    ax5.tick_params(axis = 'y', length=5, width = 1.6)
    ax5.tick_params(axis = 'x', length=5, width = 1.6)
    ax5.set_ylabel('Bd score', FontSize = 10)
    score_top = np.max([np.max(bd_score_show), 5])
    plt.ylim([0, score_top])
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

def get_CTCF_peaks_core_region(df_boundary_core_region_judge_no0, cell_type_list, region_tp, mat_len = 20):    
   ctcf_region_cell = {}
   for cell in cell_type_list:
       #if cell == 'KBM7':
           #continue
       ctcf_region_cell[cell] = []   
   df_boundary_core_region_judge_part = df_boundary_core_region_judge_no0[df_boundary_core_region_judge_no0['type'] == region_tp]
   df_boundary_core_region_judge_part = df_boundary_core_region_judge_part.reset_index(drop = True)
   for i in range(len(df_boundary_core_region_judge_part)):
       if i % 50 == 0 and i != 0:
           print('50 regions done!')
       region = df_boundary_core_region_judge_part['region'][i]
       mid_ind = int((region[0] + region[-1]) / 2)
       if mid_ind < mat_len or mid_ind > len(df_bd_score_all_cell) - mat_len:
           continue
       for cell in cell_type_list:
           if cell == 'KBM7':
               continue
           peak_target = ctcf_region_cell[cell]
           peak_region = copy.deepcopy(df_CTCF_peak_num_bin_chr2[cell + '_CTCF_peak_num'].iloc[mid_ind - mat_len : mid_ind + mat_len+1])
           peak_target.append(list(peak_region))
           ctcf_region_cell[cell] = peak_target 
   return ctcf_region_cell 
    


def get_CTCF_peaks_mean_region(df_boundary_core_region_judge_no0, cell_type_list, region_tp):
   ctcf_region_cell = {}
   for cell in cell_type_list:
       #if cell == 'KBM7':
           #continue
       ctcf_region_cell[cell] = []   
   df_boundary_core_region_judge_part = df_boundary_core_region_judge_no0[df_boundary_core_region_judge_no0['type'] == region_tp]
   df_boundary_core_region_judge_part = df_boundary_core_region_judge_part.reset_index(drop = True)
   for i in range(len(df_boundary_core_region_judge_part)):
       if i % 50 == 0 and i != 0:
           print('50 regions done!')
       region = list(df_boundary_core_region_judge_part['region'][i])
       for cell in cell_type_list:
           if cell == 'KBM7':
               continue
           peak_target = ctcf_region_cell[cell]
           peak_region = copy.deepcopy(df_CTCF_peak_num_bin_chr2[cell + '_CTCF_peak_num'].iloc[region])
           peak_target.append(np.mean(peak_region))
           ctcf_region_cell[cell] = peak_target 
   return ctcf_region_cell 


def draw_core_region_CTCF_peaks(ctcf_region_cell, cell_type_list, cell_color, save_name = ''):
    
    plt.figure(figsize=(5,4.5))
    for cell in cell_type_list:
        if cell == 'KBM7':
            continue
        ctcf_peaks = np.mean(ctcf_region_cell[cell], axis = 0)
        plt.plot(list(range(len(ctcf_peaks))), ctcf_peaks, c = cell_color[cell], linewidth = 2)    
    #plt.title()
    plt.xticks([0, 10, 20, 30, 40], ['-1Mb', '-500kb', 'region center', '500kb', '1Mb'], FontSize = 10)
    #plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'], FontSize = 12)
    plt.yticks(FontSize = 8)
    #plt.ylim([0,1])
    plt.ylabel('CTCF peaks',  FontSize = 10)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)
    #plt.legend()
    if save_name != '':    
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

       
def draw_core_region_CTCF_mean_peaks(ctcf_region_cell, cell_type_list, cell_color, save_name = ''):    
    df_region_mean_ctcf = pd.DataFrame(columns = ['mean_peaks', 'cell_line'])
    peak_l = []
    cell_l = []
    color = []
    for cell in cell_type_list:
        color.append(cell_color[cell])
        peak_mean = ctcf_region_cell[cell]
        peak_l += list(peak_mean)
        cell_l += [cell for i in range(len(peak_mean))]
    df_region_mean_ctcf['mean_peaks'] = peak_l    
    df_region_mean_ctcf['cell_line'] = cell_l        
    plt.figure(figsize=(5,5))
    sns.boxplot(x = 'cell_line', y = 'mean_peaks', data = df_region_mean_ctcf, fliersize=1, palette = color, saturation=1)       
    #plt.title()
    plt.xticks(FontSize = 10)
    plt.yticks(FontSize = 8)
    #plt.ylim([0,1])
    plt.ylabel('CTCF peaks',  FontSize = 10)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)  
    #plt.legend()
    if save_name != '':    
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)
    
    
region_tp = (0,1,2,3,4,5,6)
mat_region_cell, score_region_cell, pos_region = get_mat_and_bd_score_specific_region_with_pos(df_boundary_core_region_judge_no0, df_bd_score_all_cell, hic_mat_all_cell_replicate, cell_type_list, region_tp, mat_len = 10, norm = False)
#mat_region_cell, score_region_cell, pos_region = get_mat_and_bd_score_specific_region_with_pos(df_boundary_core_region_judge_no0, df_bd_score_all_cell, hic_mat_all_cell_replicate_znorm, cell_type_list, region_tp, mat_len = 10, norm = True)



#### cons bd region

mat_region_cell_cons = copy.deepcopy(mat_region_cell)
score_region_cell_cons = copy.deepcopy(score_region_cell)
pos_region_cons = copy.deepcopy(pos_region)


for cell_line in cell_type_list:
    mat_dense = mat_region_cell_cons[cell_line][0]
    for i in range(1, len(mat_region_cell_cons[cell_line])):
        mat_dense += mat_region_cell_cons[cell_line][i]
    bd_score_show_cell = copy.deepcopy(np.array(score_region_cell_cons[cell_line]))
    bd_score_show = np.mean(bd_score_show_cell, axis = 0) 

    contact_map = mat_dense / np.max(mat_dense)
    save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/core_boundary/cell_cons/square/' + cell_line + '_MboI_cons_contact_and_score.svg'
    #save_name = ''
    draw_contact_map_and_score_square(contact_map, bd_score_show, cell_line, cell_color, save_name)
    
    #save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/core_boundary/cell_cons/trangular/' + cell_line + '_MboI_cons_contact_and_score.svg'
    #plot_triangular_contact_map_boundary_score_compare_cell_line(cell_line, mat_dense, bd_score_show, cell_type_list, cell_color, save_name)



### specific bd region

region_tp = (5,)
mat_region_cell, score_region_cell, pos_region = get_mat_and_bd_score_specific_region_with_pos(df_boundary_core_region_judge_no0, df_bd_score_all_cell, hic_mat_all_cell_replicate, cell_type_list, region_tp, mat_len = 10, norm = False)
    
spe_add = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/core_boundary/cell_spe'
spe_add_cell = spe_add + '/' + cell_type_list[5] 
    
    
for cell_line in cell_type_list:
    mat_dense = mat_region_cell[cell_line][0]
    for i in range(1, len(mat_region_cell[cell_line])):
        mat_dense += mat_region_cell[cell_line][i]
    bd_score_show_cell = copy.deepcopy(np.array(score_region_cell[cell_line]))
    bd_score_show = np.mean(bd_score_show_cell, axis = 0) 
  
    contact_map = mat_dense / np.max(mat_dense)
    type_ = 'square'
    if not os.path.exists(spe_add_cell + '/' + type_):
        os.makedirs(spe_add_cell + '/' + type_)

    save_name = spe_add_cell + '/' + type_ + '/' + cell_line + '_MboI_spe_contact_and_score.svg'
    #save_name = ''
    draw_contact_map_and_score_square(contact_map, bd_score_show, cell_line, cell_color, save_name)
       
    #type_ = 'trangular'
    #if not os.path.exists(spe_add_cell + '/' + type_):
        #os.makedirs(spe_add_cell + '/' + type_)
    #save_name = spe_add_cell + '/' + type_ + '/' + cell_line + '_MboI_spe_contact_and_score.svg'
    #plot_triangular_contact_map_boundary_score_compare_cell_line(cell_line, mat_dense, bd_score_show, cell_type_list, cell_color, save_name)





### case plot

region_tp = (1,2,3,4,6)
mat_region_cell, score_region_cell, pos_region = get_mat_and_bd_score_specific_region_with_pos(df_boundary_core_region_judge_no0, df_bd_score_all_cell, hic_mat_all_cell_replicate, cell_type_list, region_tp, mat_len = 20, norm = False)
#mat_region_cell, score_region_cell, pos_region = get_mat_and_bd_score_specific_region_with_pos(df_boundary_core_region_judge_no0, df_bd_score_all_cell, hic_mat_all_cell_replicate, cell_type_list, region_tp, mat_len = 20, norm = True)


for cell_line in cell_type_list:
    mat_dense_all = copy.deepcopy(mat_region_cell[cell_line])
    bd_score_show_all = copy.deepcopy(score_region_cell[cell_line])
    pos_region_all = copy.deepcopy(pos_region[cell_line])
    index = 0
    contact_map = mat_dense_all[index]
    bd_score_show = bd_score_show_all[index]   
    pos = pos_region_all[index]
    #save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/core_boundary/some case/all/' + cell_line + '_' + str(index) + '.svg'
    save_name = ''
    draw_contact_map_and_score_square(contact_map, bd_score_show, cell_line, cell_color, save_name, pos_draw = pos)  
    
    #plot_triangular_contact_map_boundary_score_compare_cell_line(cell_line, mat_dense, bd_score_show, cell_type_list, cell_color)

'''    
# case record, index for core boundary region, draw heatmap and check     
'K562' : 1, 2, 9 
(4,) 

'IMR90': 0, 1
(3,)  

'HMEC','NHEK' : 0, 5
(1,6)

'HUVEC', 'IMR90': 0
(2,3)

'K562', 'KBM7': 0
(4,5)
      
'IMR90','HMEC','NHEK' : 0
(1,3,6)

'HMEC', 'HUVEC', 'IMR90','NHEK' : 0, 5
(1,2,3,6) 


'GM12878', 'HUVEC', 'K562', 'KBM7', 'NHEK'
(0,2,4,5,6) 

'HUVEC less': 10
(0,1,3,4,5,6) 

'all':
(0,1,2,3,4,5,6)
'''


## CTCF enrich check
region_tp = (6,)
ctcf_region_cell = get_CTCF_peaks_core_region(df_boundary_core_region_judge_no0, cell_type_list, region_tp, mat_len = 20)
   
ctcf_mean_region_cell = get_CTCF_peaks_mean_region(df_boundary_core_region_judge_no0, cell_type_list, region_tp)


save_name = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/cell_type_analysis/core_boundary/cell_cons/NHEK_core_region_CTCF_peak.svg'
draw_core_region_CTCF_peaks(ctcf_region_cell, cell_type_list, cell_color, save_name)    
      
draw_core_region_CTCF_mean_peaks(ctcf_mean_region_cell, cell_type_list, cell_color, save_name = '')   
    
    
    






















