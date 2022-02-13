# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:15:24 2022

@author: dcdang
"""

import os
import pandas as pd
import numpy as np
import time
import scipy.sparse 
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import random
import seaborn as sns
import pickle
import scipy
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker  import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from tqdm import tqdm
from scipy import ndimage
import matplotlib.ticker as ticker
from itertools import combinations
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn2, venn2_circles
from matplotlib.colors import ListedColormap
import matplotlib


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

### loading boundary region type

bd_region_type_record = read_save_data(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape' + '/' + 'boundary_region_type_for_all_cell_type.pkl')


###### boundary matching

## This results show example of how boundary matching works
def draw_square_region(st, ed, color, size_v, size_h):  
    ## 画竖线
    plt.vlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_v)
    plt.vlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_v)
    ## 画横线
    plt.hlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_h)
    plt.hlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_h)


def draw_tad_region(st, ed, color, size_v, size_h):  
    ## 画竖线
    plt.vlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_v)
    plt.vlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_v)
    ## 画横线
    plt.hlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_h)
    plt.hlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_h)

 
def draw_tad_region_upper_half(st, ed, color, size_v, size_h):  
    ## 画竖线
    #plt.vlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_v)
    plt.vlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_v)
    ## 画横线
    plt.hlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_h)
    #plt.hlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_h)

def get_bd_type_symbol(df_bd_region_type, bd_score_cell_combine):
    bd_symbol = np.zeros(len(bd_score_cell_combine))
    for i in range(len(df_bd_region_type)):
        region = df_bd_region_type['region'][i]
        bd_type = df_bd_region_type['region_type_adjust'][i]
        symbol = symbol_dic[bd_type]
        for x in region:
            bd_symbol[x] = symbol
    return bd_symbol



cell_type = 'GM12878'
enzyme = 'MboI'
mat_dense = copy.deepcopy(hic_mat_all_cell_replicate[cell_type][enzyme]['iced'])
mat_norm = copy.deepcopy(hic_mat_all_cell_replicate_znorm[cell_type][enzyme])
df_bd_insul_pvalue = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['pvalue'])
result_record = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['BD_region'])
df_boundary_region_combine = copy.deepcopy(result_record['Combine']['bd_region'])
bd_score_cell_combine = copy.deepcopy(result_record['Combine']['TAD_score'])
df_bd_region_type = copy.deepcopy(bd_region_type_record[cell_type + '_' + enzyme])

mat_use = copy.deepcopy(mat_dense)

symbol_dic = {'No-bd':0, 'sharp_weak':1, 'sharp_strong':2, 'wide':3}
bd_symbol = get_bd_type_symbol(df_bd_region_type, bd_score_cell_combine)


#save_data(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_score_all_cell.pkl', result_record_all)

def build_mat_target_new(bin_st, bin_ed, mat_use, resolution, cut_dist = 5000000):

    mat_target = np.zeros([len(mat_use), len(mat_use)])
    mat_use_target = mat_use[bin_st : bin_ed, bin_st:bin_ed] 

    norm_type = 'z-score'
    mat_target_norm_z = dist_normalize_matrix(mat_use_target, resolution, norm_type, cut_dist)
    mat_target[bin_st : bin_ed, bin_st : bin_ed] =  mat_target_norm_z
    mat_ori = mat_use[bin_st : bin_ed, bin_st:bin_ed]
    return mat_target, mat_target_norm_z, mat_ori

def get_region_bin_pair_best(region_st, score_st, region_ed, score_ed, up_bound_use, down_bound_use, st_cut, mat_use, mat_target, resolution, weight = 0.5):
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
            domain_value, up_value, down_value, fold_v, pvalue_vec = get_domain_value(bin1, bin2, up_bound_use, down_bound_use, mat_target, diag_cut = True)
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
    
    
def get_domain_value(bin1, bin2, up_bound_use, down_bound_use, mat_target, diag_cut = True):
    domain_l = bin2 - bin1 + 1
    domain_mat = mat_target[bin1:bin2+1,bin1:bin2+1]
    if diag_cut == True:
        if domain_l <= 5:
            domain_value = np.mean(domain_mat)
            domain_vec = domain_mat.flatten()
        else:
            # 去掉主、次对角线
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

def draw_explain_for_boundary_matching(norm_st, norm_ed, bin_st, bin_ed, up_bound_use, down_bound_use, Chr, mat_use, bd_score_cell_combine, resolution, save_name = '', expand_s = 10, bin_size = 10):
    st_all = norm_st - expand_s
    ed_all = norm_ed + expand_s     
    start_ =  st_all * resolution 
    end_ = ed_all * resolution
    cord_list = []
    pos_list = []
    pos_start = start_ 
    x_ticks_l = []
    y_ticks_l = []
    for i in range(ed_all - st_all):
        if i % bin_size == 0:
            cord_list.append(i)
            pos = pos_start + i*resolution
            pos_list.append(pos)
            if i + bin_size < ed_all - st_all:
                pos_label = str(pos / 1000000)
            else:
                pos_label = str(pos / 1000000) + 'Mb'
                #pos_label = str(pos / 1000000) 
            x_ticks_l.append(pos_label)
            y_ticks_l.append(str(pos / 1000000))
    region_name = Chr + ':' + str(start_ / 1000000) + '-' + str(end_ / 1000000) + ' Mb'  
    contact_map = mat_use[st_all:ed_all+1, st_all:ed_all+1]    
    fig = plt.figure(figsize=(7,7))
    ax1 = plt.subplot2grid((7, 7), (0, 0), rowspan=6,colspan=6)
    dense_matrix_part = contact_map
    img = ax1.imshow(dense_matrix_part, cmap='coolwarm', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
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
    ax1.set_title('Domain value calculate in :' + region_name, fontsize=12, pad = 15.0)
    # domain region
    st = bin_st - st_all
    ed = bin_ed - st_all
    plt.hlines(st, st, ed, linestyles='--', linewidth = 4)
    plt.hlines(ed, st, ed, linestyles='--', linewidth = 4)
    plt.vlines(st, st, ed, linestyles='--', linewidth = 4)
    plt.vlines(ed, st, ed, linestyles='--', linewidth = 4)  
    # up region
    st = bin_st - up_bound_use - st_all
    ed = bin_st - st_all
    ed2 = bin_ed - st_all
    plt.hlines(st, ed, ed2, linestyles='--', linewidth = 4)
    plt.hlines(ed, ed, ed2, linestyles='--', linewidth = 4)
    plt.vlines(ed, st, ed, linestyles='--', linewidth = 4)
    plt.vlines(ed2, st, ed, linestyles='--', linewidth = 4)     
    # down region
    st = bin_st - st_all
    ed = bin_ed - st_all
    ed2 = bin_ed + down_bound_use - st_all + 1
    plt.hlines(st, ed, ed2, linestyles='--', linewidth = 4)
    plt.hlines(ed, ed, ed2, linestyles='--', linewidth = 4)
    plt.vlines(st, st, ed, linestyles='--', linewidth = 4)
    plt.vlines(ed2, st, ed, linestyles='--', linewidth = 4)  
    
    cax = plt.subplot2grid((7, 7), (0, 6), rowspan=6,colspan=1)
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes("right", size="1.5%", pad= 0.2)
    #cbar = plt.colorbar(img, cax=cax, ticks=MultipleLocator(2.0), format="%.1f",orientation='vertical',extendfrac='auto',spacing='uniform')
    cbaxes = inset_axes(cax, width="30%", height="40%", loc=2) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])
   
    x_axis_range = list(range(len(bd_score_cell_combine['bd_score'][st_all:ed_all])))
    ax2 = plt.subplot2grid((7, 7), (6, 0), rowspan=1,colspan=6,sharex=ax1)
    ax2.plot(list(bd_score_cell_combine['bd_score'][st_all:ed_all]), color='black')
    ax2.bar(x_axis_range, list(bd_score_cell_combine['bd_score'][st_all:ed_all]))
    ax2.spines['bottom'].set_linewidth(1.6)
    ax2.spines['left'].set_linewidth(1.6)
    ax2.spines['right'].set_linewidth(1.6)
    ax2.spines['top'].set_linewidth(1.6)
    ax2.tick_params(axis = 'y', length=5, width = 1.6)
    ax2.tick_params(axis = 'x', length=5, width = 1.6)
    ax2.set_ylabel('Bd score', FontSize = 10)
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    #plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)


def boundary_match(df_bd_region_type, mat_use, resolution, weight):
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
        mat_target, mat_target_norm_z, mat_ori = build_mat_target_new(norm_st, norm_ed, mat_use, resolution, cut_dist = 5000000)

        df_res = get_region_bin_pair_best(region_st, score_st, region_ed, score_ed, up_bound_use, down_bound_use, st_cut, mat_use, mat_target, resolution, weight)
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


def draw_bd_region_case(st, ed, contact_map, TAD_list, bd_score_cell_combine, bd_symbol, Chr, save_name = '', bin_size = 8, resolution = 50000):
    x_axis_range = range(len(bd_score_cell_combine['bd_score'][st:ed]))
    start = st * resolution
    end = ed * resolution + resolution
    start_ = start / 1000000 
    end_ = end / 1000000
    region_name = Chr + ':' + str(start_) + '-' + str(end_) + ' Mb'
    x_ticks_l = []
    y_ticks_l = []
    cord_list = []
    for i in range(ed - st):
        if i % bin_size == 0:
            pos = (st+i) * resolution
            cord_list.append(i)
            x_ticks_l.append(str(pos / 1000000))
            y_ticks_l.append(str(pos / 1000000))
    
    plt.figure(figsize=(8, 8))     
    ax0 = plt.subplot2grid((6, 6), (0, 0), rowspan=4,colspan=4)
    dense_matrix_part = contact_map[st:ed+1, st:ed+1]
    img = ax0.imshow(dense_matrix_part, cmap='seismic', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
    #img = ax0.imshow(dense_matrix_part, cmap='coolwarm', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
    ax0.set_xticks([])
    #ax0.set_yticks([])
    ax0.spines['bottom'].set_linewidth(0)
    ax0.spines['left'].set_linewidth(1.6)
    ax0.spines['right'].set_linewidth(0)
    ax0.spines['top'].set_linewidth(0)
    ax0.tick_params(axis = 'y', length=5, width = 1.6)
    ax0.tick_params(axis = 'x', length=5, width = 1.6)
    plt.xticks(cord_list, x_ticks_l, FontSize = 10)
    plt.yticks(cord_list, y_ticks_l, FontSize = 10)
    ax0.set_title(region_name, fontsize=12, pad = 15.0)
    
    TAD_color = 'black'
    if len(TAD_list) != 0:
        for TAD in TAD_list:
            st_tad = TAD[0] - st
            ed_tad = TAD[1] - st
            print(st_tad, ed_tad)
            draw_tad_region(st_tad, ed_tad, TAD_color, size_v=3, size_h=3)
            #draw_tad_region_upper_half(st_tad, ed_tad, TAD_color, size_v=3, size_h=3)

    cax = plt.subplot2grid((6, 6), (0, 4), rowspan=4,colspan=1)
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes("right", size="1.5%", pad= 0.2)
    #cbar = plt.colorbar(img, cax=cax, ticks=MultipleLocator(2.0), format="%.1f",orientation='vertical',extendfrac='auto',spacing='uniform')
    cbaxes = inset_axes(cax, width="30%", height="50%", loc=2) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])

    ax1 = plt.subplot2grid((6, 6), (4, 0), rowspan=1, colspan=4, sharex=ax0)    
    ax1.plot(x_axis_range, bd_score_cell_combine['bd_score'][st:ed], marker = '.', linewidth = 2, c = 'black')
    ax1.bar(x_axis_range, list(bd_score_cell_combine['bd_score'][st:ed]))
    plt.ylabel('bd_score')
    ax1.set_xticks([])
    #ax1.set_yticks([])
    ax1.spines['bottom'].set_linewidth(1.6)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(1.6)
    ax1.spines['top'].set_linewidth(1.6)
    ax1.tick_params(axis = 'y', length=5, width = 1.6)
    ax1.tick_params(axis = 'x', length=5, width = 1.6)

    ax2 = plt.subplot2grid((6, 6), (5, 0), rowspan=1,colspan=4, sharex=ax0)    
    bd_data = []
    cmap=['#B0B0B0','#459457','#D65F4D','#4392C3']
    my_cmap = ListedColormap(cmap)
    bounds=[0,0.9,1.9,2.9,3.9]
    norm = matplotlib.colors.BoundaryNorm(bounds, my_cmap.N)    
    for i in range(10):
        bd_data.append(bd_symbol[st:ed])
    #bd_data_expand = np.reshape(np.array(bd_data), (10, len(bd_data[0])))
    ax2.imshow(bd_data, cmap = my_cmap, Norm = norm)
    plt.ylabel('bd type')
    ax2.spines['bottom'].set_linewidth(1.6)
    ax2.spines['left'].set_linewidth(1.6)
    ax2.spines['right'].set_linewidth(1.6)
    ax2.spines['top'].set_linewidth(1.6)
    ax2.tick_params(axis = 'y', length=0, width = 0)
    ax2.tick_params(axis = 'x', length=0, width = 0)    
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

df_record = boundary_match(df_bd_region_type, mat_use, resolution, weight = 0.5)

start = 2734 - 32
end = 2734 + 22

start = 1422 - 30
end = 1422 + 30

start = 177 - 30
end = 177 + 30

start = 0
end = 80


TAD_list = []
for i in range(len(df_record)):
    st = df_record['region_pair'][i][0]
    ed = df_record['region_pair'][i][-1]
    if st >= start and ed <= end:
        TAD_list.append(df_record['region_pair'][i])

draw_bd_region_case(start, end, mat_use, TAD_list, bd_score_cell_combine, bd_symbol, Chr, save_name = '', bin_size = 10, resolution = 50000)


    

### Perform boundary matching and analysis the results

def get_TAD_domain_boundary(df_record, df_bd_region_type, Chr, resolution):
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


df_tad_cons, df_boundary_cons = get_TAD_domain_boundary(df_record, df_bd_region_type, Chr, resolution)



## load bio data
bio_data_add = 'E:/Users/dcdang/share/TAD_integrate/GM12878_data_for_use/bigwig_bed_file'
bio_type_list = ['epigenetics', 'RNA-seq', 'Repli-seq']
df_human_chr_bio_GM12878 = pd.DataFrame()

bio_name_list = []
bio_name_type_list = []
for bio_type in bio_type_list:   
    if bio_type == 'epigenetics':
        for bio_label in ['Histone', 'Tfbs', 'Methylation', 'OpenChromatin']:
            bio_import_add = bio_data_add + '/' + bio_type + '/' + bio_label
            Files = os.listdir(bio_import_add)
            for file in Files:
                bio_name = file.split('_')[2]
                bio_name_type = file.split('_')[0]
                df_bio_data = pd.read_csv(bio_import_add + '/' + file, sep = '\t', header = None)
                df_human_chr_bio_GM12878[bio_name] = df_bio_data[0]
                bio_name_list.append(bio_name)
                bio_name_type_list.append(bio_name_type)
    else:
        bio_import_add = bio_data_add + '/' + bio_type
        Files = os.listdir(bio_import_add)
        for file in Files:
            bio_name = file.split('_')[2]
            #if bio_name == 'RepWave':
                #continue
            bio_name_type = file.split('_')[0]
            df_bio_data = pd.read_csv(bio_import_add + '/' + file, sep = '\t', header = None)
            df_human_chr_bio_GM12878[bio_name] = df_bio_data[0]
            bio_name_list.append(bio_name)
            bio_name_type_list.append(bio_name_type)


### enrich analysis
            
def get_tad_bio_signal(df_tad, df_human_chr_bio_GM12878, bio_type, resolution, norm = True):
    signal_mean = np.mean(df_human_chr_bio_GM12878[bio_type])
    signal_l = []
    for i in range(len(df_tad)):
        st = df_tad['start'][i]
        ed = df_tad['end'][i]
        st_index = int(st / resolution)
        ed_index = int(ed / resolution - 1)
        signal_vec = np.array(df_human_chr_bio_GM12878[bio_type][st_index:ed_index + 1])
        if norm == True:
            signal_norm = signal_vec / signal_mean
        else:
            signal_norm = signal_vec
        signal_l.append(np.mean(signal_norm))
    df_tad[bio_type] = signal_l
    return df_tad

def get_method_tad_domain_signal(df_tad_cons_c, df_GM12878_mboi_chr2, resolution):
    for bio_type in ['H3K36me3', 'H3K27me3', 'H3K36me3/H3K27me3']:
        if bio_type == 'H3K36me3': 
            norm = True
            df_tad_cons_c = get_tad_bio_signal(df_tad_cons_c, df_GM12878_mboi_chr2, bio_type, resolution, norm = norm)
        elif bio_type == 'H3K27me3':
            norm = True
            df_tad_cons_c = get_tad_bio_signal(df_tad_cons_c, df_GM12878_mboi_chr2, bio_type, resolution, norm = norm)
        elif bio_type == 'H3K36me3/H3K27me3':
            norm = False
            df_tad_cons_c = get_tad_bio_signal(df_tad_cons_c, df_GM12878_mboi_chr2, bio_type, resolution, norm = norm)
    return df_tad_cons_c


def get_random_background_fold_value(df_tad_cons_c, df_tad_cons, df_GM12878_mboi_chr2, resolution, rand_num = 1000):
    index_bin = list(range(len(df_GM12878_mboi_chr2)))
    random_record = []
    for num in range(rand_num):
        if num != 0 and num % 200 == 0:
            print('200 random done!')
        index_shullf = np.random.permutation(index_bin)
        df_GM12878_mboi_chr2_shullf = df_GM12878_mboi_chr2.iloc[index_shullf]
        df_GM12878_mboi_chr2_shullf = df_GM12878_mboi_chr2_shullf.reset_index(drop = True)   
        # df_tad_cons is always new
        df_tad_cons_random = copy.deepcopy(df_tad_cons)
        bio_type = 'H3K36me3/H3K27me3'
        norm = False    
        df_tad_cons_random = get_tad_bio_signal(df_tad_cons_random, df_GM12878_mboi_chr2_shullf, bio_type, resolution, norm = norm)
        random_record.append(list(df_tad_cons_random[bio_type]))    
    up_ratio = []
    blow_ratio = []
    random_record = np.array(random_record)
    for i in range(len(df_tad_cons_c)):
        fold_v = df_tad_cons_c['H3K36me3/H3K27me3'][i]
        random_record[:,0]
        p1 = np.sum(random_record[:,i] >= fold_v) / len(random_record[:,i])
        p2 = np.sum(random_record[:,i] <= fold_v) / len(random_record[:,i])
        up_ratio.append(p1)
        blow_ratio.append(p2)
    df_tad_cons_c['K36/K27_up'] = up_ratio
    df_tad_cons_c['K36/K27_low'] = blow_ratio
    
    df_tad_cons_c_part = df_tad_cons_c[(df_tad_cons_c['end'] - df_tad_cons_c['start']) >= 200000]
    df_tad_cons_c_part = df_tad_cons_c_part.reset_index(drop = True)
    #df_tad_cons_c_part = copy.deepcopy(df_tad_cons_c)
    up_num = np.sum(np.array(df_tad_cons_c_part['K36/K27_up']) <= 0.05)    
    low_num = np.sum(np.array(df_tad_cons_c_part['K36/K27_low']) <= 0.05)        
    up_domain_r = up_num / len(df_tad_cons_c_part)
    low_domain_r = low_num / len(df_tad_cons_c_part)
    return df_tad_cons_c, up_num, low_num, up_domain_r, low_domain_r  

def get_random_background_fold_value_with_fixed_shullf_index(df_tad_cons_c, df_tad_cons, df_GM12878_mboi_chr2, random_shullf_all, resolution):
    random_record = []
    for num in range(len(random_shullf_all)):
        index_shullf = random_shullf_all[num]
        if num != 0 and num % 200 == 0:
            print('200 random done!')
        df_GM12878_mboi_chr2_shullf = df_GM12878_mboi_chr2.iloc[index_shullf]
        df_GM12878_mboi_chr2_shullf = df_GM12878_mboi_chr2_shullf.reset_index(drop = True)   
        # df_tad_cons is always new
        df_tad_cons_random = copy.deepcopy(df_tad_cons)
        bio_type = 'H3K36me3/H3K27me3'
        norm = False    
        df_tad_cons_random = get_tad_bio_signal(df_tad_cons_random, df_GM12878_mboi_chr2_shullf, bio_type, resolution, norm = norm)
        random_record.append(list(df_tad_cons_random[bio_type]))    
    up_ratio = []
    blow_ratio = []
    random_record = np.array(random_record)
    for i in range(len(df_tad_cons_c)):
        fold_v = df_tad_cons_c['H3K36me3/H3K27me3'][i]
        #random_record[:,0]
        p1 = np.sum(random_record[:,i] >= fold_v) / len(random_record[:,i])
        p2 = np.sum(random_record[:,i] <= fold_v) / len(random_record[:,i])
        up_ratio.append(p1)
        blow_ratio.append(p2)
    df_tad_cons_c['K36/K27_up'] = up_ratio
    df_tad_cons_c['K36/K27_low'] = blow_ratio
    df_tad_cons_c_part = df_tad_cons_c[(df_tad_cons_c['end'] - df_tad_cons_c['start']) >= 200000]
    df_tad_cons_c_part = df_tad_cons_c_part.reset_index(drop = True)
    #df_tad_cons_c_part = copy.deepcopy(df_tad_cons_c)
    up_num = np.sum(np.array(df_tad_cons_c_part['K36/K27_up']) <= 0.05)    
    low_num = np.sum(np.array(df_tad_cons_c_part['K36/K27_low']) <= 0.05)        
    up_domain_r = up_num / len(df_tad_cons_c_part)
    low_domain_r = low_num / len(df_tad_cons_c_part)
    true_n = len(df_tad_cons_c_part)
    return df_tad_cons_c, up_num, low_num, up_domain_r, low_domain_r, true_n  


df_GM12878_mboi_chr2 = copy.deepcopy(df_human_chr_bio_GM12878[['H3K36me3', 'H3K27me3']])
fold_l = []
for i in range(len(df_GM12878_mboi_chr2)):
    k36_value = df_GM12878_mboi_chr2['H3K36me3'][i] / np.mean(df_GM12878_mboi_chr2['H3K36me3'])
    k27_value = df_GM12878_mboi_chr2['H3K27me3'][i] / np.mean(df_GM12878_mboi_chr2['H3K27me3'])
    if k36_value == 0 or k27_value == 0:
        fold_v = 0
    else:
        fold_v = np.log2(k36_value / k27_value)
    fold_l.append(fold_v)
df_GM12878_mboi_chr2['H3K36me3/H3K27me3'] = fold_l        


TAD_result_GM12878_MboI = copy.deepcopy(TAD_result_all_cell_type['GM12878']['MboI'])
TAD_result_GM12878_MboI['Consensus'] = {}
TAD_result_GM12878_MboI['Consensus']['TAD_domain'] = df_tad_cons
TAD_result_GM12878_MboI['Consensus']['TAD_boundary'] = df_boundary_cons

### test
df_tad_cons = copy.deepcopy(TAD_result_GM12878_MboI['Consensus']['TAD_domain'])
df_tad_cons_copy = copy.deepcopy(df_tad_cons)
df_tad_cons = get_method_tad_domain_signal(df_tad_cons, df_GM12878_mboi_chr2, resolution)
df_tad_cons, up_num, low_num, up_domain_r, low_domain_r = get_random_background_fold_value(df_tad_cons, df_tad_cons_copy, df_GM12878_mboi_chr2, resolution, rand_num = 10)


df_tad_method = copy.deepcopy(TAD_result_GM12878_MboI['DI']['TAD_domain'])
df_tad_method_copy = copy.deepcopy(df_tad_method)
df_tad_method = get_method_tad_domain_signal(df_tad_method, df_GM12878_mboi_chr2, resolution)
df_tad_method, up_num, low_num, up_domain_r, low_domain_r = get_random_background_fold_value(df_tad_method, df_tad_method_copy, df_GM12878_mboi_chr2, resolution, rand_num = 1000)


## all method test for k36/k27 signal

random.seed(2021)

shullf_index_all = []
rand_num = 1000
index_bin = list(range(len(df_GM12878_mboi_chr2)))
for i in range(rand_num):
    index_shullf = np.random.permutation(index_bin)
    shullf_index_all.append(index_shullf)

df_record_signal = pd.DataFrame(columns = ['up_num', 'up_ratio', 'low_num', 'low_ratio'])
TAD_signal_result = {}
up_n_l = []
up_r_l = []
low_n_l = []
low_r_l = []
all_n_l = []
true_n_l = []
for method in list(TAD_result_GM12878_MboI.keys()):
    print('This is ' + method)
    df_tad_method = copy.deepcopy(TAD_result_GM12878_MboI[method]['TAD_domain'])
    df_tad_method_copy = copy.deepcopy(df_tad_method)
    all_n_l.append(len(df_tad_method))
    df_tad_method = get_method_tad_domain_signal(df_tad_method, df_GM12878_mboi_chr2, resolution)
    #df_tad_method = copy.deepcopy(TAD_signal_result[method])
    df_tad_method, up_num, low_num, up_domain_r, low_domain_r, true_n = get_random_background_fold_value_with_fixed_shullf_index(df_tad_method, df_tad_method_copy, df_GM12878_mboi_chr2, shullf_index_all, resolution)
    up_n_l.append(up_num)
    up_r_l.append(up_domain_r)
    low_n_l.append(low_num)
    low_r_l.append(low_domain_r)
    true_n_l.append(true_n)
    TAD_signal_result[method] = df_tad_method
df_record_signal['up_num'] = up_n_l
df_record_signal['up_ratio'] = up_r_l
df_record_signal['low_num'] = low_n_l
df_record_signal['low_ratio'] = low_r_l
df_record_signal['all_number'] = all_n_l
df_record_signal['true_number'] = true_n_l
df_record_signal['method'] = list(TAD_result_GM12878_MboI.keys())


#import statsmodels
import mne

df_record_signal_new = copy.deepcopy(df_record_signal)

up_adj = []
low_adj = []
true_num = []
up_adj_ratio = []
low_adj_ratio = []
for method in list(df_record_signal['method']):
    df_tad = TAD_signal_result[method]
    df_tad_part = copy.deepcopy(df_tad[(df_tad['end'] - df_tad['start']) >= 200000])
    df_tad_part = df_tad_part.reset_index(drop = True)
    true_num.append(len(df_tad_part))
    up_pvalue = np.array(df_tad_part['K36/K27_up'])
    low_pvalue = np.array(df_tad_part['K36/K27_low'])   
    #up_p_adjust = statsmodels.stats.multitest.fdrcorrection(up_pvalue, alpha=0.1, method='indep', is_sorted=False)
    #low_p_adjust = statsmodels.stats.multitest.fdrcorrection(low_pvalue, alpha=0.1, method='indep', is_sorted=False)
    up_p_adjust = mne.stats.fdr_correction(up_pvalue, alpha=0.1, method='indep')
    low_p_adjust = mne.stats.fdr_correction(low_pvalue, alpha=0.1, method='indep')
    df_tad_part['K36/K27_up_adj'] = up_p_adjust[-1]
    df_tad_part['K36/K27_low_adj'] = low_p_adjust[-1]
    up_adj.append(np.sum(up_p_adjust[0]))
    low_adj.append(np.sum(low_p_adjust[0]))
    up_adj_ratio.append(np.sum(up_p_adjust[0]) / len(df_tad_part))
    low_adj_ratio.append(np.sum(low_p_adjust[0]) / len(df_tad_part))

df_record_signal_new['up_num_adj'] = np.array(up_adj)    
df_record_signal_new['low_num_adj'] = np.array(low_adj)
df_record_signal_new['up_ratio_adj'] = up_adj_ratio   
df_record_signal_new['low_ratio_adj'] = low_adj_ratio
df_record_signal_new['true_num'] = true_num


'''
for method in list(df_record_signal['method']):
    print('This is ' + method)
    df_tad = copy.deepcopy(TAD_signal_result[method])
    df_tad_part = df_tad[(df_tad['end'] - df_tad['start']) >= 200000]
    up_num = np.sum(np.array(df_tad_part['K36/K27_up']) <= 0.05)    
    low_num = np.sum(np.array(df_tad_part['K36/K27_low']) <= 0.05)        
    up_domain_r = up_num / len(df_tad_part)
    low_domain_r = low_num / len(df_tad_part)
    print('Up num:' + str(up_num))
    print('Low num:' + str(low_num))
    print('Low ratio:' + str(up_domain_r))
    print('Low ratio:' + str(low_domain_r))
    up_pvalue = np.array(df_tad_part['K36/K27_up'])
    low_pvalue = np.array(df_tad_part['K36/K27_low'])   
    up_p_adjust = statsmodels.stats.multitest.fdrcorrection(up_pvalue, alpha=0.1, method='indep', is_sorted=False)
    low_p_adjust = statsmodels.stats.multitest.fdrcorrection(low_pvalue, alpha=0.1, method='indep', is_sorted=False)
    print('Up adj:' + str(np.sum(up_p_adjust[0])))
    print('Low adj:' + str(np.sum(low_p_adjust[0])))
'''

from colormap import Color, Colormap

def generate_cmap(colors):
    red_list=list()
    green_list=list()
    blue_list=list()
    #['darkblue','seagreen','yellow','gold','coral','hotpink','red'],['white','green','blue','red']
    rgb_list = []
    for color in colors:
        col=Color(color).rgb
        rgb_list.append(col)
        red_list.append(col[0])
        green_list.append(col[1])
        blue_list.append(col[2])
    c = Colormap()
    d=  {   'blue': blue_list,
            'green':green_list,
            'red':red_list}
    mycmap = c.cmap(d) 
    return mycmap, rgb_list
    
def draw_differential_signal_domain_number(df_record_signal):
    df_num_k36 = pd.DataFrame(columns = ['method', 'num'])
    df_num_k27 = pd.DataFrame(columns = ['method', 'num'])    
    color_singl = ['#FF3333', '#00CC00']  
    method_l = []
    k36_l = []
    k27_l = []
    sum_num = []               
    for i in range(len(df_record_signal)):
        method_l.append(df_record_signal['method'][i])
        k36_l.append(df_record_signal['up_ratio_adj'][i])
        k27_l.append(-df_record_signal['low_ratio_adj'][i])
        sum_num.append(df_record_signal['up_ratio_adj'][i] + df_record_signal['low_ratio_adj'][i])    
    df_num_k36['method'] = method_l    
    df_num_k36['num'] = k36_l    
    df_num_k27['method'] = method_l    
    df_num_k27['num'] = k27_l 
    
    method_ord = []
    for x in np.argsort(-np.array(sum_num)):
        method_ord.append(method_l[x])    
    mycmap, rgb_list_k36= generate_cmap(['#00CC00'])
    mycmap, rgb_list_k27 = generate_cmap(['#FF3333'])    
    plt.figure(figsize=(6, 6))
    sns.barplot(x = 'method', y = 'num', data = df_num_k36, order = method_ord, palette = rgb_list_k36)
    sns.barplot(x = 'method', y = 'num', data = df_num_k27, order = method_ord, palette = rgb_list_k27)
    plt.xticks(rotation= 30, FontSize = 10)
    plt.yticks(FontSize = 10, weight = 'semibold')
    plt.ylabel('Significant TAD ratio',  FontSize = 10)
    plt.axhline(0, color = 'black', linestyle='-')
    #plt.legend(loc = 'upper left')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0) 
                                           
                                           
def draw_differential_signal_domain_number_separate(df_record_signal, type_ = 'ratio', save_name1 = '', save_name2 = ''):
    if type_ == 'ratio':
        up_type = 'up_ratio_adj'
        low_type = 'low_ratio_adj'
    elif type_ == 'number':
        up_type = 'up_num_adj'
        low_type = 'low_num_adj'
       
    df_num_k36 = pd.DataFrame(columns = ['method', 'num'])
    df_num_k27 = pd.DataFrame(columns = ['method', 'num'])    
    color_singl = ['#FF3333', '#00CC00']  
    method_l = []
    k36_l = []
    k27_l = []
    sum_num = []               
    for i in range(len(df_record_signal)):
        method_l.append(df_record_signal['method'][i])
        k36_l.append(df_record_signal[up_type][i])
        k27_l.append(df_record_signal[low_type][i])
        sum_num.append(df_record_signal[up_type][i] + df_record_signal[low_type][i])    
    df_num_k36['method'] = method_l    
    df_num_k36['num'] = k36_l    
    df_num_k27['method'] = method_l    
    df_num_k27['num'] = k27_l 
    
    method_ord = []
    for x in np.argsort(-np.array(sum_num)):
        method_ord.append(method_l[x])    
    mycmap, rgb_list_k36= generate_cmap(['#00CC00'])
    mycmap, rgb_list_k27 = generate_cmap(['#FF3333'])    
    plt.figure(figsize=(6, 3))
    sns.barplot(x = 'method', y = 'num', data = df_num_k36, order = method_ord, palette = rgb_list_k36)
    plt.xticks(rotation= 30, FontSize = 10)
    if type_ == 'ratio':
        plt.yticks([0, 0.05, 0.10, 0.15, 0.20], FontSize = 10)
    plt.ylabel('Significant TAD ' + type_,  FontSize = 10)
    plt.axhline(0, color = 'black', linestyle='-')
    #plt.legend(loc = 'upper left')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0) 
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    if save_name1 != '':
        plt.savefig(save_name1, format = 'svg', transparent = True) 
        
    plt.figure(figsize=(6, 3))
    sns.barplot(x = 'method', y = 'num', data = df_num_k27, order = method_ord, palette = rgb_list_k27)
    plt.xticks(rotation= 30, FontSize = 10)
    if type_ == 'ratio':
        plt.yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25], FontSize = 10)
    plt.ylabel('Significant TAD ' + type_,  FontSize = 10)
    plt.axhline(0, color = 'black', linestyle='-')
    #plt.legend(loc = 'upper left')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0) 
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    if save_name2 != '':
        plt.savefig(save_name2, format = 'svg', transparent = True) 
    return method_ord
                                          
save_name1 = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\boundary_match' + '/' + 'differential_epi/' + 'domain_H3K36me3_number.svg'
save_name2 = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\boundary_match' + '/' + 'differential_epi/' + 'domain_H3K27me3_number.svg'                                   
method_ord = draw_differential_signal_domain_number_separate(df_record_signal_new, type_ = 'number', save_name1 = save_name1, save_name2 = save_name2)  
    
save_name1 = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\boundary_match' + '/' + 'differential_epi/' + 'domain_H3K36me3_ratio.svg'
save_name2 = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\boundary_match' + '/' + 'differential_epi/' + 'domain_H3K27me3_ratio.svg'                                   
method_ord = draw_differential_signal_domain_number_separate(df_record_signal_new, type_ = 'ratio', save_name1 = save_name1, save_name2 = save_name2)  
 


## use less, can't see difference

df_tad_cons = copy.deepcopy(TAD_signal_result['Consensus'])
up_pvalue = np.array(df_tad_cons['K36/K27_up'])
low_pvalue = np.array(df_tad_cons['K36/K27_low'])   
up_p_adjust = statsmodels.stats.multitest.fdrcorrection(up_pvalue, alpha=0.1, method='indep', is_sorted=False)
low_p_adjust = statsmodels.stats.multitest.fdrcorrection(low_pvalue, alpha=0.1, method='indep', is_sorted=False)
df_tad_cons['K36/K27_up_adj'] = up_p_adjust[-1]
df_tad_cons['K36/K27_low_adj'] = low_p_adjust[-1]


df_tad_cons_up = df_tad_cons[df_tad_cons['K36/K27_up_adj'] <= 0.1]
df_tad_cons_up = df_tad_cons_up.reset_index(drop = True)
bd_type_list = ['sharp_weak', 'sharp_strong', 'wide']
for bd_type in bd_type_list:
    print(bd_type)
    st_num = np.sum(df_tad_cons_up['st_region_type'] == bd_type)
    ed_num = np.sum(df_tad_cons_up['ed_region_type'] == bd_type)
    print( st_num / len(df_tad_cons_up)) 
    print( ed_num / len(df_tad_cons_up)) 
    print((st_num + ed_num) / (2*len(df_tad_cons_up))) 

df_tad_cons_low = df_tad_cons[df_tad_cons['K36/K27_low_adj'] <= 0.1]
df_tad_cons_low = df_tad_cons_low.reset_index(drop = True)
for bd_type in bd_type_list:
    print(bd_type)
    st_num = np.sum(df_tad_cons_low['st_region_type'] == bd_type)
    ed_num = np.sum(df_tad_cons_low['ed_region_type'] == bd_type)
    print( st_num / len(df_tad_cons_low)) 
    print( ed_num / len(df_tad_cons_low)) 
    print((st_num + ed_num) / (2*len(df_tad_cons_low))) 

### this should use

k36_k27_LR_indicator = []
for i in range(len(df_GM12878_mboi_chr2)):
    if df_GM12878_mboi_chr2['H3K36me3/H3K27me3'][i] <= 0:
        k36_k27_LR_indicator.append(-1)
    elif df_GM12878_mboi_chr2['H3K36me3/H3K27me3'][i] > 0:
        k36_k27_LR_indicator.append(1)

df_GM12878_mboi_chr2['LR_indicator'] = k36_k27_LR_indicator        

lr_change_point = []
for i in range(1, len(df_GM12878_mboi_chr2)-1):
    sign_b = df_GM12878_mboi_chr2['LR_indicator'][i-1]
    sign_c = df_GM12878_mboi_chr2['LR_indicator'][i]
    sign_n = df_GM12878_mboi_chr2['LR_indicator'][i+1]
    if sign_c != sign_b and sign_c == sign_n:
        lr_change_point.append(i)

lr_c_dist = {}
for method in list(TAD_result_GM12878_MboI.keys()):
    print(method)
    dist_l = []
    #df_TAD = copy.deepcopy(TAD_signal_result[method])  
    df_TAD = copy.deepcopy(TAD_result_GM12878_MboI[method]['TAD_domain'])
    st_l = np.array(df_TAD['start']) / resolution
    ed_l = np.array(df_TAD['end']) / resolution - 1
    for i in range(len(lr_change_point)):
        pos = lr_change_point[i]
        st_d = np.min(np.abs(st_l - pos))
        ed_d = np.min(np.abs(ed_l - pos))
        dist_l.append(np.min([st_d, ed_d]))
    lr_c_dist[method] = dist_l         
    


''' 
#calculate the minmum distance to LR changing point or boundary ratio that close to LR changing poing
# not good for consensus boundary, but it's not shure all boundary should be speration for H3K36 and H3K27 signal
lr_c_dist = {}
lr_c_boundary_ratio = {}
lr_c_boundary_dist = {}
dist_cut = 5
for method in list(TAD_result_GM12878_MboI.keys()):
    print(method)
    dist_l = []
    df_boundary = copy.deepcopy(TAD_result_GM12878_MboI[method]['TAD_boundary'])
    print(len(df_boundary))
    st_l = np.array(df_boundary['start']) / resolution
    st_l = set(st_l)
    st_l = sorted(st_l)
    for i in range(len(lr_change_point)):
        pos = lr_change_point[i]
        st_d = np.min(np.abs(np.array(st_l) - pos))
        dist_l.append(st_d)
    lr_c_dist[method] = dist_l
    bd_dist_l = []
    count = 0
    for j in range(len(st_l)):
        bd = st_l[j]
        bd_dist = np.min(np.abs(np.array(lr_change_point) - bd))
        bd_dist_l.append(bd_dist)
        if bd_dist <= dist_cut:
            count += 1
    lr_c_boundary_dist[method] = bd_dist_l
    lr_c_boundary_ratio[method] = count / len(st_l)
'''    

    
def draw_method_domain_bd_to_differential_change_point(lr_c_dist, method_color, save_name = ''):
    df_dist = pd.DataFrame(columns = ['dist', 'method'])
    dist_l = []
    m_l = []
    method_ord = []
    color_use = []
    for method in list(lr_c_dist.keys()):
        dist_l += lr_c_dist[method]
        m_l += [method for i in range(len(lr_c_dist[method]))]
        method_ord.append(np.mean(lr_c_dist[method]))
    index_l = np.argsort(method_ord)
    method_ord = []
    for i in range(len(index_l)):
        method = list(lr_c_dist.keys())[index_l[i]]
        method_ord.append(method)
        if method != 'Consensus':
            color_use.append(method_color[method])
        elif method == 'Consensus':
            color_use.append('#3F85FF')     
    df_dist['dist'] = dist_l
    df_dist['method'] = m_l        
    plt.figure(figsize=(9, 4))
    sns.barplot(x = 'method', y = 'dist', data = df_dist, order = method_ord, capsize = 0.2, saturation = 8,             
                errcolor = 'black', errwidth = 1.5, ci = 95, edgecolor='black', palette = color_use)    
    #plt.legend(loc = 'upper left', fontsize = 12)
    plt.xlabel('Method',  FontSize = 0)
    plt.ylabel('Minimum distance of LR values changing point to domain boundary (#bins)',  FontSize = 12)
    plt.xticks(FontSize = 12, rotation = -30)
    plt.yticks(FontSize = 12)
    #plt.ylim([0,40])
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
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

save_name = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\boundary_match\differential_epi' + '/' + 'H3K36me3_H3K27me3_changing_point_dist_to_domain_bd.svg'    
draw_method_domain_bd_to_differential_change_point(lr_c_dist, method_color, save_name = save_name)

#for method in list(TAD_signal_result.keys()):
    #print(method + ' ' + str(len(TAD_signal_result[method])))


### check LR changing point position and domain bd, compare with random control,
### we find the LR point close to consensus domain bd not significant better than random 
def get_domain_partion(df_tad_cons_c, df_human_chr_bio_GM12878, resolution):   
    df_domain_info = pd.DataFrame(columns = ['length', 'type', 'index'])
    length_l = []
    type_l = []
    index_l = []
    st = df_tad_cons_c['start'][0]
    if int(st / resolution) != 0:
        length_l.append(int(st / resolution))
        type_l.append('segment')
        index_l.append(0)
    for i in range(len(df_tad_cons_c) - 1):
        st = df_tad_cons_c['start'][i]
        ed = df_tad_cons_c['end'][i]
        st_n = df_tad_cons_c['start'][i+1]
        st_ = int(st / resolution)
        ed_ = int(ed / resolution) - 1
        st_n_ = int(st_n / resolution)
        length_l.append(ed_ - st_ + 1)
        type_l.append('domain')
        index_l.append(i)
        if st_n_ != ed_ + 1:
            length_l.append(st_n_ - ed_ - 1)
            type_l.append('segment')
            index_l.append(i)
    i = len(df_tad_cons_c)-1
    st = df_tad_cons_c['start'][i]
    ed = df_tad_cons_c['end'][i]
    st_ = int(st / resolution)
    ed_ = int(ed / resolution) - 1
    length_l.append(ed_ - st_ + 1)
    type_l.append('domain')
    index_l.append(i)
    if ed_ != len(df_human_chr_bio_GM12878) - 1:
        length_l.append(len(df_human_chr_bio_GM12878) - 1 - ed_)
        type_l.append('segment')
        index_l.append(i)        
    df_domain_info['length'] = length_l
    df_domain_info['type'] = type_l
    df_domain_info['index'] = index_l
    return df_domain_info

def get_random_domain_by_shullfe_info(df_domain_info, df_human_chr_bio_GM12878, Chr = 'chr2'):
    index = list(range(len(df_domain_info)))
    index_shullf = np.random.permutation(index)
    df_domain_info_shullf = df_domain_info.iloc[index_shullf]
    df_domain_info_shullf = df_domain_info_shullf.reset_index(drop = True)   
    st_l = []
    ed_l = []
    type_l = []
    d_index_l = []
    sum_start = 0
    for i in range(len(df_domain_info_shullf)):
        d_len = df_domain_info_shullf['length'][i]
        type_ = df_domain_info_shullf['type'][i]
        d_index = df_domain_info_shullf['index'][i]
        st_l.append(sum_start)
        ed_l.append(sum_start + d_len - 1)
        type_l.append(type_)
        d_index_l.append(d_index)
        sum_start = sum_start + d_len
    if sum_start != len(df_human_chr_bio_GM12878):
        print('Error!')
    df_domain_random = pd.DataFrame(columns = ['chr', 'start', 'end', 'type', 'domain_index'])
    df_domain_random['start'] = st_l
    df_domain_random['end'] = ed_l
    df_domain_random['type'] = type_l
    df_domain_random['domain_index'] = d_index_l
    df_domain_random['chr'] = [Chr for i in range(len(df_domain_random))]    
    df_domain_random_part = df_domain_random[df_domain_random['type'] == 'domain']
    df_domain_random_part = df_domain_random_part.reset_index(drop = True)
    return df_domain_random_part


def get_random_domain_by_sample_info(df_domain_info, df_human_chr_bio_GM12878, Chr = 'chr2'):
    df_domain_info_part = df_domain_info[df_domain_info['type'] == 'domain']
    df_domain_info_part = df_domain_info_part.reset_index(drop = True)
    st_l = []
    ed_l = []
    type_l = []
    d_index_l = []
    sum_start = 0
    for i in range(len(df_domain_info_part)):
        index = np.random.randint(0, len(df_domain_info_part))
        length = df_domain_info_part['length'][index]
        st_l.append(sum_start)
        if (sum_start + length) >= len(df_human_chr_bio_GM12878)-1:
            ed_l.append(len(df_human_chr_bio_GM12878)-1)
        else:
            ed_l.append(sum_start + length - 1)
        sum_start = sum_start + length  
    df_domain_random = pd.DataFrame(columns = ['chr', 'start', 'end'])
    df_domain_random['start'] = st_l
    df_domain_random['end'] = ed_l
    df_domain_random['chr'] = [Chr for i in range(len(df_domain_random))]    
    return df_domain_random

def get_dist_of_LR_to_bd_random_domain(df_domain_random, lr_change_point):
    df_lr_bd = pd.DataFrame(columns = ['distance', 'domain_index', 'st_ed'])
    dist_l = []
    index_l = []
    st_ed_l = []
    st_l = np.array(df_domain_random['start'])
    ed_l = np.array(df_domain_random['end'])
    for i in range(len(lr_change_point)):
        pos = lr_change_point[i]
        dist_st = np.min(np.abs(st_l - pos))
        st_index = np.argmin(np.abs(st_l - pos))
        dist_ed = np.min(np.abs(ed_l - pos))
        ed_index = np.argmin(np.abs(ed_l - pos))
        if dist_st <= dist_ed:
            dist_l.append(dist_st)
            index_l.append(st_index)
            st_ed_l.append('st')
        else:
            dist_l.append(dist_ed)
            index_l.append(ed_index)
            st_ed_l.append('ed')
    df_lr_bd['distance'] = dist_l
    df_lr_bd['domain_index'] = index_l
    df_lr_bd['st_ed'] = st_ed_l
    return df_lr_bd

def get_dist_of_LR_to_bd(df_tad_cons, lr_change_point):
    df_lr_bd = pd.DataFrame(columns = ['distance', 'domain_index', 'st_ed', 'bd_type'])
    dist_l = []
    index_l = []
    bd_t_l = []
    st_ed_l = []
    st_l = np.array(df_tad_cons['bd_st'])
    ed_l = np.array(df_tad_cons['bd_ed'])
    for i in range(len(lr_change_point)):
        pos = lr_change_point[i]
        dist_st = np.min(np.abs(st_l - pos))
        st_index = np.argmin(np.abs(st_l - pos))
        st_type = df_tad_cons['st_region_type'][st_index]
        dist_ed = np.min(np.abs(ed_l - pos))
        ed_index = np.argmin(np.abs(ed_l - pos))
        ed_type = df_tad_cons['ed_region_type'][ed_index]
        if dist_st <= dist_ed:
            dist_l.append(dist_st)
            index_l.append(st_index)
            st_ed_l.append('st')
            bd_t_l.append(st_type)
        else:
            dist_l.append(dist_ed)
            index_l.append(ed_index)
            st_ed_l.append('ed')
            bd_t_l.append(ed_type)
    df_lr_bd['distance'] = dist_l
    df_lr_bd['domain_index'] = index_l
    df_lr_bd['st_ed'] = st_ed_l
    df_lr_bd['bd_type'] = bd_t_l
    return df_lr_bd


df_tad_cons = copy.deepcopy(TAD_signal_result['Consensus'])

## change format
df_tad_cons_copy = copy.deepcopy(df_tad_cons)
ed_l = []
for i in range(len(df_tad_cons_copy) - 1):
    st = df_tad_cons_copy['start'][i]
    ed = df_tad_cons_copy['end'][i]
    st_n = df_tad_cons_copy['start'][i+1]
    if ed > st_n:
        ed_l.append(ed - resolution)
    else:
        ed_l.append(ed)
ed_l.append(df_tad_cons_copy['end'][len(df_tad_cons_copy)-1])
df_tad_cons_copy['end'] = ed_l

df_domain_info = get_domain_partion(df_tad_cons_copy, df_human_chr_bio_GM12878, resolution)

random_ratio = []
for i in range(1000):
    if i != 0 and i % 200 == 0:
        print('200 random done!')
    #df_domain_random = get_random_domain_by_shullfe_info(df_domain_info, df_human_chr_bio_GM12878, Chr = 'chr2')    
    df_domain_random = get_random_domain_by_sample_info(df_domain_info, df_human_chr_bio_GM12878, Chr = 'chr2')    
    df_lr_bd_random = get_dist_of_LR_to_bd_random_domain(df_domain_random, lr_change_point)
    df_lr_bd_part_random = df_lr_bd_random[df_lr_bd_random['distance'] <= 1]
    random_ratio.append(len(df_lr_bd_part_random) / len(df_lr_bd_random))


df_lr_bd = get_dist_of_LR_to_bd(df_tad_cons, lr_change_point)

df_lr_bd_part = df_lr_bd[df_lr_bd['distance'] <= 1]
print(len(df_lr_bd_part))
print(len(df_lr_bd_part) / len(df_lr_bd))
for bd_type in ['sharp_weak', 'sharp_strong', 'wide']:
    print(bd_type)
    print(np.sum(df_lr_bd_part['bd_type'] == bd_type))
    print(np.sum(df_lr_bd_part['bd_type'] == bd_type) / len(df_lr_bd_part))

df_ratio = pd.DataFrame(columns = ['ratio', 'type'])
ratio_l = [len(df_lr_bd_part) / len(df_lr_bd)]
type_l = ['domain']
ratio_l += random_ratio
type_l += ['random' for i in range(len(random_ratio))]

df_ratio['ratio'] = ratio_l 
df_ratio['type'] = type_l

plt.figure(figsize=(4, 4))
sns.barplot(x = 'type', y = 'ratio', data = df_ratio)




######### repli-seq dealing

def kmeans_Score_with_max_Cnum(df_domain_repli_signal, K, save_name = ''):
    X = np.array(df_domain_repli_signal[list(range(0,500))])
    #X = np.array(df_domain_repli_signal[list(range(100,600))])
    Scores = [] 
    for k in range(2, K):
        cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward', compute_distances = True)
        model = cluster.fit(X)
        labels = cluster.fit_predict(X)    
        Scores.append(silhouette_score(X, labels, metric='euclidean'))       
    print(Scores)
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
    #plt.close(fig)

def plot_dendrogram(model):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    plt.figure(figsize=(10, 7))
    shc.dendrogram(linkage_matrix, **kwargs)

def get_cluster_change_point(df_domain_repli_signal):
    point_c = []
    for i in range(len(df_domain_repli_signal)-1):
        lb = df_domain_repli_signal['label'][i]
        lb_n = df_domain_repli_signal['label'][i+1]
        if lb != lb_n:
            point_c.append(i)
    return point_c

def check_cluster(dend, df_domain_repli_signal, cluster_color):
    link_color = np.array(dend['color_list'])
    for cluster_lb in list(cluster_color.keys()):
        print('For ' + cluster_lb)
        c_color = cluster_color[cluster_lb]
        print('Link number:')
        print(np.sum(link_color == c_color))
        print('Label number:')
        print(np.sum(df_domain_repli_signal['label'] == int(cluster_lb)))
            
def stats_bd_num_ratio(df_domain_repli_signal):
    df_stats = pd.DataFrame(columns = ['label', 'st_num', 'st_ratio', 'ed_num', 'ed_ratio'])
    lb_l = []
    st_num = []
    st_ratio = []
    ed_num = []
    ed_ratio = []
    total_num = []
    total_ratio = []
    for lb in np.unique(df_domain_repli_signal['label']):
        print(lb)
        lb_l.append(lb)
        df_domain_repli_signal_part = df_domain_repli_signal[df_domain_repli_signal['label'] == lb]    
        st_num_t = []
        st_ratio_t = []
        ed_num_t = []
        ed_ratio_t = []
        total_num_t = []
        total_ratio_t = []
        for bd_type in ['sharp_weak', 'sharp_strong', 'wide']:
            bd_num_st = np.sum(df_domain_repli_signal_part['st_bd_type'] == bd_type)
            st_num_t.append(bd_num_st)
            st_ratio_t.append(bd_num_st / len(df_domain_repli_signal_part))
            
            bd_num_ed = np.sum(df_domain_repli_signal_part['ed_bd_type'] == bd_type)
            ed_num_t.append(bd_num_ed)
            ed_ratio_t.append(bd_num_ed / len(df_domain_repli_signal_part))       
            
            bd_num_all = bd_num_st + bd_num_ed
            total_num_t.append(bd_num_all)
            total_ratio_t.append(bd_num_all / (2*len(df_domain_repli_signal_part)))
            
        st_num.append(st_num_t)
        st_ratio.append(st_ratio_t)   
        ed_num.append(ed_num_t)
        ed_ratio.append(ed_ratio_t)
        total_num.append(total_num_t)
        total_ratio.append(total_ratio_t)
            
    df_stats['label'] = lb_l        
    df_stats['st_num'] = st_num
    df_stats['st_ratio'] = st_ratio
    df_stats['ed_num'] = ed_num
    df_stats['ed_ratio'] = ed_ratio
    df_stats['total_num'] = total_num
    df_stats['total_ratio'] = total_ratio
    return df_stats
    

## load data from ubuntu
df_domain_repli_signal = read_save_data(r'E:\Users\dcdang\share\TAD_integrate\Epi_signal_in_domain' + '/' + 'GM12878_chr2_repli_original_signal.pkl')
#df_domain_repli_signal = read_save_data(r'E:\Users\dcdang\share\TAD_integrate\Epi_signal_in_domain' + '/' + 'GM12878_chr2_repli_signal.pkl')

df_domain_repli_signal['index'] = list(range(len(df_domain_repli_signal)))
df_domain_repli_signal['st_bd_type'] = df_tad_cons['st_region_type']
df_domain_repli_signal['ed_bd_type'] = df_tad_cons['ed_region_type']

X = np.array(df_domain_repli_signal[list(range(100,601))])

## explore cluster number
K = 12    
kmeans_Score_with_max_Cnum(df_domain_repli_signal, K, save_name = '')    


## dendrogram plot to decide cluster number
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist,squareform

color_link = ['#3A4CC0', '#7C9FF9', '#EDCBBA', '#E9795E', '#BC1F2C' ]
shc.set_link_color_palette(color_link)
plt.figure(figsize=(9, 4))
plt.title("Customer Dendograms")
with plt.rc_context({'lines.linewidth': 2.5}):
    dend = shc.dendrogram(shc.linkage(X, method='ward', metric='euclidean'), 
                          above_threshold_color='#AAAAAA', no_labels = True, 
                          color_threshold=2000)
#plt.ylim([-1000, 16000])
#plt.xlim([-300, 6100])
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.6)
ax.spines['left'].set_linewidth(1.6)
ax.spines['right'].set_linewidth(1.6)
ax.spines['top'].set_linewidth(1.6)
ax.tick_params(axis = 'y', length=7, width = 1.6)
ax.tick_params(axis = 'x', length=3, width = 1.6)
plt.savefig(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/heatmap/Cluster_Dendograms_domain_repli.svg', 
            format = 'svg', dpi = 600, transparent = True) 


#row_linkage = shc.linkage(X, method='ward', metric='euclidean')

cluster_color = {'0':'#EDCBBA' , '1':'#7C9FF9', 
                 '2':'#BC1F2C', '3':'#3A4CC0', 
                 '4':'#E9795E'}

## Agglomerative Clustering with fix cluster number
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward', compute_distances = True)
#cluster = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='euclidean', linkage='ward')
model = cluster.fit(X)
labels = cluster.fit_predict(X)

# sort domain by label
df_domain_repli_signal['label'] = labels
df_domain_repli_signal = df_domain_repli_signal.sort_values(by = ['label'])
df_domain_repli_signal = df_domain_repli_signal.reset_index(drop = True)

check_cluster(dend, df_domain_repli_signal, cluster_color)
   
# sort domain by fixed label
list_sorted = [2, 4, 0, 1, 3]
df_domain_repli_signal['label'] = df_domain_repli_signal['label'].astype('category').cat.set_categories(list_sorted)
df_domain_repli_signal = df_domain_repli_signal.sort_values(by=['label'])
df_domain_repli_signal = df_domain_repli_signal.reset_index(drop = True)

# sort by dene leaves, not good, no use
list_sorted = np.array(dend['leaves'])
df_domain_repli_signal['index'] = df_domain_repli_signal['index'].astype('category').cat.set_categories(list_sorted)
df_domain_repli_signal = df_domain_repli_signal.sort_values(by=['index'])
df_domain_repli_signal = df_domain_repli_signal.reset_index(drop = True)

## heatmap plot
point_c = get_cluster_change_point(df_domain_repli_signal) 

plt.figure(figsize=(10, 10))
heatmap_plot = np.array(df_domain_repli_signal[list(range(100, 601))])
plt.imshow(heatmap_plot, cmap = 'coolwarm', vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85))
for pos in point_c:
    plt.hlines(pos, 0, 500, linewidth = 2, linestyle = '--')
plt.colorbar()
plt.yticks([])


color_bd_use = {'sharp_strong':'#D65F4D', 'sharp_weak':'#459457', 'wide':'#4392C3'}
st_color = []
ed_color = []
lb_color = []
for i in range(len(df_domain_repli_signal)):
    st_type = df_domain_repli_signal['st_bd_type'][i]
    ed_type = df_domain_repli_signal['ed_bd_type'][i]
    lb = df_domain_repli_signal['label'][i]
    st_color.append(color_bd_use[st_type])
    ed_color.append(color_bd_use[ed_type])
    lb_color.append(cluster_color[str(lb)])
df_domain_repli_signal['st_color'] = st_color
df_domain_repli_signal['ed_color'] = ed_color
df_domain_repli_signal['label_color'] = lb_color


def sort_domain_by_bd_type(df_domain_repli_signal):
    bd_label_order = ['wide', 'sharp_strong', 'sharp_weak']
    domain_sort = []
    for lb in [2, 4, 0, 1, 3]:
        df_domain_part = copy.deepcopy(df_domain_repli_signal[df_domain_repli_signal['label'] == lb])
        df_domain_part['st_bd_type'] = df_domain_part['st_bd_type'].astype('category').cat.set_categories(bd_label_order)
        df_domain_part['ed_bd_type'] = df_domain_part['ed_bd_type'].astype('category').cat.set_categories(bd_label_order)
        df_domain_part = df_domain_part.sort_values(by=['st_bd_type', 'ed_bd_type'])
        df_domain_part = df_domain_part.reset_index(drop = True)
        domain_sort.append(df_domain_part)
    df_domain_sort = domain_sort[0]
    for i in range(1, len(domain_sort)):
        df_domain_sort = pd.concat([df_domain_sort, domain_sort[i]], axis = 0)
    df_domain_sort = df_domain_sort.reset_index(drop = True)
    return df_domain_sort


def sort_domain_by_mean_value(df_domain_repli_signal):
    domain_sort = []
    for lb in [2, 4, 0, 1, 3]:
        df_domain_part = copy.deepcopy(df_domain_repli_signal[df_domain_repli_signal['label'] == lb])
        df_domain_part['mean_value'] = np.mean(df_domain_part[list(range(100, 601))], axis = 1)   
        df_domain_part = df_domain_part.sort_values(by=['mean_value'], ascending=False)
        df_domain_part = df_domain_part.reset_index(drop = True)
        domain_sort.append(df_domain_part)    
    df_domain_sort = domain_sort[0]
    for i in range(1, len(domain_sort)):
        df_domain_sort = pd.concat([df_domain_sort, domain_sort[i]], axis = 0)
    df_domain_sort = df_domain_sort.reset_index(drop = True)
    return df_domain_sort

def sort_domain_by_var(df_domain_repli_signal):
    domain_sort = []
    for lb in [2, 4, 0, 1, 3]:
        df_domain_part = copy.deepcopy(df_domain_repli_signal[df_domain_repli_signal['label'] == lb])
        df_domain_part['variation'] = np.std(df_domain_part[list(range(100, 601))], axis = 1)   
        df_domain_part = df_domain_part.sort_values(by=['variation'], ascending=True)
        df_domain_part = df_domain_part.reset_index(drop = True)
        domain_sort.append(df_domain_part)    
    df_domain_sort = domain_sort[0]
    for i in range(1, len(domain_sort)):
        df_domain_sort = pd.concat([df_domain_sort, domain_sort[i]], axis = 0)
    df_domain_sort = df_domain_sort.reset_index(drop = True)
    return df_domain_sort

def sort_domain_by_head_tail_fold(df_domain_repli_signal):
    domain_sort_new = []
    for lb in [2, 4, 0, 1, 3]:
        df_domain_sort_part = copy.deepcopy(df_domain_repli_signal[df_domain_repli_signal['label'] == lb])
        df_domain_sort_part = df_domain_sort_part.reset_index(drop = True)
        fold_l = []
        for i in range(len(df_domain_sort_part)):
            head = np.mean(df_domain_sort_part[list(range(100, 350))].iloc[i])
            tail = np.mean(df_domain_sort_part[list(range(350, 601))].iloc[i])
            fold = head / tail
            fold_l.append(fold)
        df_domain_sort_part['fold'] = fold_l
        df_domain_sort_part = df_domain_sort_part.sort_values(by = ['fold'], ascending = False)
        domain_sort_new.append(df_domain_sort_part)
    df_domain_sort_new = domain_sort_new[0]
    for i in range(1, len(domain_sort_new)):
        df_domain_sort_new = pd.concat([df_domain_sort_new, domain_sort_new[i]], axis = 0)
    df_domain_sort_new = df_domain_sort_new.reset_index(drop = True)
    return df_domain_sort_new



df_domain_sort = sort_domain_by_bd_type(df_domain_repli_signal)
df_domain_sort = sort_domain_by_mean_value(df_domain_repli_signal)
df_domain_sort = sort_domain_by_var(df_domain_repli_signal)
df_domain_sort = sort_domain_by_head_tail_fold(df_domain_repli_signal)


heatmap_plot = np.array(df_domain_sort[list(range(100, 601))])
plt.figure(figsize=(8, 8))
plt.imshow(heatmap_plot, cmap = 'coolwarm', vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85))
for pos in point_c:
    plt.hlines(pos, 0, 500, linewidth = 5, linestyle = '--')



save_add = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/heatmap'

row_colors = df_domain_sort['st_color']
row_colors2 = df_domain_sort['ed_color']
lb_color_row = df_domain_sort['label_color']
heatmap_plot = np.array(df_domain_sort[list(range(100, 601))])

save_name = save_add + '/' + 'heatmap_row_color_st.png'
sns.clustermap(heatmap_plot, row_cluster = False, col_cluster = False,
               row_colors=[row_colors], method='ward', 
               metric='euclidean', cmap = 'coolwarm', 
               vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85), 
               yticklabels=False, xticklabels=False)
plt.savefig(save_name, format = 'png', dpi = 600, transparent = True) 

save_name = save_add + '/' + 'heatmap_row_color_ed.png'
sns.clustermap(heatmap_plot, row_cluster = False, col_cluster = False,
               row_colors=[row_colors2], method='ward', 
               metric='euclidean', cmap = 'coolwarm', 
               vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85), 
               yticklabels=False, xticklabels=False)
plt.savefig(save_name, format = 'png', dpi = 600, transparent = True) 

save_name = save_add + '/' + 'heatmap_row_color_st_ed.png'
sns.clustermap(heatmap_plot, row_cluster = False, col_cluster = False,
               row_colors=[row_colors, row_colors2], method='ward', 
               metric='euclidean', cmap = 'coolwarm', 
               vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85), 
               yticklabels=False, xticklabels=False)
plt.savefig(save_name, format = 'png', dpi = 600, transparent = True) 

save_name = save_add + '/' + 'heatmap_row_color_label_color.png'
sns.clustermap(heatmap_plot, row_cluster = False, col_cluster = False,
               row_colors=[lb_color_row], method='ward', 
               metric='euclidean', cmap = 'coolwarm', 
               vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85), 
               yticklabels=False, xticklabels=False)
plt.savefig(save_name, format = 'png', dpi = 600, transparent = True) 

save_name = save_add + '/' + 'heatmap_cluster_domain.svg'
plt.figure(figsize=(10, 10))
plt.imshow(heatmap_plot, cmap = 'coolwarm', vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85))
for pos in point_c:
    plt.hlines(pos, 0, 500, linewidth = 5, linestyle = '--')
plt.colorbar()
plt.yticks([])
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.6)
ax.spines['left'].set_linewidth(1.6)
ax.spines['right'].set_linewidth(1.6)
ax.spines['top'].set_linewidth(1.6)
ax.tick_params(axis = 'y', length=7, width = 1.6)
ax.tick_params(axis = 'x', length=3, width = 1.6)   
plt.savefig(save_name, format = 'svg', transparent = True) 



# draw mean value scatter plot
cluster_order = [2, 4, 0, 1, 3]
plt.figure(figsize=(12, 2))
pos_st = 0
for pos in point_c:
    ind = point_c.index(pos)
    color = cluster_color[str(cluster_order[ind])]
    value = np.mean(heatmap_plot, axis = 1)[pos_st:pos +1]
    index = list(range(pos_st, pos+1))
    plt.scatter(index, value, c = color)
    pos_st = pos + 1
value = np.mean(heatmap_plot, axis = 1)[pos_st:len(heatmap_plot)]
index = list(range(pos_st, len(heatmap_plot)))
color = cluster_color[str(cluster_order[-1])]
plt.scatter(index, value, c = color)
plt.xlim(-20, 620)
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.6)
ax.spines['left'].set_linewidth(1.6)
ax.spines['right'].set_linewidth(1.6)
ax.spines['top'].set_linewidth(1.6)
ax.tick_params(axis = 'y', length=7, width = 1.6)
ax.tick_params(axis = 'x', length=3, width = 1.6) 
plt.savefig(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/heatmap/domain_mean_repli_plot.svg', 
            format = 'svg', transparent = True) 


def draw_cluster_mean_value(df_domain_repli_signal, bio_t = 'mean', save_name = ''):
    heatmap_plot_use = np.array(df_domain_repli_signal[list(range(100, 601))])
    lb_l = list(df_domain_repli_signal['label'])
    if bio_t == 'mean':
        value_l = np.mean(heatmap_plot_use, axis = 1)
    elif bio_t == 'var':
        value_l = np.std(heatmap_plot_use, axis = 1)        
    df_v_lb = pd.DataFrame(columns = ['label', 'value'])
    df_v_lb['label'] = lb_l
    df_v_lb['value'] = value_l
    color_use = []
    for i in range(len(df_domain_repli_signal)):
        color = df_domain_repli_signal['label_color'][i]
        if color not in color_use:
            color_use.append(color)
    plt.figure(figsize=(5, 2))
    sns.barplot(x = 'label', y = 'value', data = df_v_lb, order = cluster_order, palette = color_use,
                capsize = 0.15, saturation = 1, errcolor = 'black', errwidth = 2, 
                ci = 95, linewidth = 2, edgecolor='black')
    plt.title(bio_t)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)     
    #save_name = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/Epi_mean' + '/' + bio_t + '_domain_mean_value.svg'
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True)

save_name = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli' + '/' \
    + 'Std_within_domain.svg'
draw_cluster_mean_value(df_domain_repli_signal, bio_t = 'var', save_name = save_name)

save_name = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli' + '/' \
    + 'Mean_within_domain.svg'
draw_cluster_mean_value(df_domain_repli_signal, bio_t = 'mean', save_name = save_name)



def draw_all_value_in_domain(df_domain_repli_signal, save_name = ''):
    df_repli_copy = copy.deepcopy(df_domain_repli_signal)
    df_biodata_pos = pd.DataFrame(columns = ['pos_l', 'value', 'cluster_label'])    
    pos_l = []
    value_l = []
    label_l = []
    heatmap_plot = []
    for i in range(len(df_repli_copy)):
        lb = df_repli_copy['label'][i]
        value = list(df_repli_copy[list(range(100, 601))].iloc[i])
        head = np.mean(df_repli_copy[list(range(100, 350))].iloc[i])
        tail = np.mean(df_repli_copy[list(range(350, 601))].iloc[i])
        fold = head / tail
        if fold < 1:
            #value_use = np.flip(np.array(value))
            value_use = value
        else:
            value_use = value        
        pos_l += list(range(len(value)))
        value_l += list(value_use)
        label_l += [lb for j in range(len(value))]
        heatmap_plot.append(value_use)
    df_biodata_pos['pos_l'] = pos_l
    df_biodata_pos['value'] = value_l
    df_biodata_pos['cluster_label'] = label_l   
    
    label_order = [2, 4, 0, 1, 3]       
    df_biodata_pos['cluster_label'] = df_biodata_pos['cluster_label'].astype('category')
    df_biodata_pos['cluster_label'].cat.reorder_categories(label_order, inplace=True)
    df_biodata_pos.sort_values('cluster_label', inplace=True)            
    cluster_color = {'0':'#EDCBBA' , '1':'#7C9FF9', 
                 '2':'#BC1F2C', '3':'#3A4CC0', 
                 '4':'#E9795E'}    
    color_l = [] 
    for lb in label_order:
        color_l.append(cluster_color[str(lb)])   
    plt.figure(figsize=(4,6))
    sns.lineplot(x = 'pos_l', y = 'value', data = df_biodata_pos, hue = 'cluster_label',  
                 style='cluster_label',  palette = color_l, markers=False, dashes=False, 
                 linewidth = 3, ci = 'sd', err_style = 'band', alpha  = 1, )   
    #plt.xticks([0, 5, 10, 15, 20], ['-500kb', '-250kb', 'boundary center', '250kb', '500kb'], FontSize = 10)
    plt.yticks(FontSize = 10)
    plt.title(bio_type)
    plt.ylabel(bio_type,  fontSize = 12)
    plt.xlabel('',  fontSize = 0)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    ax.legend_ = None
    #plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)  
    #plt.legend(loc = 'best', prop = {'size':10}, fancybox = None, edgecolor = 'white', facecolor = None, title = False, title_fontsize = 0)
    if save_name != '':    
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)   
    return heatmap_plot

save_name = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli' + '/' \
    + 'cluster_repli_profile.svg'
heatmap_plot = draw_all_value_in_domain(df_domain_repli_signal, save_name = save_name)
 
## adjust domain value from high to low from left to right with value_use = np.flip(np.array(value))
plt.figure(figsize=(10, 10))
plt.imshow(np.array(heatmap_plot), cmap = 'coolwarm', vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85))
for pos in point_c:
    plt.hlines(pos, 0, 500, linewidth = 5, linestyle = '--')
plt.colorbar()
plt.yticks([])
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.6)
ax.spines['left'].set_linewidth(1.6)
ax.spines['right'].set_linewidth(1.6)
ax.spines['top'].set_linewidth(1.6)
ax.tick_params(axis = 'y', length=7, width = 1.6)
ax.tick_params(axis = 'x', length=3, width = 1.6)



df_stats = stats_bd_num_ratio(df_domain_repli_signal)

def draw_pie_plot_of_domain_bd_type(df_stats, target = 'total', save_add = ''):  
    if target == 'Start':
        target_col = 'st_num'
    elif target == 'End':
        target_col = 'ed_num'
    elif target == 'total':
        target_col = 'total_num'
    for i in range(len(df_stats)):
        num_list = df_stats[target_col][i]
        label_list = ['sharp_weak', 'sharp_strong', 'wide']
        color_use = ['#459457', '#D65F4D', '#4392C3']
        plt.figure(figsize= (7, 5))       
        patches,l_text,p_text = plt.pie(num_list, explode=None, labels=None, colors=color_use, autopct="%0.1f%%", pctdistance=0.6, 
                                        shadow=False, labeldistance=1.5, startangle=None, radius=None, counterclock=True, 
                                        wedgeprops={'linewidth': 2, 'edgecolor': "black"}, textprops=None, center=(0, 0), frame=False, rotatelabels=False, data=None)    
        plt.legend(labels=label_list , markerscale = 2, fontsize = 14, frameon = False, bbox_to_anchor=(0.9, 0.9))
        plt.title(target + ' bd of type ' + str(i))
        for t in l_text:
            t.set_size(0)
        for t in p_text:
            t.set_size(14)
        if save_add != '':
            save_name = save_add + '/' + target + ' bd of type ' + str(i) + '.svg'
        else:
            save_name = ''
        if save_name != '':    
            plt.savefig(save_name, format = 'svg', transparent = True) 
        plt.show()
        fig = plt.gcf() #获取当前figure
        plt.close(fig)   

save_add = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/boundary_ratio'

draw_pie_plot_of_domain_bd_type(df_stats, target = 'total', save_add = save_add)
    
draw_pie_plot_of_domain_bd_type(df_stats, target = 'Start', save_add = save_add)
 
draw_pie_plot_of_domain_bd_type(df_stats, target = 'End', save_add = save_add)
 



def draw_bar_plot_of_domain_bd_type(df_stats, cluster_ord, color_bd_use, save_name = '', target = 'total'):  
    if target == 'Start':
        target_col = 'st_ratio'
    elif target == 'End':
        target_col = 'ed_ratio'
    elif target == 'total':
        target_col = 'total_ratio'
 
    type_ord = ['sharp_weak','sharp_strong','wide', ]
    df_number = pd.DataFrame(columns = ['cluster_type', 'sharp_strong', 'sharp_weak', 'wide'])
    sharp_wl = []
    sharp_sl = []
    wide_l = [] 
    for cluster in cluster_ord:
        ind = list(df_stats['label']).index(cluster)
        sharp_wl.append(df_stats[target_col][ind][0])
        sharp_sl.append(df_stats[target_col][ind][1])
        wide_l.append(df_stats[target_col][ind][2])
    df_number['cluster_type'] = cluster_ord
    df_number['sharp_weak'] = sharp_wl
    df_number['sharp_strong'] = sharp_sl
    df_number['wide'] = wide_l
    
    plt.figure(figsize=(5,5))
    bottom_list = np.zeros(len(cluster_ord))
    for i in range(len(type_ord)):
        bd_type = type_ord[i]
        plt.bar(range(len(cluster_ord)), list(df_number[bd_type]), align="center", bottom=list(bottom_list), color=color_bd_use[bd_type], edgecolor = 'black', linewidth = 1.5)
        bottom_list += np.array(df_number[bd_type])
    
    plt.xticks(list(range(len(cluster_ord))), list(df_number['cluster_type']), rotation= -30, FontSize = 12)
    #plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'], FontSize = 12)
    #plt.ylim([0,1])
    plt.ylabel('Components of boundary type',  FontSize = 12)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)
    return df_number
    
cluster_order = [2, 4, 0, 1, 3]
color_bd_use = {'sharp_strong':'#D65F4D', 'sharp_weak':'#459457', 'wide':'#4392C3'}
save_name = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli' + '/' + 'boundary_ratio_for_repli_cluster_domain.svg'
draw_bar_plot_of_domain_bd_type(df_stats, cluster_order, color_bd_use, save_name = save_name, target = 'total')


###### domain with segway and subcompartment

#### segway result
     
def get_bd_region_state_FC_segway(domain_segway_state, region_type = [2, 4, 0, 1, 3]):               
    #state_all_order = ['QUI', 'CON', 'FAC', 'BRD', 'SPC']
    state_all_order = ['SPC', 'BRD', 'FAC', 'CON','QUI',]
    region_type_ord = region_type    
    df_bd_region_state_FC = pd.DataFrame(columns = region_type_ord)
    for r_type in region_type_ord:
        state_FC = []
        bd_region_state_dic = domain_segway_state[r_type]
        for state in state_all_order:
            state_FC.append(bd_region_state_dic[state])      
        df_bd_region_state_FC[r_type] = state_FC    
    df_bd_region_state_FC.index = state_all_order    
    sns.clustermap(df_bd_region_state_FC, figsize=(5, 5), center = 1, row_cluster=False, col_cluster=False, cmap = 'coolwarm')
    return df_bd_region_state_FC      
       
domain_segway_state = read_save_data(r'E:/Users/dcdang/share/TAD_integrate/Epi_signal_in_domain/domain_chrom_state' + '/' + 'GM12878_MboI_domain_region_segway_FC_result.pkl')
df_domain_region_segway_FC = get_bd_region_state_FC_segway(domain_segway_state, region_type = [2, 4, 0, 1, 3])
save_name5 = r'E:/Users/dcdang/share/TAD_integrate/Epi_signal_in_domain/domain_chrom_state/result' + '/' + 'GM12878_domain_segway_state_FC.bed'  
df_domain_region_segway_FC.to_csv(save_name5, sep = '\t', header = True, index = True)


### subcompartment enrich

def get_bd_region_state_FC_subcom(domain_subcompartment, region_type = [2, 4, 0, 1, 3]):               
    state_all_order = ['A1', 'A2', 'B1', 'B2', 'B3']
    region_type_ord = region_type    
    df_bd_region_state_FC = pd.DataFrame(columns = region_type_ord)
    for r_type in region_type_ord:
        state_FC = []
        bd_region_state_dic = domain_subcompartment[r_type]
        for state in state_all_order:
            state_FC.append(bd_region_state_dic[state])      
        df_bd_region_state_FC[r_type] = state_FC    
    df_bd_region_state_FC.index = state_all_order    
    sns.clustermap(df_bd_region_state_FC, figsize=(5, 5), center = 1, row_cluster=False, col_cluster=False, cmap = 'coolwarm')
    return df_bd_region_state_FC      
    
  
domain_subcompartment = read_save_data(r'E:/Users/dcdang/share/TAD_integrate/Epi_signal_in_domain/domain_chrom_state' + '/' + 'GM12878_MboI_domain_region_subcompartment_FC_result.pkl')
df_domain_region_subc_FC = get_bd_region_state_FC_subcom(domain_subcompartment, region_type = [2, 4, 0, 1, 3])
save_name6 = r'E:/Users/dcdang/share/TAD_integrate/Epi_signal_in_domain/domain_chrom_state/result' + '/' + 'GM12878_domain_subcompartment_state_FC.bed'  
df_domain_region_subc_FC.to_csv(save_name6, sep = '\t', header = True, index = True)


###### chromHMM enrich

def get_bd_region_state_FC_chrom(domain_chromhmm_state, region_type = [2, 4, 0, 1, 3]):               
    state_all_order = ['A1', 'A2', 'B1', 'B2', 'B3']    
    state_all = list(domain_chromhmm_state[region_type[0]].keys())
    state_index = []
    for state in state_all:        
        state_index.append(int(state.split('_')[0]))        
    index_ord = np.argsort(state_index)
    state_all_order = []
    for index in index_ord:
        state_all_order.append(state_all[index])        
    region_type_ord = region_type    
    df_bd_region_state_FC = pd.DataFrame(columns = region_type_ord)
    for r_type in region_type_ord:
        state_FC = []
        bd_region_state_dic = domain_chromhmm_state[r_type]
        for state in state_all_order:
            state_FC.append(bd_region_state_dic[state])      
        df_bd_region_state_FC[r_type] = state_FC    
    df_bd_region_state_FC.index = state_all_order    
    sns.clustermap(df_bd_region_state_FC, figsize=(5, 5), center = 1, row_cluster=False, col_cluster=False, cmap = 'coolwarm')
    return df_bd_region_state_FC      
    
  
domain_chromhmm_state = read_save_data(r'E:/Users/dcdang/share/TAD_integrate/Epi_signal_in_domain/domain_chrom_state' + '/' + 'GM12878_MboI_domain_region_chromhmm15_FC_result.pkl')
df_domain_region_subc_FC = get_bd_region_state_FC_chrom(domain_chromhmm_state, region_type = [2, 4, 0, 1, 3])
save_name7 = r'E:/Users/dcdang/share/TAD_integrate/Epi_signal_in_domain/domain_chrom_state/result' + '/' + 'GM12878_domain_chromhmm_state_FC.bed'  
df_domain_region_subc_FC.to_csv(save_name7, sep = '\t', header = True, index = True)




def get_bd_region_state_FC(cell_type, enzyme, bd_region_state_15_cell_all, region_type = ['sharp_strong', 'sharp_weak', 'wide']):        
    cell_FC_result = bd_region_state_15_cell_all[cell_type + '_' + enzyme]    
    state_all = list(cell_FC_result['wide'].keys())
    state_index = []
    for state in state_all:        
        state_index.append(int(state.split('_')[0]))        
    index_ord = np.argsort(state_index)
    state_all_order = []
    for index in index_ord:
        state_all_order.append(state_all[index])
    region_type_ord = ['wide', 'sharp_strong', 'sharp_weak']
    df_bd_region_state_FC = pd.DataFrame(columns = region_type_ord)
    for r_type in region_type_ord:
        state_FC = []
        bd_region_state_dic = cell_FC_result[r_type]
        for state in state_all_order:
            state_FC.append(bd_region_state_dic[state])      
        df_bd_region_state_FC[r_type] = state_FC    
    df_bd_region_state_FC.index = state_all_order    
    #sns.clustermap(df_bd_region_state_FC, center = 0, row_cluster=False, col_cluster=False, cmap = 'coolwarm')
    return df_bd_region_state_FC      
    
    
save_add = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\State_enrich\ChromHMM'

for cell_type in ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'K562', 'NHEK']:
    enzyme = 'MboI'
    
    df_bd_region_15_state_FC = get_bd_region_state_FC(cell_type, enzyme, bd_region_state_15_cell_all, region_type = ['sharp_strong', 'sharp_weak', 'wide'])  
    save_name15 = save_add + '/' + cell_type + '_' + enzyme + '_chromHMM_15_state_FC.bed'
    df_bd_region_15_state_FC.to_csv(save_name15, sep = '\t', header = True, index = True)

    df_bd_region_18_state_FC = get_bd_region_state_FC(cell_type, enzyme, bd_region_state_18_cell_all, region_type = ['sharp_strong', 'sharp_weak', 'wide'])  
    save_name18 = save_add + '/' + cell_type + '_' + enzyme + '_chromHMM_18_state_FC.bed'
    df_bd_region_18_state_FC.to_csv(save_name18, sep = '\t', header = True, index = True)



### compare with method

repli_domain_all_method = read_save_data(r'E:/Users/dcdang/share/TAD_integrate/Epi_signal_in_domain/TAD_repli_domain_all.pkl')

TAD_result_all_GM12878_MboI = read_save_data(r'E:\Users\dcdang\share\TAD_integrate\Epi_signal_in_domain' + '/' + 'TAD_result_all.pkl')



for method in list(repli_domain_all_method.keys()):
    df_domain_repli_m = copy.deepcopy(repli_domain_all_method[method])

    point_c = get_cluster_change_point(df_domain_repli_m)         
    plt.figure(figsize=(5, 5))
    heatmap_plot = np.array(df_domain_repli_m[list(range(100, 601))])
    #plt.imshow(heatmap_plot, cmap = 'coolwarm', vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85))
    plt.imshow(heatmap_plot, cmap = 'coolwarm', vmin = 19.27041512997945, vmax = 73.33538607279459)
    
    for pos in point_c:
        plt.hlines(pos, 0, 500, linewidth = 2, linestyle = '--')
    plt.colorbar()
    plt.yticks([])
    plt.title(method)



old_order = [2,4,0,1,3]
cluster_color = {'0':'#EDCBBA' , '1':'#7C9FF9', 
                 '2':'#BC1F2C', '3':'#3A4CC0', 
                 '4':'#E9795E'}

cluster_color_use = {}
for i in range(5):
    cluster_color_use[i] = cluster_color[str(old_order[i])]

for method in list(repli_domain_all_method.keys()):
    df_domain_repli_m = copy.deepcopy(repli_domain_all_method[method])
    point_c = get_cluster_change_point(df_domain_repli_m) 
    heatmap_plot = np.array(df_domain_repli_m[list(range(100, 601))])
    ord_use = []
    for i in range(len(df_domain_repli_m)):
        lb = df_domain_repli_m['label'][i]
        if lb not in ord_use:
            ord_use.append(lb) 
    cluster_order = ord_use
    cluster_color_use = {}   
    for j in range(len(ord_use)):
        lb = ord_use[j]
        lb_old = old_order[j]
        cluster_color_use[str(lb)] = cluster_color[str(lb_old)]
    plt.figure(figsize=(12, 2))
    pos_st = 0
    for pos in point_c:
        ind = point_c.index(pos)
        color = cluster_color_use[str(cluster_order[ind])]
        value = np.mean(heatmap_plot, axis = 1)[pos_st:pos +1]
        index = list(range(pos_st, pos+1))
        plt.scatter(index, value, c = color)
        pos_st = pos + 1
    value = np.mean(heatmap_plot, axis = 1)[pos_st:len(heatmap_plot)]
    index = list(range(pos_st, len(heatmap_plot)))
    color = cluster_color_use[str(cluster_order[-1])]
    plt.scatter(index, value, c = color)
    #plt.xlim(-20, 620)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6) 
    plt.title(method)


def draw_domain_number(repli_domain_all_method, cluster_color_use, save_name = ''):   
    df_domain_number = pd.DataFrame(columns = [0, 1, 2, 3, 4, 'method', 'number_all'])
    for i in range(5):
        num_l = []
        for method in list(repli_domain_all_method.keys()):
            df_domain_repli_m = copy.deepcopy(repli_domain_all_method[method])
            #heatmap_plot = np.array(df_domain_repli_m[list(range(100, 601))])
            ord_use = []
            for j in range(len(df_domain_repli_m)):
                lb = df_domain_repli_m['label'][j]
                if lb not in ord_use:
                    ord_use.append(lb)
            lb_target = ord_use[i]
            df_domain_repli_m_part = df_domain_repli_m[df_domain_repli_m['label'] == lb_target]
            num_l.append(len(df_domain_repli_m_part))
        df_domain_number[i] = num_l
    df_domain_number['method'] = list(repli_domain_all_method.keys())
    num_all = []
    for method in list(repli_domain_all_method.keys()):
        df_domain_repli_m = copy.deepcopy(repli_domain_all_method[method])
        num_all.append(len(df_domain_repli_m))          
    df_domain_number['number_all'] = num_all
    df_domain_number = df_domain_number.sort_values(by = ['number_all'], ascending = False)
    df_domain_number = df_domain_number.reset_index(drop = True)
      
    plt.figure(figsize=(6, 3))
    bottom_list = np.zeros(len(df_domain_number))
    for i in range(5):
        plt.bar(range(len(bottom_list)), list(df_domain_number[i]), align="center", bottom=list(bottom_list), color=cluster_color_use[i], edgecolor = 'black', linewidth = 1.5)
        bottom_list += np.array(df_domain_number[i])   
    plt.xticks(list(range(len(bottom_list))), list(df_domain_number['method']), rotation= -30, FontSize = 8)
    #plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'], FontSize = 12)
    #plt.ylim([0,1])
    plt.ylabel('Domain number of replication cluster',  FontSize = 8)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)
    return df_domain_number

save_name = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/compare' + '/' + 'repli_domain_number_compare.svg'
df_domain_number = draw_domain_number(repli_domain_all_method, cluster_color_use, save_name = save_name)
    


bin_num = len(mat_dense)
domain_label_all = []
for method in list(TAD_result_all_GM12878_MboI.keys()):
    domain_label = np.zeros(bin_num)
    df_result = copy.deepcopy(repli_domain_all_method[method])
    df_tad = copy.deepcopy(TAD_result_all_GM12878_MboI[method]['TAD_domain'])
    ord_use = []
    for i in range(len(df_result)):
        lb = df_result['label'][i]
        if lb not in ord_use:
            ord_use.append(lb)
    for i in range(len(df_result)):
        ind = df_result['index'][i] 
        lb = df_result['label'][i]
        lb_ind = ord_use.index(lb)
        st = df_tad['start'][ind]
        ed = df_tad['end'][ind]
        start = int(st / resolution)
        end = int(ed / resolution) - 1
        domain_label[start:end+1] = 4 - lb_ind + 1
        #domain_label[start:end+1] = lb_ind + 1
    domain_label_all.append(list(domain_label))

df_domain_label = pd.DataFrame(np.array(domain_label_all))
df_domain_label['method'] = list(TAD_result_all_GM12878_MboI.keys())



def draw_consistency_of_repli_domain_type(df_domain_label, method_color, save_name = ''):
    score_all = {}
    mean_value = []
    for method in list(df_domain_label['method']):
        print('This is ' + method)
        df_domain_label_part = df_domain_label[df_domain_label['method'] == method]
        df_domain_label_part = df_domain_label_part.reset_index(drop = True)
        score_lb = []
        for i in range(bin_num):
            lb = df_domain_label_part[i][0]
            if lb == 0:
                score = 0
            else:
                count = list(df_domain_label[i]).count(lb)
                score = count / len(list(df_domain_label[i]))
            score_lb.append(score)
        print(np.mean(score_lb))
        mean_value.append(np.mean(score_lb))
        score_all[method] = score_lb       
    order = np.argsort(-np.array(mean_value))
    method_ord = []
    color_use = []
    for ind in order:
        method_current = list(df_domain_label['method'])[ind]
        method_ord.append(method_current)
        if method_current != 'Consensus':
            color_use.append(method_color[method_current])
        elif method_current == 'Consensus':
            color_use.append('#3F85FF')
    df_score = pd.DataFrame(columns = ['score', 'method'])
    score_l = []
    m_l = []
    for method in list(df_domain_label['method']):
        score_l += score_all[method]
        m_l += [method for i in range(len(score_all[method]))]   
    df_score['score'] = score_l 
    df_score['method'] = m_l

    plt.figure(figsize=(6, 3))
    sns.barplot(x = 'method', y = 'score', data = df_score, order = method_ord, capsize = 0.2, saturation = 8,             
                errcolor = 'black', errwidth = 1.5, ci = 95, edgecolor='black', palette = color_use)    
    #plt.legend(loc = 'upper left', fontsize = 12)
    plt.xlabel('Method',  FontSize = 0)
    plt.ylabel('Replication consistency score',  FontSize = 8)
    plt.xticks(FontSize = 8, rotation = -30)
    plt.yticks(FontSize = 8)
    #plt.ylim([0,40])
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)    
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)
    return method_ord

save_name = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/compare' + '/' + 'repli_domain_consistency_score_compare.svg'
method_ord = draw_consistency_of_repli_domain_type(df_domain_label, method_color, save_name = save_name)



def draw_domain_type_all_method_along_chr(st, ed, df_domain_label, method_ord, save_name = '', bin_inter = 20, resolution = 50000):
    x_l = []
    x_name = []
    for i in range(0, ed-st+1):
        if i % bin_inter == 0:
            x_l.append(i)
            pos = (st + i) * resolution / 1000000
            x_name.append(str(pos) + 'Mb')
    cmap = ['#B0B0B0', '#3A4CC0', '#7C9FF9', '#EDCBBA', '#E9795E', '#BC1F2C']
    my_cmap = ListedColormap(cmap)
    bounds=[0, 0.9, 1.9, 2.9, 3.9, 4.9, 5.9]
    norm = matplotlib.colors.BoundaryNorm(bounds, my_cmap.N)
    
    df_domain_label_sort = copy.deepcopy(df_domain_label)
    df_domain_label_sort['method'] = df_domain_label_sort['method'].astype('category').cat.set_categories(method_ord)
    df_domain_label_sort = df_domain_label_sort.sort_values(by=['method'])
    df_domain_label_sort = df_domain_label_sort.reset_index(drop = True)
    
    heatmap_plot = np.array(df_domain_label_sort[list(range(st, ed))])    
    plt.figure(figsize=(14, 3))
    plt.imshow(heatmap_plot,  cmap = my_cmap, Norm = norm)
    #sns.clustermap(heatmap_plot, row_cluster=False, col_cluster = False, cmap = my_cmap, Norm = norm, linewidth=0.2)
    #plt.colorbar()
    for i in range(len(heatmap_plot)):
        plt.hlines(i+0.5, 0-0.5, heatmap_plot.shape[1]-0.5, color = 'black')
    for i in range(heatmap_plot.shape[1]):
        plt.vlines(i-0.5, 0-0.5, heatmap_plot.shape[0]-1 + 0.5, color = 'black')
    plt.yticks(range(len(df_domain_label_sort['method'])), df_domain_label_sort['method'])
    plt.xticks(np.array(x_l) - 0.5, x_name, fontSize = 8)
    #plt.xlim([0, heatmap_plot.shape[1]-1])
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)   
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

st = 400
ed = 500
save_name = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/compare' + '/' + 'repli_domain_type_compare_case_all_method.svg'
draw_domain_type_all_method_along_chr(st, ed, df_domain_label, method_ord, save_name = save_name, bin_inter = 20, resolution = 50000)






record_repli_ori_all_stage = read_save_data(r'E:/Users/dcdang/share/TAD_integrate/Epi_signal_in_domain/GM12878_chr2_repli_all_original_signal.pkl')


for stage in list(record_repli_ori_all_stage.keys()):
    df_repli_stage = copy.deepcopy(record_repli_ori_all_stage[stage])
    df_repli_stage['index'] = list(range(len(df_repli_stage)))
    order_use = list(df_domain_sort['index'])
    df_repli_stage['index'] = df_repli_stage['index'].astype('category').cat.set_categories(order_use)
    df_repli_stage = df_repli_stage.sort_values(by=['index'])
    df_repli_stage = df_repli_stage.reset_index(drop = True)
    df_repli_stage['label'] = df_domain_sort['label']
    point_c = get_cluster_change_point(df_repli_stage) 
    heatmap_plot = np.array(df_repli_stage[list(range(100, 601))])
    plt.figure(figsize=(5, 4))
    #plt.imshow(heatmap_plot, cmap = 'coolwarm', vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85))
    plt.imshow(heatmap_plot, cmap = 'coolwarm', vmin = 0, vmax = 40)        
    for pos in point_c:
        plt.hlines(pos, 0, 500, linewidth = 5, linestyle = '--')
    plt.colorbar()
    plt.yticks([])
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)   
    plt.title(stage)
    save_name = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/compare' + '/' + stage + '_' + 'domain_cluster_signal.svg'
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    
        






### draw Agglomerative Clustering dendrogram
#plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
#plot_dendrogram(model, truncate_mode='level', p=4)
#plot_dendrogram(model)



## other epi

bio_data_list = ['H2A.Z', 'H3K4me3', 'H3K4me1', 'H3K4me2',
                  'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K9ac',
                  'H3K9me3', 'H3K79me2', 'CTCF', 'SMC3', 'RAD21',
                  'Pol2', 'YY1', 'DNA_methylation', 'DNase', 'Repli',
                  'RNA-', 'RNA+', 'RNA']

for bio_t in bio_data_list:
    df_domain_h3k27ac_signal = read_save_data(r'E:\Users\dcdang\share\TAD_integrate\Epi_signal_in_domain' + '/' + 'GM12878_chr2_' + bio_t + '_original_signal.pkl')    
    df_domain_h3k27ac_signal['index'] = list(range(len(df_domain_h3k27ac_signal)))    
    order_use = np.array(df_domain_sort['index'])
    df_domain_h3k27ac_signal['index'] = df_domain_h3k27ac_signal['index'].astype('category').cat.set_categories(order_use)
    df_domain_h3k27ac_signal = df_domain_h3k27ac_signal.sort_values(by=['index'])
    df_domain_h3k27ac_signal = df_domain_h3k27ac_signal.reset_index(drop = True)
    
    heatmap_plot_h3k27ac = np.array(df_domain_h3k27ac_signal[list(range(100, 601))])
    if bio_t == 'RNA-':
        heatmap_plot_h3k27ac = -heatmap_plot_h3k27ac
    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap_plot_h3k27ac, cmap = 'rainbow', vmin = np.percentile(heatmap_plot_h3k27ac, 20), vmax = np.percentile(heatmap_plot_h3k27ac, 80))
    for pos in point_c:
        plt.hlines(pos, 0, 500, linewidth = 5, linestyle = '--')
    plt.colorbar()
    plt.yticks([])
    plt.title(bio_t)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)   
    save_name = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/Epi_heatmap' + '/' + bio_t + '_domain_heatmap.svg'
    plt.savefig(save_name, format = 'svg', transparent = True)
   
    


cluster_order = [2, 4, 0, 1, 3]
color_use = []
for i in range(len(cluster_order)):
    color = cluster_color[str(cluster_order[i])]
    color_use.append(color)

for bio_t in bio_data_list:
    df_domain_h3k27ac_signal = read_save_data(r'E:\Users\dcdang\share\TAD_integrate\Epi_signal_in_domain' + '/' + 'GM12878_chr2_' + bio_t + '_original_signal.pkl')    
    df_domain_h3k27ac_signal['index'] = list(range(len(df_domain_h3k27ac_signal)))   
    order_use = np.array(df_domain_sort['index'])
    df_domain_h3k27ac_signal['index'] = df_domain_h3k27ac_signal['index'].astype('category').cat.set_categories(order_use)
    df_domain_h3k27ac_signal = df_domain_h3k27ac_signal.sort_values(by=['index'])
    df_domain_h3k27ac_signal = df_domain_h3k27ac_signal.reset_index(drop = True)   
    df_domain_h3k27ac_signal['label'] = copy.deepcopy(df_domain_sort['label'])
    heatmap_plot_h3k27ac = np.array(df_domain_h3k27ac_signal[list(range(100, 601))])
    if bio_t == 'RNA-':
        heatmap_plot_h3k27ac = - heatmap_plot_h3k27ac
    lb_l = list(df_domain_h3k27ac_signal['label'])
    value_l = np.mean(heatmap_plot_h3k27ac, axis = 1)
    # exist one very large value
    if bio_t == 'H3K27ac':
        ind = list(value_l).index(np.max(value_l))
        value_l[ind] = np.mean(value_l)
    df_v_lb = pd.DataFrame(columns = ['label', 'value'])
    df_v_lb['label'] = lb_l
    df_v_lb['value'] = value_l
    plt.figure(figsize=(5, 3))
    sns.barplot(x = 'label', y = 'value', data = df_v_lb, order = cluster_order, palette = color_use,
                capsize = 0.15, saturation = 1, errcolor = 'black', errwidth = 2, 
                ci = 95, linewidth = 2, edgecolor='black')
    plt.title(bio_t)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)     
    save_name = r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/Epi_mean' + '/' + bio_t + '_domain_mean_value.svg'
    plt.savefig(save_name, format = 'svg', transparent = True)


## type length
label_l = []
df_tad_cons_copy = copy.deepcopy(df_tad_cons)
for i in range(len(df_tad_cons_copy)):
    ind = list(df_domain_sort['index']).index(i)
    label_l.append(df_domain_sort['label'][ind])
df_tad_cons_copy['label'] = label_l

df_len = pd.DataFrame(columns = ['label', 'length'])
df_len['label'] = df_tad_cons_copy['label']
df_len['length'] = df_tad_cons_copy['end'] - df_tad_cons_copy['start']

plt.figure(figsize=(5, 3))
sns.barplot(x = 'label', y = 'length', data = df_len, order = cluster_order, palette = color_use,
            capsize = 0.15, saturation = 1, errcolor = 'black', errwidth = 2, 
            ci = 95, linewidth = 2, edgecolor='black')

save_data(r'E:/Users/dcdang/share/TAD_integrate/Epi_signal_in_domain/domain_chrom_state/df_tad_cons_GM12878.pkl', df_tad_cons_copy)

def draw_repwave_across_domain(df_tad_cons_copy, st_ind, ed_ind, df_human_chr_bio_GM12878):  
    cut_pos = []
    for ind in range(st_ind, ed_ind):
        st = df_tad_cons_copy['bd_st'][ind]
        ed = df_tad_cons_copy['bd_ed'][ind]
        if ind != 26:
            continue
        cut_pos.append((st, ed))
    st_all = df_tad_cons_copy['bd_st'][st_ind]
    ed_all = df_tad_cons_copy['bd_ed'][ed_ind]
    value_l = df_human_chr_bio_GM12878['RepWave'][st_all:ed_all+1]
    plt.figure(figsize=(5, 3))
    plt.plot(value_l)
    for pos in cut_pos:
        pos_use = pos
        plt.vlines(pos_use, np.min(value_l), np.max(value_l), linestyles = '--', color = 'black')
    
st_ind = 0
ed_ind = 30
draw_repwave_across_domain(df_tad_cons_copy, st_ind, ed_ind, df_human_chr_bio_GM12878)
    
        



bio_data_list = ['H2A.Z', 'H3K4me3', 'H3K4me1', 'H3K4me2',
                  'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K9ac',
                  'H3K9me3', 'H3K79me2', 'CTCF', 'SMC3', 'RAD21',
                  'Pol2', 'YY1', 'DNA_methylation', 'DNase', 'Repli',
                  'RNA-', 'RNA+', 'RNA']


bio_t = 'Repli'
df_domain_repwave_signal = read_save_data(r'E:\Users\dcdang\share\TAD_integrate\Epi_signal_in_domain' + '/' + 'GM12878_chr2_' + bio_t + '_original_signal.pkl')    
df_domain_repwave_signal['index'] = list(range(len(df_domain_repwave_signal)))    
order_use = np.array(df_domain_sort['index'])
df_domain_repwave_signal['index'] = df_domain_repwave_signal['index'].astype('category').cat.set_categories(order_use)
df_domain_repwave_signal = df_domain_repwave_signal.sort_values(by=['index'])
df_domain_repwave_signal = df_domain_repwave_signal.reset_index(drop = True)

heatmap_plot_repwave = np.array(df_domain_repwave_signal[list(range(100, 601))])
mean_repwave = np.mean(heatmap_plot_repwave, axis = 1)
    
for bio_t in bio_data_list:   
    if bio_t == 'Repli':
        continue
    print(bio_t)
    cor_l = []
    df_domain_h3k27ac_signal = read_save_data(r'E:\Users\dcdang\share\TAD_integrate\Epi_signal_in_domain' + '/' + 'GM12878_chr2_' + bio_t + '_original_signal.pkl')    
    df_domain_h3k27ac_signal['index'] = list(range(len(df_domain_h3k27ac_signal)))    
    order_use = np.array(df_domain_sort['index'])
    df_domain_h3k27ac_signal['index'] = df_domain_h3k27ac_signal['index'].astype('category').cat.set_categories(order_use)
    df_domain_h3k27ac_signal = df_domain_h3k27ac_signal.sort_values(by=['index'])
    df_domain_h3k27ac_signal = df_domain_h3k27ac_signal.reset_index(drop = True)
    df_domain_h3k27ac_signal['label'] = df_domain_sort['label']
    
    heatmap_plot_h3k27ac = np.array(df_domain_h3k27ac_signal[list(range(100, 601))])
    
    for i in range(len(heatmap_plot_repwave)):
        cor = scipy.stats.spearmanr(heatmap_plot_repwave[i], heatmap_plot_h3k27ac[i])[0] 
        cor_l.append(cor)
        
    
df_domain_repwave_signal['label'] = df_domain_sort['label']
df_domain_repwave_signal['cor'] = cor_l


df_cor = pd.DataFrame(columns = ['label', 'cor'])

df_cor['label'] = df_domain_repwave_signal['label']
df_cor['cor'] = df_domain_repwave_signal['cor']

plt.figure(figsize=(5, 3))
sns.barplot(x = 'label', y='cor', data = df_cor)



    




def draw_bd_region_case_plus(st, ed, contact_map, TAD_list, bd_score_cell_combine, bd_symbol, domain_symbol, repli_signal, Chr, save_name = '', bin_size = 8, resolution = 50000):
    x_axis_range = range(len(bd_score_cell_combine['bd_score'][st:ed]))
    start = st * resolution
    end = ed * resolution + resolution
    start_ = start / 1000000 
    end_ = end / 1000000
    region_name = Chr + ':' + str(start_) + '-' + str(end_) + ' Mb'
    x_ticks_l = []
    y_ticks_l = []
    cord_list = []
    for i in range(ed - st):
        if i % bin_size == 0:
            pos = (st+i) * resolution
            cord_list.append(i)
            x_ticks_l.append(str(pos / 1000000))
            y_ticks_l.append(str(pos / 1000000))
    
    plt.figure(figsize=(8, 8))     
    ax0 = plt.subplot2grid((8, 8), (0, 0), rowspan=4,colspan=4)
    dense_matrix_part = contact_map[st:ed+1, st:ed+1]
    img = ax0.imshow(dense_matrix_part, cmap='seismic', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 91))
    #img = ax0.imshow(dense_matrix_part, cmap='coolwarm', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
    ax0.set_xticks([])
    #ax0.set_yticks([])
    ax0.spines['bottom'].set_linewidth(0)
    ax0.spines['left'].set_linewidth(1.6)
    ax0.spines['right'].set_linewidth(0)
    ax0.spines['top'].set_linewidth(0)
    ax0.tick_params(axis = 'y', length=5, width = 1.6)
    ax0.tick_params(axis = 'x', length=5, width = 1.6)
    plt.xticks(cord_list, x_ticks_l, FontSize = 10)
    plt.yticks(cord_list, y_ticks_l, FontSize = 10)
    ax0.set_title(region_name, fontsize=12, pad = 15.0)
    
    TAD_color = 'black'
    if len(TAD_list) != 0:
        for TAD in TAD_list:
            st_tad = TAD[0] - st
            ed_tad = TAD[1] - st
            print(st_tad, ed_tad)
            draw_tad_region(st_tad, ed_tad, TAD_color, size_v=3, size_h=3)
            #draw_tad_region_upper_half(st_tad, ed_tad, TAD_color, size_v=3, size_h=3)

    cax = plt.subplot2grid((8, 8), (0, 4), rowspan=4,colspan=1)
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes("right", size="1.5%", pad= 0.2)
    #cbar = plt.colorbar(img, cax=cax, ticks=MultipleLocator(2.0), format="%.1f",orientation='vertical',extendfrac='auto',spacing='uniform')
    cbaxes = inset_axes(cax, width="30%", height="50%", loc=2) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])

    ax1 = plt.subplot2grid((8, 8), (4, 0), rowspan=1, colspan=4, sharex=ax0)    
    ax1.plot(x_axis_range, bd_score_cell_combine['bd_score'][st:ed], marker = '.', linewidth = 2, c = 'black')
    ax1.bar(x_axis_range, list(bd_score_cell_combine['bd_score'][st:ed]))
    plt.ylabel('bd_score')
    ax1.set_xticks([])
    #ax1.set_yticks([])
    ax1.spines['bottom'].set_linewidth(1.6)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(1.6)
    ax1.spines['top'].set_linewidth(1.6)
    ax1.tick_params(axis = 'y', length=5, width = 1.6)
    ax1.tick_params(axis = 'x', length=5, width = 1.6)

    ax2 = plt.subplot2grid((8, 8), (5, 0), rowspan=1,colspan=4, sharex=ax0)    
    bd_data = []
    cmap=['#B0B0B0','#459457','#D65F4D','#4392C3']
    my_cmap = ListedColormap(cmap)
    bounds=[0,0.9,1.9,2.9,3.9]
    norm = matplotlib.colors.BoundaryNorm(bounds, my_cmap.N)    
    for i in range(10):
        bd_data.append(bd_symbol[st:ed])
    #bd_data_expand = np.reshape(np.array(bd_data), (10, len(bd_data[0])))
    ax2.imshow(bd_data, cmap = my_cmap, Norm = norm)
    plt.ylabel('bd type')
    ax2.spines['bottom'].set_linewidth(1.6)
    ax2.spines['left'].set_linewidth(1.6)
    ax2.spines['right'].set_linewidth(1.6)
    ax2.spines['top'].set_linewidth(1.6)
    ax2.tick_params(axis = 'y', length=0, width = 0)
    ax2.tick_params(axis = 'x', length=0, width = 0)    
    
    ax3 = plt.subplot2grid((8, 8), (6, 0), rowspan=1,colspan=4, sharex=ax0)    
    domain_data = []
    cmap = ['#B0B0B0', '#3A4CC0', '#7C9FF9', '#EDCBBA', '#E9795E', '#BC1F2C']
    my_cmap = ListedColormap(cmap)
    bounds=[0, 0.9, 1.9, 2.9, 3.9, 4.9, 5.9]
    norm = matplotlib.colors.BoundaryNorm(bounds, my_cmap.N)    
    for i in range(10):
        domain_data.append(domain_symbol[st:ed])
    #bd_data_expand = np.reshape(np.array(bd_data), (10, len(bd_data[0])))
    ax3.imshow(domain_data, cmap = my_cmap, Norm = norm)
    plt.ylabel('Domain type')
    ax3.spines['bottom'].set_linewidth(1.6)
    ax3.spines['left'].set_linewidth(1.6)
    ax3.spines['right'].set_linewidth(1.6)
    ax3.spines['top'].set_linewidth(1.6)
    ax3.tick_params(axis = 'y', length=0, width = 0)
    ax3.tick_params(axis = 'x', length=0, width = 0)    

    ax4 = plt.subplot2grid((8, 8), (7, 0), rowspan=1, colspan=4, sharex=ax0)    
    ax4.plot(x_axis_range, repli_signal[st:ed], marker = '.', linewidth = 2, c = 'black')
    ax4.bar(x_axis_range, list(repli_signal[st:ed]))
    plt.ylabel('Replication timing')
    ax4.set_xticks([])
    #ax1.set_yticks([])
    ax4.spines['bottom'].set_linewidth(1.6)
    ax4.spines['left'].set_linewidth(1.6)
    ax4.spines['right'].set_linewidth(1.6)
    ax4.spines['top'].set_linewidth(1.6)
    ax4.tick_params(axis = 'y', length=5, width = 1.6)
    ax4.tick_params(axis = 'x', length=5, width = 1.6)
         
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)


start = 2734 - 32
end = 2734 + 22

start = 1422 - 30
end = 1422 + 30

start = 177 - 30
end = 177 + 30


start = 750
end = 880

start = 475 - 50
end = 475 + 50


start = 923 - 50
end = 923 + 50


start = 1178 - 50
end = 1178 + 50


start = 2878 - 50
end = 2878 + 50


start = 3059 - 50
end = 3059 + 50

start = 3244 - 50
end = 3244 + 50

start = 3397 - 50
end = 3397 + 50

start = 3643 - 30
end = 3643 + 50


TAD_list = []
for i in range(len(df_record)):
    st = df_record['region_pair'][i][0]
    ed = df_record['region_pair'][i][-1]
    if st >= start and ed <= end:
        TAD_list.append(df_record['region_pair'][i])



cluster_order = [2, 4, 0, 1, 3]
cluster_score = {'0': 3, '1': 2, '2': 5, '3': 1, '4': 4}
cluster_color = {'0': '#EDCBBA', '1': '#7C9FF9', '2': '#BC1F2C', '3': '#3A4CC0', '4': '#E9795E'}

domain_symbol = np.zeros(len(mat_dense))
for i in range(len(df_tad_cons_copy)):
    st = df_tad_cons_copy['bd_st'][i]
    ed = df_tad_cons_copy['bd_ed'][i]
    label = df_tad_cons_copy['label'][i]
    score = cluster_score[str(label)]
    domain_symbol[st:ed] = score
    
repli_signal = df_human_chr_bio_GM12878['RepWave']

mat_norm = (mat_use - np.min(mat_use)) / (np.max(mat_use) - np.min(mat_use))

mat_norm = copy.deepcopy(mat_use)

draw_bd_region_case_plus(start, end, mat_norm, TAD_list, bd_score_cell_combine, bd_symbol, domain_symbol, repli_signal, Chr, save_name = '', bin_size = 10, resolution = 50000)






save_data(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/GM12878_K562_compare/GM12878_chr2_domain_type_symbol.pkl', domain_symbol)

save_data(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/GM12878_K562_compare/GM12878_chr2_bio_signal_combine.pkl', df_GM12878_mboi_chr2)





### relative replicate in domain

## load data from ubuntu
df_domain_repli_signal_relative = read_save_data(r'E:\Users\dcdang\share\TAD_integrate\Epi_signal_in_domain' + '/' + 'GM12878_chr2_repli_original_signal.pkl')

df_domain_repli_signal_relative['index'] = list(range(len(df_domain_repli_signal_relative)))
df_domain_repli_signal_relative['st_bd_type'] = df_tad_cons['st_region_type']
df_domain_repli_signal_relative['ed_bd_type'] = df_tad_cons['ed_region_type']


st = 100
ed = 600

def min_max_norm_repli(df_domain_repli_signal, st, ed):
    value_all = []
    for i in range(len(df_domain_repli_signal)):
        value = np.array(df_domain_repli_signal[list(range(st, ed))].iloc[i])
        if np.max(value) != 0:
            value_norm = (np.array(value) - np.min(value)) / (np.max(value) - np.min(value))
        else:
            value_norm = value_tad
        value_all.append(value_norm)
    df_domain_repli_cut = pd.DataFrame(np.array(value_all))
    df_domain_repli_cut['index'] = df_domain_repli_signal['index'] 
    df_domain_repli_cut['st_bd_type'] = df_domain_repli_signal['st_bd_type']
    df_domain_repli_cut['ed_bd_type'] =  df_domain_repli_signal['ed_bd_type']
    return df_domain_repli_cut
 
       
df_domain_repli_cut = min_max_norm_repli(df_domain_repli_signal_relative, st, ed)      

X = df_domain_repli_cut[list(range(ed-st))]


## dendrogram plot to decide cluster number
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist,squareform

color_link = ['#3A4CC0', '#7C9FF9', '#EDCBBA', '#E9795E', '#BC1F2C' ]
shc.set_link_color_palette(color_link)
plt.figure(figsize=(9, 4))
plt.title("Customer Dendograms")
with plt.rc_context({'lines.linewidth': 2.5}):
    dend = shc.dendrogram(shc.linkage(X, method='ward', metric='euclidean'), 
                          above_threshold_color='#AAAAAA', no_labels = True, 
                          color_threshold=2000)
#plt.ylim([-1000, 16000])
#plt.xlim([-300, 6100])
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.6)
ax.spines['left'].set_linewidth(1.6)
ax.spines['right'].set_linewidth(1.6)
ax.spines['top'].set_linewidth(1.6)
ax.tick_params(axis = 'y', length=7, width = 1.6)
ax.tick_params(axis = 'x', length=3, width = 1.6)
#plt.savefig(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/heatmap/Cluster_Dendograms_domain_repli.svg', 
            #format = 'svg', dpi = 600, transparent = True) 



cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward', compute_distances = True)
#cluster = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='euclidean', linkage='ward')
model = cluster.fit(X)
labels = cluster.fit_predict(X)


# sort domain by label
df_domain_repli_cut['label'] = labels
df_domain_repli_cut = df_domain_repli_cut.sort_values(by = ['label'])
df_domain_repli_cut = df_domain_repli_cut.reset_index(drop = True)

point_c = get_cluster_change_point(df_domain_repli_cut) 

plt.figure(figsize=(10, 10))
heatmap_plot = np.array(df_domain_repli_cut[list(range(ed-st))])
plt.imshow(heatmap_plot, cmap = 'YlGnBu', vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85))
plt.colorbar()
plt.yticks([])
for pos in point_c:
    plt.hlines(pos, 0, 499, linewidth = 5, linestyle = '--')
plt.xticks([0, 100, 200, 300, 400, 499])
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.6)
ax.spines['left'].set_linewidth(1.6)
ax.spines['right'].set_linewidth(1.6)
ax.spines['top'].set_linewidth(1.6)
ax.tick_params(axis = 'y', length=5, width = 1.6)
ax.tick_params(axis = 'x', length=5, width = 1.6)
plt.savefig(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/relative_type/domain_relative_repli_signal.svg', 
            format = 'svg', transparent = True) 





lb_color ={'0':'#d7191c',
           '1':'#fdae61', 
           '2':'#7b3294', 
           '3':'#abdda4',
           '4':'#2b83ba'}

record = {}
for lb in list(np.unique(df_domain_repli_cut['label'])):
    print(lb)
    color = lb_color[str(lb)]
    print(color)
    df_repli_part = df_domain_repli_cut[df_domain_repli_cut['label'] == lb]
    df_repli_part = df_repli_part.sort_values(by = ['index'])
    df_repli_part = df_repli_part.reset_index(drop = True)
    record[lb] = df_repli_part
    df_data = pd.DataFrame(columns = ['pos', 'value'])
    pos_l = []
    v_l = []
    for i in range(len(df_repli_part)):
        vec = df_repli_part[list(range(ed-st))].iloc[i]
        v_l += list(vec)
        pos_l += list(range(len(vec)))
    df_data['pos'] = pos_l
    df_data['value'] = v_l    
    plt.figure(figsize=(5, 3))
    #plt.plot(np.mean(df_repli_part[list(range(ed - st))], axis = 0), linewidth = 3)
    sns.lineplot(x = 'pos', y = 'value', data = df_data, ci = None, color=color, linewidth = 8)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.tick_params(axis = 'y', length=5, width = 3)
    ax.tick_params(axis = 'x', length=5, width = 3)
    plt.savefig(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/relative_type/mean_domain_relative_repli_signal_' + str(lb) + '.svg', 
                format = 'svg', transparent = True) 
        


''' mei you yong, kan bu chu cha bie
bio_t = 'CTCF'
df_domain_CTCF_signal = read_save_data(r'E:\Users\dcdang\share\TAD_integrate\Epi_signal_in_domain' + '/' + 'GM12878_chr2_' + bio_t + '_original_signal.pkl')    
df_domain_CTCF_signal['index'] = list(range(len(df_domain_CTCF_signal)))    

order_use = np.array(df_domain_repli_cut['index'])
df_domain_CTCF_signal['index'] = df_domain_CTCF_signal['index'].astype('category').cat.set_categories(order_use)
df_domain_CTCF_signal = df_domain_CTCF_signal.sort_values(by=['index'])
df_domain_CTCF_signal = df_domain_CTCF_signal.reset_index(drop = True)
df_domain_CTCF_signal['label'] = df_domain_repli_cut['label']
   

ctcf_left = np.mean(df_domain_CTCF_signal[list(range(70,130))], axis = 1)
ctcf_right = np.mean(df_domain_CTCF_signal[list(range(570,630))], axis = 1)

df_ctcf = pd.DataFrame(columns = ['value', 'pos', 'label'])
df_ctcf['value'] = list(ctcf_left) + list(ctcf_right)
df_ctcf['pos'] = ['left' for i in range(len(ctcf_left))] + ['right' for i in range(len(ctcf_right))]
df_ctcf['label'] = list(df_domain_CTCF_signal['label']) + list(df_domain_CTCF_signal['label'])

plt.figure(figsize=(4, 4))
sns.barplot(x = 'label', y = 'value', data = df_ctcf, hue = 'pos')


plt.figure(figsize=(10, 10))
heatmap_plot = np.array(df_domain_CTCF_signal[list(range(st-20, st+20)) + list(range(ed-20, ed+20))])
plt.imshow(heatmap_plot, cmap = 'YlGnBu', vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85))
plt.colorbar()
plt.yticks([])
for pos in point_c:
    plt.hlines(pos, 0, 79, linewidth = 5, linestyle = '--')
#plt.xticks([0, 100, 200, 300, 400, 499])
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.6)
ax.spines['left'].set_linewidth(1.6)
ax.spines['right'].set_linewidth(1.6)
ax.spines['top'].set_linewidth(1.6)
ax.tick_params(axis = 'y', length=5, width = 1.6)
ax.tick_params(axis = 'x', length=5, width = 1.6)


for lb in list(np.unique(df_domain_CTCF_signal['label'])):
    print(lb)
    color = lb_color[str(lb)]
    print(color)
    df_domain_CTCF_signal_part = df_domain_CTCF_signal[df_domain_CTCF_signal['label'] == lb]
    df_domain_CTCF_signal_part = df_domain_CTCF_signal_part.sort_values(by = ['index'])
    df_domain_CTCF_signal_part = df_domain_CTCF_signal_part.reset_index(drop = True)
    df_data = pd.DataFrame(columns = ['pos', 'value'])
    pos_l = []
    v_l = []
    for i in range(len(df_domain_CTCF_signal_part)):
        vec = df_domain_CTCF_signal_part[list(range(0, 700))].iloc[i]
        v_l += list(vec)
        pos_l += list(range(len(vec)))
    df_data['pos'] = pos_l
    df_data['value'] = v_l    
    plt.figure(figsize=(5, 3))
    #plt.plot(np.mean(df_repli_part[list(range(ed - st))], axis = 0), linewidth = 3)
    sns.lineplot(x = 'pos', y = 'value', data = df_data, ci = 'sd', color=color, linewidth = 5)
    #plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
'''

#### Use CTCF peaks number around boundary 

df_CTCF_peak_num_bin_chr2 = pd.read_csv('E:/Users/dcdang/share/TAD_integrate/CTCF_peak_overlap_bin/human_chr2_50000_bin_with_CTCF_peak_multiple_cells.bed', sep = '\t', header = 0)

cell_with_CTCF_peak = ['GM12878_CTCF_peak_num', 'K562_CTCF_peak_num',
       'IMR90_CTCF_peak_num', 'HUVEC_CTCF_peak_num', 'HMEC_CTCF_peak_num',
       'NHEK_CTCF_peak_num']

df_CTCF_peak_num_bin_chr2_part = df_CTCF_peak_num_bin_chr2[cell_with_CTCF_peak]

ctcf_target = df_CTCF_peak_num_bin_chr2_part['GM12878_CTCF_peak_num']

left_l = []
right_l = []
df_tad_cons_copy = copy.deepcopy(df_tad_cons)
for i in range(len(df_domain_repli_cut)):
    ind = df_domain_repli_cut['index'][i]
    bd_st = df_tad_cons_copy['start'][ind]
    bd_ed = df_tad_cons_copy['end'][ind]
    
    st_ind = np.int(bd_st / resolution)
    ed_ind = np.int(bd_ed / resolution)-1
    st_1 = np.max([0, st_ind - 2])
    st_2 = np.min([len(df_CTCF_peak_num_bin_chr2), st_ind + 2])
    ed_1 = np.max([0, ed_ind - 2])
    ed_2 = np.min([len(df_CTCF_peak_num_bin_chr2), ed_ind + 2])  
    #left_l.append(ctcf_target[st_ind])
    #right_l.append(ctcf_target[ed_ind]) 
    left_l.append(np.mean(ctcf_target[st_1:st_2+1]))
    right_l.append(np.mean(ctcf_target[ed_1:ed_2+1])) 

df_ctcf = pd.DataFrame(columns = ['value', 'pos', 'label'])
df_ctcf['value'] = list(left_l) + list(right_l)
df_ctcf['pos'] = ['left' for i in range(len(left_l))] + ['right' for i in range(len(right_l))]
df_ctcf['label'] = list(df_domain_CTCF_signal['label']) + list(df_domain_CTCF_signal['label'])

plt.figure(figsize=(6, 4))
sns.barplot(x = 'label', y = 'value', data = df_ctcf, hue = 'pos', 
            capsize = 0.15, saturation = 8, errcolor = 'black', errwidth = 1.8, ci = 95, edgecolor='black', palette = ['#fc8d62', '#8da0cb'])
plt.ylabel('Average CTCF peak number around boundary', FontSize = 10)
plt.xlabel('', FontSize = 0)
plt.ylim([0, 1.5])
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.6)
ax.spines['left'].set_linewidth(1.6)
ax.spines['right'].set_linewidth(1.6)
ax.spines['top'].set_linewidth(1.6)
ax.tick_params(axis = 'y', length=5, width = 1.6)
ax.tick_params(axis = 'x', length=5, width = 1.6)
ax.legend_ = None
plt.savefig(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/boundary_region/analysis_new/boundary_match/domain_repli/relative_type/Average_CTCF_peaks_around_boundary.svg', 
            format = 'svg', transparent = True) 
    

df_ctcf_combine = pd.DataFrame(columns = ['left', 'right', 'label'])
df_ctcf_combine['left'] = left_l
df_ctcf_combine['right'] = right_l
df_ctcf_combine['label'] = list(df_domain_CTCF_signal['label'])
for label in list(np.unique(df_ctcf_combine['label'])):
    df_ctcf_combine_part = df_ctcf_combine[df_ctcf_combine['label'] == label]
    df_ctcf_combine_part = df_ctcf_combine_part.reset_index(drop = True)
    left_vec = list(df_ctcf_combine_part['left'])
    right_vec = list(df_ctcf_combine_part['right'])
    sta, pvalue = scipy.stats.mannwhitneyu(left_vec, right_vec)
    print('This is ' + str(label))
    print((sta, pvalue))
    print('Left:')
    print(np.median(left_vec))
    print('Right:')
    print(np.median(right_vec))     



### relative domain and boundary type, domain replication type

df_stats_relative = stats_bd_num_ratio(df_domain_repli_cut)

color_bd_use = {'sharp_strong':'#D65F4D', 'sharp_weak':'#459457', 'wide':'#4392C3'}
st_color = []
ed_color = []
cluster_lb_l = []
cluster_lb_color = []
for i in range(len(df_domain_repli_cut)):
    st_type = df_domain_repli_cut['st_bd_type'][i]
    ed_type = df_domain_repli_cut['ed_bd_type'][i]
    ind = df_domain_repli_cut['index'][i]
    ind_lb = list(df_domain_sort['index']).index(ind)
    lb = df_domain_sort['label'][ind_lb]
    lb_color = df_domain_sort['label_color'][ind_lb]
    cluster_lb_l.append(lb)
    cluster_lb_color.append(lb_color)
    st_color.append(color_bd_use[st_type])
    ed_color.append(color_bd_use[ed_type])
df_domain_repli_cut['cluster_label'] = cluster_lb_l   
df_domain_repli_cut['cluster_color'] = cluster_lb_color   
df_domain_repli_cut['st_color'] = st_color   
df_domain_repli_cut['ed_color'] = ed_color   



bd_label_order = ['wide', 'sharp_strong', 'sharp_weak']
domain_sort_replative = []
for lb in [0, 1, 2, 3, 4]:
    df_domain_repli_cut_part = copy.deepcopy(df_domain_repli_cut[df_domain_repli_cut['label'] == lb])
    df_domain_repli_cut_part['st_bd_type'] = df_domain_repli_cut_part['st_bd_type'].astype('category').cat.set_categories(bd_label_order)
    df_domain_repli_cut_part['ed_bd_type'] = df_domain_repli_cut_part['ed_bd_type'].astype('category').cat.set_categories(bd_label_order)
    df_domain_repli_cut_part = df_domain_repli_cut_part.sort_values(by=['st_bd_type', 'ed_bd_type'])
    df_domain_repli_cut_part = df_domain_repli_cut_part.reset_index(drop = True)
    domain_sort_replative.append(df_domain_repli_cut_part)


cluster_label_order = [2, 4, 0, 1, 3]
domain_sort_replative = []
for lb in [0, 1, 2, 3, 4]:
    df_domain_repli_cut_part = copy.deepcopy(df_domain_repli_cut[df_domain_repli_cut['label'] == lb])
    df_domain_repli_cut_part['cluster_label'] = df_domain_repli_cut_part['cluster_label'].astype('category').cat.set_categories(cluster_label_order)
    df_domain_repli_cut_part = df_domain_repli_cut_part.sort_values(by=['cluster_label'])
    df_domain_repli_cut_part = df_domain_repli_cut_part.reset_index(drop = True)
    domain_sort_replative.append(df_domain_repli_cut_part)


df_domain_replative_sort = domain_sort_replative[0]
for i in range(1, len(domain_sort_replative)):
    df_domain_replative_sort = pd.concat([df_domain_replative_sort, domain_sort_replative[i]], axis = 0)
df_domain_replative_sort = df_domain_replative_sort.reset_index(drop = True)




row_colors = df_domain_replative_sort['st_color']
row_colors2 = df_domain_replative_sort['ed_color']
lb_color_row = df_domain_replative_sort['cluster_color']


heatmap_plot = np.array(df_domain_replative_sort[list(range(ed-st))])
sns.clustermap(heatmap_plot, row_cluster = False, col_cluster = False,
               row_colors=[lb_color_row], method='ward', 
               metric='euclidean', cmap = 'YlGnBu', 
               vmin = np.percentile(heatmap_plot, 15), vmax = np.percentile(heatmap_plot, 85), 
               yticklabels=False, xticklabels=False)




##################

repli_label = []
for i in range(len(df_domain_sort)):
    ind = df_domain_sort['index'][i]
    ind_rep = list(df_domain_repli_cut['index']).index(ind)
    repli_label.append(df_domain_repli_cut['label'][ind_rep])
    
df_domain_sort['Repli_label'] = repli_label


label_l = []
rel_label_l = []
df_tad_cons_copy = copy.deepcopy(df_tad_cons)
for i in range(len(df_tad_cons_copy)):
    ind = list(df_domain_sort['index']).index(i)
    label_l.append(df_domain_sort['label'][ind])
    rel_label_l.append(df_domain_sort['Repli_label'][ind])

df_tad_cons_copy['label'] = label_l
df_tad_cons_copy['rep_label'] = rel_label_l





    



