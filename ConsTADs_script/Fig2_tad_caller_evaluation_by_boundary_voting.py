# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 20:27:26 2022

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
import random
import seaborn as sns
import pickle
import scipy
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import heapq
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker  import MultipleLocator

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
filter_size_indic = 500000
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


cell_type = 'GM12878'
enzyme = 'DpnII'
mat_dense = copy.deepcopy(hic_mat_all_cell_replicate[cell_type][enzyme]['iced'])

IS_value_all = copy.deepcopy(indictor_record_all_cell[cell_type][enzyme]['IS'])

DI_value_all = copy.deepcopy(indictor_record_all_cell[cell_type][enzyme]['DI'])

CI_value_all = copy.deepcopy(indictor_record_all_cell[cell_type][enzyme]['CI'])


## calculate boundary score by boundary voting

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

cell_type = 'GM12878'
enzyme = 'DpnII'        
bd_score_final = boundary_score_cell_all[cell_type][enzyme] 


##################  TAD caller evaluation    
        
def get_method_boundary_score_composition(TAD_result_all, enzyme, boundary_score_final, method_list, bin_name_Chr):
    method_boundary_score = {}
    for method in method_list:
        print('This is ' + method)
        df_tad_m = TAD_result_all[enzyme][method]['TAD_domain']
        df_boundary_score = pd.DataFrame(columns = ['index', 'score'])
        ind_list = []
        score_list = []
        for i in range(len(df_tad_m)):
            bd_st = df_tad_m['boundary_st'][i]
            bd_ed = df_tad_m['boundary_ed'][i]
            ind_st = bin_name_Chr.index(bd_st)
            ind_ed = bin_name_Chr.index(bd_ed)
            ind_list.append(ind_st)
            ind_list.append(ind_ed)
            score_list.append(boundary_score_final['bd_score'][ind_st])
            score_list.append(boundary_score_final['bd_score'][ind_ed])
        print(np.max(score_list))
        df_boundary_score['index'] = ind_list
        df_boundary_score['score'] = score_list
        method_boundary_score[method] = df_boundary_score
    return method_boundary_score

cell_type = 'GM12878'
enzyme = 'DpnII'  
TAD_result_all = TAD_result_all_cell_type[cell_type]
method_boundary_score = get_method_boundary_score_composition(TAD_result_all, enzyme, bd_score_final, method_list, bin_name_Chr)
    

##### draw   

def draw_bd_score_pie(bd_score_final, save_name):
    score_all = np.unique(bd_score_final['bd_score'])
    number_list = []
    for score in score_all:
        number = list(bd_score_final['bd_score']).count(score)
        number_list.append(number)
        print('score: ' + str(score) + '; ' 
              + 'number: ' + str(number) + ';' 
              + 'ratio: ' + str(number / len(bd_score_final))
              )    
    plt.figure(figsize= (5, 5))   
    plt.pie(number_list, explode=None, labels=score_all, autopct=None, pctdistance=0.6, colors= ['grey', 'lightskyblue', 'palegreen', 'khaki', 'navajowhite', 'lightcoral', 'violet', 'mediumslateblue', 'deepskyblue', 'limegreen', 'gold', 'orange','red', 'purple', 'slateblue', 'navy'],
        shadow=False, labeldistance=1.1, startangle=None, radius=None, counterclock=True, 
        wedgeprops={'linewidth': 1, 'edgecolor': "black"}, textprops=None, center=(0, 0), frame=False, rotatelabels=False, data=None)
    plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)

  
def get_cut_off_ratio(method_boundary_score, cut_off_list, method_list, save_name):
    #color = ['#2D84BB','#ADD8A4','#FAAE62', '#D72027']
    color = ['#2D84BB','#ADD8A4','#E6F598','#FAAE62', '#D72027']   
    #color = ['#3288BD', '#99D594', '#E6F598', '#FEE08B', '#FC8D59', '#D53E4F']            
    color_all = color
    method_ratio_all = {}
    ratio_record = []
    for method in method_list:
        print(method)
        method_ratio = []
        df_boundary_score = method_boundary_score[method]
        for cut_interval in cut_off_list:
             bd_num = np.sum(df_boundary_score['score'].isin(cut_interval))
             method_ratio.append(bd_num / len(df_boundary_score))
        method_ratio_all[method] = method_ratio
        print(np.sum(method_ratio))
        ratio_record.append(method_ratio)
    df_ratio_all = pd.DataFrame(ratio_record)
    df_ratio_all['method'] = method_list
    df_ratio_all_sort = df_ratio_all.sort_values(by = [0,1,2])
    df_ratio_all_sort = df_ratio_all_sort.reset_index(drop = True)
    df_ratio_all_sort[-1] = [0 for i in range(len(df_ratio_all_sort))]
    method_ord = list(df_ratio_all_sort['method'])
    method_ord_new = []
    for method in method_ord:
        if method != 'InsulationScore':
            method_ord_new.append(method)
        else:
            method_ord_new.append('IS')
    plt.figure(figsize=(8,5))
    bottom_list = np.zeros(len(method_ord))
    for i in range(len(cut_off_list)):
        plt.bar(range(len(method_ord)), list(df_ratio_all_sort[i]), align="center", bottom=list(bottom_list), color=color_all[i], edgecolor = 'black', linewidth = 1.5)
        bottom_list += np.array(df_ratio_all_sort[i])
    
    plt.xticks(list(range(len(method_ord))), method_ord_new, rotation= -30, FontSize = 12)
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
    return method_ratio_all

def get_cut_off_number(method_boundary_score, cut_off_list, method_list, save_name):
    #color = ['#2D84BB','#ADD8A4','#FAAE62', '#D72027']
    color = ['#2D84BB','#ADD8A4','#E6F598','#FAAE62', '#D72027']   
    #color = ['#3288BD', '#99D594', '#E6F598', '#FEE08B', '#FC8D59', '#D53E4F']            
    color_all = color
    method_num_all = {}
    num_record = []
    bd_num_all = []
    for method in method_list:
        print(method)
        method_num = []
        df_boundary_score = method_boundary_score[method]
        bd_num_all.append(len(df_boundary_score))
        for cut_interval in cut_off_list:
             bd_num = np.sum(df_boundary_score['score'].isin(cut_interval))
             method_num.append(bd_num)
        method_num_all[method] = method_num
        print(np.sum(method_num))
        num_record.append(method_num)
    df_num_all = pd.DataFrame(num_record)
    df_num_all['method'] = method_list
    df_num_all['bd_num'] = bd_num_all
    
    df_num_all['ord']= df_num_all[0] + df_num_all[1]
    #df_num_all_sort = df_num_all.sort_values(by = ['ord'])
    df_num_all_sort = df_num_all.sort_values(by = [0,1,2])
    df_num_all_sort = df_num_all_sort.reset_index(drop = True)
    df_num_all_sort[-1] = [0 for i in range(len(df_num_all_sort))]
    
    #ord by all number use this
    df_num_all_sort = df_num_all.sort_values(by = ['bd_num'])
    df_num_all_sort = df_num_all_sort.reset_index(drop = True)
    df_num_all_sort[-1] = [0 for i in range(len(df_num_all_sort))]

    method_ord = list(df_num_all_sort['method'])
    method_ord_new = []
    for method in method_ord:
        if method != 'InsulationScore':
            method_ord_new.append(method)
        else:
            method_ord_new.append('IS')
    plt.figure(figsize=(8,5))
    bottom_list = np.zeros(len(method_ord))
    for i in range(len(cut_off_list)):
        plt.bar(range(len(method_ord)), list(df_num_all_sort[i]), align="center", bottom=list(bottom_list), color=color_all[i], edgecolor = 'black', linewidth = 1.5)
        bottom_list += np.array(df_num_all_sort[i])
    
    plt.xticks(list(range(len(method_ord))), method_ord_new, rotation= -30, FontSize = 12)
    #plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'], FontSize = 12)
    #plt.ylim([0,1])
    plt.ylabel('Boundary number',  FontSize = 12)
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
    return method_num_all
    

### score pie
    
save_name_pie = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/DpnII/method_evalute/geneome_bin_score_pie.svg'    
draw_bd_score_pie(bd_score_final, save_name_pie)


#cut_off_list = [[1], [2,3,4], [5,6,7], [8,9,10,11]]
#cut_off_list = [[1], [2,3,4,5], [6,7,8,9], [10,11,12,13], [14, 15, 16]]
#cut_off_list = [[1], [2,3,4,5,6], [7,8,9,10], [11,12,13,14,15,16]]
cut_off_list = [[1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14,15,16]]
#cut_off_list = [[1], [2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]

save_name = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/DpnII/method_evalute/method_boundary_score_proportion_5class.svg'
method_ratio_all = get_cut_off_ratio(method_boundary_score, cut_off_list, method_list, save_name)
save_name_num = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/DpnII/method_evalute/method_boundary_score_number_5class.svg'
method_num_all = get_cut_off_number(method_boundary_score, cut_off_list, method_list, save_name_num)
    
 
    
 
### profile of 1D indicator of boundary with different score      
def get_ratio_part_bd_indicator(method_boundary_score, cut_off_list, method_list, IS_value_all, data_type, expand_size, mat_blank_row, random_num, random_round, save_add):
    if not os.path.exists(save_add):
        os.makedirs(save_add)
    #color = ['#2D84BB','#ADD8A4','#FAAE62', '#D72027']
    color = ['#2D84BB','#ADD8A4','#E6F598','#FAAE62', '#D72027']   
    #color = ['#3288BD', '#99D594', '#E6F598', '#FEE08B', '#FC8D59', '#D53E4F']            
    color_all = color
    for method in method_list:
        print('Dealing with ' + method)
        bd_value = []
        cut_label_l = []
        df_boundary_score = method_boundary_score[method]
        for j in range(len(cut_off_list)):
             cut_interval = cut_off_list[j]
             bd_value_cut = []
             df_boundary_score_part = copy.deepcopy(df_boundary_score[df_boundary_score['score'].isin(cut_interval)])
             df_boundary_score_part = df_boundary_score_part.reset_index(drop = True)
             for i in range(len(df_boundary_score_part)):
                 ind = df_boundary_score_part['index'][i]
                 if ind < expand_size or ind > (len(IS_value_all) - expand_size):
                     continue
                 target = list(IS_value_all[data_type][ind - expand_size: ind + expand_size + 1])
                 bd_value_cut.append(target)
             cut_label_l += [j for k in range(len(bd_value_cut))]
             bd_value += bd_value_cut
        df_bd_value_all_cut = pd.DataFrame(bd_value)
        df_bd_value_all_cut['cut_label'] = cut_label_l
        random_control = []
        for k in range(random_round):
            ind_record = []
            ind_random = random.sample(range(len(IS_value_all)), random_num)
            for l in range(len(ind_random)):
                ind = ind_random[l]
                if ind < expand_size or ind > (len(IS_value_all) - expand_size):
                     continue
                target = list(IS_value_all[data_type][ind - expand_size: ind + expand_size + 1])
                ind_record.append(target)
            df_random = pd.DataFrame(ind_record)
            random_control.append(np.mean(df_random, axis = 0))
        df_random_all = pd.DataFrame(random_control)        
        random_mean = np.mean(df_random_all, axis = 0)    
        plt.figure(figsize=(4,3.7))
        for j in range(len(cut_off_list)):
            df_bd_value_all_cut_part = df_bd_value_all_cut[df_bd_value_all_cut['cut_label'] == j]
            col_use = list(range(2*expand_size + 1))
            vec_value = np.median(df_bd_value_all_cut_part[col_use], axis = 0)
            #vec_value_low = np.percentile(df_bd_value_all_cut_part[col_use], 20, axis = 0)
            #vec_value_up = np.percentile(df_bd_value_all_cut_part[col_use], 80, axis = 0)
            plt.plot(vec_value, color = color_all[j], linewidth = 2)
            #plt.fill_between(range(len(vec_value)), vec_value_low, vec_value_up, color=color_all[j], alpha=0.2)
        plt.plot(random_mean, color = 'grey', linewidth = 2)
        plt.title(method + ' boudnary ' + data_type + ' value')
        plt.xticks([0, 10, 20, 30, 40], ['-1Mb', '-500kb', 'boundary', '500kb', '1Mb'], FontSize = 10)
        #plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'], FontSize = 12)
        plt.yticks(FontSize = 10)
        #plt.ylim([0,1])
        plt.ylabel(data_type,  FontSize = 10)
        ax=plt.gca()
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)
        ax.spines['top'].set_linewidth(1.2)
        ax.tick_params(axis = 'y', length=3, width = 1.2)
        ax.tick_params(axis = 'x', length=3, width = 1.2)
        plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)
        save_name = save_add + '/' + method + '_' + data_type + '_region_value.svg'
        #plt.savefig(save_name, format = 'svg', transparent = True) 
        plt.show()
        #fig = plt.gcf() #获取当前figure
        #plt.close(fig)

def get_blank_row_in_matrix(mat_dense):
    df_row_sum = pd.DataFrame(np.sum(mat_dense, axis = 0))
    df_row_sum_part = df_row_sum[df_row_sum[0]==0]
    blank_row = list(df_row_sum_part.index)
    return blank_row

mat_blank_row = get_blank_row_in_matrix(mat_dense)
expand_size = 20
random_num = 200
random_round = 200

data_type = 'IS'
save_add = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/' + enzyme + '/method_evalute/boundary_score_and_value' + '/' + data_type
get_ratio_part_bd_indicator(method_boundary_score, cut_off_list, method_list, IS_value_all, data_type, expand_size, mat_blank_row, random_num, random_round, save_add)

data_type = 'DI'
save_add = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/' + enzyme + '/method_evalute/boundary_score_and_value' + '/' + data_type
get_ratio_part_bd_indicator(method_boundary_score, cut_off_list, method_list, DI_value_all, data_type, expand_size, mat_blank_row, random_num, random_round, save_add)

data_type = 'CI'
save_add = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/' + enzyme + '/method_evalute/boundary_score_and_value' + '/' + data_type
get_ratio_part_bd_indicator(method_boundary_score, cut_off_list, method_list, CI_value_all, data_type, expand_size, mat_blank_row, random_num, random_round, save_add)


################  all method not split for method

def get_ratio_part_bd_indicator_all(bd_score_final, cut_off_list, method_list, IS_value_all, data_type, expand_size, mat_blank_row, random_num, random_round, save_add):
    if not os.path.exists(save_add):
        os.makedirs(save_add)
    #color = ['#2D84BB','#ADD8A4','#FAAE62', '#D72027']
    color = ['#2D84BB','#ADD8A4','#E6F598','#FAAE62', '#D72027']   
    #color = ['#3288BD', '#99D594', '#E6F598', '#FEE08B', '#FC8D59', '#D53E4F']            
    color_all = color
    bd_value = []
    cut_label_l = []
    df_boundary_score = copy.deepcopy(bd_score_final)
    df_boundary_score['index'] = list(range(len(df_boundary_score)))
    for j in range(len(cut_off_list)):
        cut_interval = cut_off_list[j]
        bd_value_cut = []
        df_boundary_score_part = copy.deepcopy(df_boundary_score[df_boundary_score['bd_score'].isin(cut_interval)])   
        df_boundary_score_part = df_boundary_score_part.reset_index(drop = True)
        for i in range(len(df_boundary_score_part)):
            ind = df_boundary_score_part['index'][i]
            if ind < expand_size or ind > (len(IS_value_all) - expand_size):
                continue
            target = list(IS_value_all[data_type][ind - expand_size: ind + expand_size + 1])
            bd_value_cut.append(target)
        cut_label_l += [j for k in range(len(bd_value_cut))]
        bd_value += bd_value_cut
    df_bd_value_all_cut = pd.DataFrame(bd_value)
    df_bd_value_all_cut['cut_label'] = cut_label_l

    random_control = []
    for k in range(random_round):
        ind_record = []
        ind_random = random.sample(range(len(IS_value_all)), random_num)
        for l in range(len(ind_random)):
            ind = ind_random[l]
            dist_ind = []
            for i in range(len(mat_blank_row)):
                dist_ind.append(np.abs(ind - mat_blank_row[i]))
            if np.min(dist_ind) <= expand_size:
                continue
            if ind < expand_size or ind > (len(IS_value_all) - expand_size):
                 continue
            target = list(IS_value_all[data_type][ind - expand_size: ind + expand_size + 1])
            ind_record.append(target)
        df_random = pd.DataFrame(ind_record)
        random_control.append(np.mean(df_random, axis = 0))
    df_random_all = pd.DataFrame(random_control)        
    random_mean = np.mean(df_random_all, axis = 0)    

    plt.figure(figsize=(4,3.7))
    for j in range(len(cut_off_list)):
        df_bd_value_all_cut_part = df_bd_value_all_cut[df_bd_value_all_cut['cut_label'] == j]
        col_use = list(range(2*expand_size + 1))
        vec_value = np.median(df_bd_value_all_cut_part[col_use], axis = 0)
        #vec_value_low = np.percentile(df_bd_value_all_cut_part[col_use], 20, axis = 0)
        #vec_value_up = np.percentile(df_bd_value_all_cut_part[col_use], 80, axis = 0)
        plt.plot(vec_value, color = color_all[j], linewidth = 2.5)
        #plt.fill_between(range(len(vec_value)), vec_value_low, vec_value_up, color=color_all[j], alpha=0.2)
    plt.plot(random_mean, color = 'grey', linewidth = 2.5)
    plt.title('All boudnary ' + data_type + ' value')
    plt.xticks([0, 10, 20, 30, 40], ['-1Mb', '-500kb', 'boundary', '500kb', '1Mb'], FontSize = 10)
    #plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'], FontSize = 12)
    plt.yticks(FontSize = 10)
    #plt.ylim([0,1])
    plt.ylabel(data_type,  FontSize = 10)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['top'].set_linewidth(1.2)
    ax.tick_params(axis = 'y', length=3, width = 1.2)
    ax.tick_params(axis = 'x', length=3, width = 1.2)
    plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)
    save_name = save_add + '/' + 'All' + '_' + data_type + '_region_value.svg'
    plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)
    return df_bd_value_all_cut



data_type = 'IS'
save_add = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/' + enzyme + '/method_evalute/boundary_score_and_value' + '/All_bd'
df_bd_value_all_cut = get_ratio_part_bd_indicator_all(bd_score_final, cut_off_list, method_list, IS_value_all, data_type, expand_size, mat_blank_row, random_num, random_round, save_add)


data_type = 'DI'
save_add = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/' + enzyme + '/method_evalute/boundary_score_and_value' + '/All_bd'
df_bd_value_all_cut = get_ratio_part_bd_indicator_all(bd_score_final, cut_off_list, method_list, DI_value_all, data_type, expand_size, mat_blank_row, random_num, random_round, save_add)


data_type = 'CI'
save_add = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/' + enzyme + '/method_evalute/boundary_score_and_value' + '/All_bd'
df_bd_value_all_cut = get_ratio_part_bd_indicator_all(bd_score_final, cut_off_list, method_list, CI_value_all, data_type, expand_size, mat_blank_row, random_num, random_round, save_add)
    

#### bio peak    
    
def get_ratio_part_bd_peak_all(bd_score_final, cut_off_list, method_list, df_bin_all_with_peak, data_type, expand_size, mat_blank_row, random_num, random_round, save_add):
    if not os.path.exists(save_add):
        os.makedirs(save_add)
    #color = ['#2D84BB','#ADD8A4','#FAAE62', '#D72027']
    color = ['#2D84BB','#ADD8A4','#E6F598','#FAAE62', '#D72027']   
    #color = ['#3288BD', '#99D594', '#E6F598', '#FEE08B', '#FC8D59', '#D53E4F']            
    color_all = color
    bd_value = []
    cut_label_l = []
    df_boundary_score = copy.deepcopy(bd_score_final)
    df_boundary_score['index'] = list(range(len(df_boundary_score)))
    for j in range(len(cut_off_list)):
        cut_interval = cut_off_list[j]
        bd_value_cut = []
        df_boundary_score_part = copy.deepcopy(df_boundary_score[df_boundary_score['bd_score'].isin(cut_interval)])   
        df_boundary_score_part = df_boundary_score_part.reset_index(drop = True)
        for i in range(len(df_boundary_score_part)):
            ind = df_boundary_score_part['index'][i]
            if ind < expand_size or ind > (len(df_bin_all_with_peak) - expand_size):
                continue
            target = list(df_bin_all_with_peak[data_type][ind - expand_size: ind + expand_size + 1])
            bd_value_cut.append(target)
        cut_label_l += [j for k in range(len(bd_value_cut))]
        bd_value += bd_value_cut
    df_bd_value_all_cut = pd.DataFrame(bd_value)
    df_bd_value_all_cut['cut_label'] = cut_label_l
    
    random_control = []
    for k in range(random_round):
        ind_record = []
        ind_random = random.sample(range(len(df_bin_all_with_peak)), random_num)
        for l in range(len(ind_random)):
            ind = ind_random[l]
            dist_ind = []
            for i in range(len(mat_blank_row)):
                dist_ind.append(np.abs(ind - mat_blank_row[i]))
            if np.min(dist_ind) <= expand_size:
                continue
            if ind < expand_size or ind > (len(df_bin_all_with_peak) - expand_size):
                 continue
            target = list(df_bin_all_with_peak[data_type][ind - expand_size: ind + expand_size + 1])
            ind_record.append(target)
        df_random = pd.DataFrame(ind_record)
        random_control.append(np.mean(df_random, axis = 0))
    df_random_all = pd.DataFrame(random_control)        
    random_mean = np.mean(df_random_all, axis = 0)    
    
    plt.figure(figsize=(4,3.7))
    for j in range(len(cut_off_list)):
        df_bd_value_all_cut_part = df_bd_value_all_cut[df_bd_value_all_cut['cut_label'] == j]
        col_use = list(range(2*expand_size + 1))
        vec_value = np.mean(df_bd_value_all_cut_part[col_use], axis = 0)
        #vec_value_low = np.percentile(df_bd_value_all_cut_part[col_use], 20, axis = 0)
        #vec_value_up = np.percentile(df_bd_value_all_cut_part[col_use], 80, axis = 0)
        plt.plot(vec_value, color = color_all[j], linewidth = 2.5)
        #plt.fill_between(range(len(vec_value)), vec_value_low, vec_value_up, color=color_all[j], alpha=0.2)
    plt.plot(random_mean, color = 'grey', linewidth = 2.5)
    plt.title('All boudnary ' + data_type + ' value')
    plt.xticks([0, 10, 20, 30, 40], ['-1Mb', '-500kb', 'boundary', '500kb', '1Mb'], FontSize = 10)
    #plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'], FontSize = 12)
    plt.yticks(FontSize = 10)
    #plt.ylim([0,1])
    plt.ylabel(data_type,  FontSize = 10)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['top'].set_linewidth(1.2)
    ax.tick_params(axis = 'y', length=3, width = 1.2)
    ax.tick_params(axis = 'x', length=3, width = 1.2)
    plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)
    save_name = save_add + '/' + 'All' + '_' + data_type + '_region_value.svg'
    plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)
    return df_bd_value_all_cut

df_bin_all_with_peak = pd.read_csv('E:/Users/dcdang/share/TAD_integrate/peak_overlap_with_bd/bin_region_all_with_peak.bed', sep = '\t', header = 0)

data_type = 'CTCF'
save_add = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/' + enzyme + '/method_evalute/boundary_score_and_value' + '/All_bd'
df_bd_value_all_cut = get_ratio_part_bd_peak_all(bd_score_final, cut_off_list, method_list, df_bin_all_with_peak, data_type, expand_size, mat_blank_row, random_num, random_round, save_add)


data_type = 'RAD21'
save_add = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/' + enzyme + '/method_evalute/boundary_score_and_value' + '/All_bd'
df_bd_value_all_cut = get_ratio_part_bd_peak_all(bd_score_final, cut_off_list, method_list, df_bin_all_with_peak, data_type, expand_size, mat_blank_row, random_num, random_round, save_add)


data_type = 'SMC3'
save_add = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/' + enzyme + '/method_evalute/boundary_score_and_value' + '/All_bd'
df_bd_value_all_cut = get_ratio_part_bd_peak_all(bd_score_final, cut_off_list, method_list, df_bin_all_with_peak, data_type, expand_size, mat_blank_row, random_num, random_round, save_add)


   
######## prove every method not cover all high score bd

def get_bd_score_interval_num(df_bd_cover, cut_off_list):
    num_l = []
    for cut_interval in cut_off_list:
        bd_num = np.sum(df_bd_cover['bd_score'].isin(cut_interval))
        num_l.append(bd_num)
    return num_l

def draw_bar_plot_bd_number_combine(df_cover_number, df_not_cover_number, cut_off_list, save_name):
    df_cover_number[-1] = [0 for i in range(len(df_cover_number))]
    #color_all = ['#2D84BB','#ADD8A4','#FAAE62', '#D72027']
    #color_all = ['#3288BD', '#99D594', '#E6F598', '#FEE08B', '#FC8D59', '#D53E4F']                    
    c_n = 2*len(cut_off_list)
    color_list = sns.diverging_palette(260, 20, n=c_n)   
    color_all_not_cover = list(reversed(color_list[0:len(cut_off_list)]))
    color_all_cover = color_list[len(cut_off_list):] 
    method_ord = list(df_cover_number['method'])
    method_ord_new = []
    for method in method_ord:
        if method != 'InsulationScore':
            method_ord_new.append(method)
        else:
            method_ord_new.append('IS')
    
    plt.figure(figsize=(8,5))
    bottom_list = np.zeros(len(method_ord))
    for i in reversed(range(len(cut_off_list))):
        plt.bar(range(len(method_ord)), list(df_not_cover_number[i]), align="center", bottom=list(bottom_list), color=color_all_not_cover[i], edgecolor = 'black', linewidth = 0.2)
        #plt.bar(range(len(method_ord)), list(df_not_cover_number[i]), align="center", bottom=list(bottom_list), color=color_all[i], edgecolor = 'black', linewidth = 0.2)
        bottom_list += np.array(df_not_cover_number[i])
    width = 0.8
    for i in range(len(method_ord)):
        plt.hlines(bottom_list[i], i - width / 2, i + width / 2, linestyles='solid', colors='black', linewidth = 2)
       
    for i in range(len(cut_off_list)):
        plt.bar(range(len(method_ord)), list(df_cover_number[i]), align="center", bottom=list(bottom_list), color=color_all_cover[i], edgecolor = 'black', linewidth = 0.2)
        #plt.bar(range(len(method_ord)), list(df_cover_number[i]), align="center", bottom=list(bottom_list), color=color_all[i], edgecolor = 'black', linewidth = 0.2)
        bottom_list += np.array(df_cover_number[i])
      
    plt.xticks(list(range(len(method_ord))), method_ord_new, rotation= -30, FontSize = 12)
    #plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'], FontSize = 12)
    #plt.ylim([0,1])
    plt.ylabel('Boundary number',  FontSize = 12)
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
    
def get_all_bd_method_cover_mat_new(bd_score_final, method_boundary_score, method_list, cut_off_list, save_name):
    df_all_bd_score = copy.deepcopy(bd_score_final)
    df_all_bd_score['index'] = list(range(len(df_all_bd_score)))
    df_all_bd_score_part = df_all_bd_score[df_all_bd_score['bd_score'] != 0]
    df_all_bd_score_part = df_all_bd_score_part .reset_index(drop = True)
    df_cover_score_mat = pd.DataFrame()
    bd_number_l = {}
    num_cover_all = []
    num_not_cover_all = []
    bd_cover_record = {}
    for method in method_list:
        print('This is ' + method)
        df_method_bd_score = method_boundary_score[method]
        df_bd_cover = pd.DataFrame(columns = ['index', 'bd_score'])
        df_bd_not_cover = pd.DataFrame(columns = ['index', 'bd_score'])
        cover_ind_l = []
        cover_ind_no_l = []
        cover_score_l = []
        cover_score_no_l = []
        bd_cover_record[method] = {}
        for i in range(len(df_all_bd_score_part)):
            ind = df_all_bd_score_part['index'][i]
            score = df_all_bd_score_part['bd_score'][i]
            if ind in list(df_method_bd_score['index']):
                cover_ind_l.append(ind)
                cover_score_l.append(score)
            else:
                cover_ind_no_l.append(ind)
                cover_score_no_l.append(score)
        df_bd_cover['index'] = cover_ind_l
        df_bd_cover['bd_score'] = cover_score_l
        df_bd_not_cover['index'] = cover_ind_no_l
        df_bd_not_cover['bd_score'] = cover_score_no_l

        df_bd_cover = df_bd_cover.sort_values(by = ['bd_score'], ascending=False)
        df_bd_cover = df_bd_cover.reset_index(drop = True)
        bd_number_l[method] = len(df_bd_cover)
        
        df_bd_not_cover = df_bd_not_cover.sort_values(by = ['bd_score'], ascending=True)
        df_bd_not_cover = df_bd_not_cover.reset_index(drop = True)
        bd_cover_record[method]['bd_cover'] = df_bd_cover
        bd_cover_record[method]['bd_not_cover'] = df_bd_not_cover
        
        score_list = list(df_bd_cover['bd_score']) + list(np.array(df_bd_not_cover['bd_score']) * (-1))
        #score_list = list(df_bd_cover['bd_score']) + list(np.array(df_bd_not_cover['bd_score']))
        
        #plt.plot(range(len(score_list)), score_list)
        df_cover_score_mat[method] = score_list

        num_l_cover = get_bd_score_interval_num(df_bd_cover, cut_off_list)
        num_l_not_cover = get_bd_score_interval_num(df_bd_not_cover, cut_off_list)
        num_cover_all.append(num_l_cover)
        num_not_cover_all.append(num_l_not_cover)
    df_cover_number = pd.DataFrame(np.array(num_cover_all))
    df_not_cover_number = pd.DataFrame(np.array(num_not_cover_all))
    df_cover_number['method'] = method_list   
    df_not_cover_number['method'] = method_list   

    high_score_bd_num = []
    for method in method_list:
        score_l = list(df_cover_score_mat[method])
        count=0
        for i in range(len(score_l)):
            if score_l[i] > 0:
            #if score_l[i] > 1:
                count += 1
            else:
                high_score_bd_num.append(count)
                break
    df_cover_number['ord'] = high_score_bd_num
    df_not_cover_number['ord'] = high_score_bd_num
    df_cover_number = df_cover_number.sort_values(by = ['ord'])
    df_cover_number = df_cover_number.reset_index(drop = True)
    df_not_cover_number = df_not_cover_number.sort_values(by = ['ord'])
    df_not_cover_number = df_not_cover_number.reset_index(drop = True)
    
    #save_name1 = 'E:/Users/dcdang/TAD_intergate/new_test/compare/result/DpnII/boundary_score_and_value/bd_all_cover_bar.svg'
    #save_name2 = 'E:/Users/dcdang/TAD_intergate/new_test/compare/result/DpnII/boundary_score_and_value/bd_all_not_cover_bar.svg'
    #draw_bar_plot_bd_number(df_cover_number, cut_off_list, save_name1, no_cover = True)
    #draw_bar_plot_bd_number(df_not_cover_number, cut_off_list, save_name2, no_cover = False)
    
    draw_bar_plot_bd_number_combine(df_cover_number, df_not_cover_number, cut_off_list, save_name)
    return bd_cover_record


cut_off_list2 = [[1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14,15,16]]

save_name = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/' + enzyme + '/method_evalute/boundary_score_and_value/bd_all_cover_and_noe_bar.svg'
bd_cover_record = get_all_bd_method_cover_mat_new(bd_score_final, method_boundary_score, method_list, cut_off_list2, save_name)


### distance between candidate bins to method boundary
def draw_bd_not_cover_distance_to_cover_bd(bd_cover_record, method_list, method_color, save_name, score_cut = 5, dist_cut = 3):
    num_row = 4
    num_col = 4
    fig = plt.figure(figsize=(16, 12)) 
    bd_not_cover_dist_record = {}
    for method in method_list:
        print('This is ' + method)
        m_ind = method_list.index(method)
        bd_cover_result = bd_cover_record[method]
        df_bd_cover = bd_cover_result['bd_cover']
        df_bd_not_cover = bd_cover_result['bd_not_cover']
        
        df_bd_not_cover_dist = pd.DataFrame(columns = ['score', 'distance'])
        score_list = []
        dist_list = []
        random_pos_add_l = []
        for i in range(len(df_bd_not_cover)):
            random_pos = random.uniform(-0.3,0.3)
            random_pos_add_l.append(random_pos)
            score_list.append(df_bd_not_cover['bd_score'][i])
            bd_index = df_bd_not_cover['index'][i]
            dist = np.min(np.abs(bd_index - df_bd_cover['index'] ))
            dist_list.append(dist)
        df_bd_not_cover_dist['score'] = score_list
        df_bd_not_cover_dist['distance'] = dist_list
        bd_not_cover_dist_record[method] = df_bd_not_cover_dist
        df_bd_not_cover_dist_part = df_bd_not_cover_dist[(df_bd_not_cover_dist['score'] >= score_cut) & (df_bd_not_cover_dist['distance'] >= dist_cut)]
        print('Boundary number with distance apart ' + str(len(df_bd_not_cover_dist_part)))
        fig.add_subplot(num_row, num_col, 1+m_ind)
        plt.scatter(np.array(df_bd_not_cover_dist['score'])+ np.array(random_pos_add_l), np.array(df_bd_not_cover_dist['distance']), c = method_color[method], s = 6, alpha=0.8)
        #plt.scatter(np.array(df_bd_not_cover_dist['score']), np.array(df_bd_not_cover_dist['distance']), c = method_color[method], s = 6, alpha=0.8)
        x_min = 1-0.5
        x_max = len(method_list)+0.5 
        y_min = np.min(df_bd_not_cover_dist['distance']) - 2
        y_max = np.max(df_bd_not_cover_dist['distance']) + 2
        plt.xlim([x_min,x_max])
        plt.ylim([y_min, y_max])
        plt.hlines(dist_cut, x_min, x_max, linestyles='--', colors='black', linewidth = 2)      
        plt.vlines(score_cut-0.5, y_min, y_max, linestyles='--', colors='black', linewidth = 2)
        plt.title(method, FontSize = 12)
        plt.xticks(FontSize = 10)
        if method in method_list[-num_col:]:
            plt.xticks(np.array(range(len(method_list))) + 1, FontSize = 10)
        else:
            plt.xticks(np.array(range(len(method_list))) + 1, FontSize = 0)
        if m_ind % num_row == 1:
            plt.ylabel('boundary distance (#bins)',  FontSize = 0)
        else:
            plt.ylabel('boundary distance (#bins)',  FontSize = 0)
        
        if m_ind % num_row == 1:
            plt.xlabel('boundary score', FontSize = 0)
        else:
            plt.xlabel('boundary score', FontSize = 0)
        plt.tight_layout()        
    plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)
    return bd_not_cover_dist_record
              
save_name = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/' + enzyme + '/method_evalute/boundary_score_and_value/bd_not_cover_score_distance.svg'
bd_not_cover_dist_record = draw_bd_not_cover_distance_to_cover_bd(bd_cover_record, method_list, method_color, save_name, score_cut = 5, dist_cut = 5)                   
                    

############# paird boundary score for TADs

def get_method_tad_bd_score(TAD_result_all, enzyme, boundary_score_final, method_list, bin_name_Chr):
    method_tad_bd_score = {}
    for method in method_list:
        print('This is ' + method)
        df_tad_m = TAD_result_all[enzyme][method]['TAD_domain']
        df_boundary_score = pd.DataFrame(columns = ['st_index', 'st_score', 'ed_index', 'ed_score'])
        st_ind_list = []
        st_score_list = []
        ed_ind_list = []
        ed_score_list = []
        tad_length_list = []
        for i in range(len(df_tad_m)):
            tad_length = df_tad_m['end'][i] - df_tad_m['start'][i]
            bd_st = df_tad_m['boundary_st'][i]
            bd_ed = df_tad_m['boundary_ed'][i]
            ind_st = bin_name_Chr.index(bd_st)
            ind_ed = bin_name_Chr.index(bd_ed)
            st_ind_list.append(ind_st)
            ed_ind_list.append(ind_ed)
            st_score_list.append(boundary_score_final['bd_score'][ind_st])
            ed_score_list.append(boundary_score_final['bd_score'][ind_ed])
            tad_length_list.append(tad_length)
        df_boundary_score['st_index'] = st_ind_list
        df_boundary_score['st_score'] = st_score_list
        df_boundary_score['ed_index'] = ed_ind_list
        df_boundary_score['ed_score'] = ed_score_list
        df_boundary_score['TAD_length'] = tad_length_list
        method_tad_bd_score[method] = df_boundary_score
    return method_tad_bd_score


method_tad_bd_score = get_method_tad_bd_score(TAD_result_all, enzyme, bd_score_final, method_list, bin_name_Chr)


def draw_tad_two_bd_score(method_tad_bd_score, method_list, method_color, n_level=10):
    num_row = 3
    num_col = 4
    density_top = [
    0.0225, 0.015, 0.012, 0.033, 
    0.015, 0.010, 0.012, 0.055,
    0.010, 0.012, 0.0135, 0.011,
    0.011, 0.022, 0.044, 0.0135
    ]
    for method in method_list:
        m_ind = method_list.index(method)
        df_bd_score_length = method_tad_bd_score[method]
        st_bin_score = list(df_bd_score_length['st_score'])
        ed_bin_score = list(df_bd_score_length['ed_score'])
        df_bd_score = pd.DataFrame(columns = ['bd1', 'bd2'])
        df_bd_score['bd1'] = st_bin_score + ed_bin_score
        df_bd_score['bd2'] = ed_bin_score + st_bin_score
        d_top = density_top[m_ind]
        level_l = []
        for j in range(n_level+1):
            level_l.append(d_top*j/n_level)
        #fig.add_subplot(gs[m_ind])
        #fig.add_subplot(num_row, num_col, 1+m_ind)
        g = sns.jointplot(x = 'bd1',y = 'bd2', data = df_bd_score, kind = 'kde', color = method_color[method], space=0, xlim = (-0.6,16), ylim = (-0.6,16), levels = level_l, cbar = True)
        #plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
        # get the current positions of the joint ax and the ax for the marginal x
        #pos_joint_ax = g.ax_joint.get_position()
        #pos_marg_x_ax = g.ax_marg_x.get_position()
        # reposition the joint ax so it has the same width as the marginal x ax
        #g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
        # reposition the colorbar using new x positions and y positions of the joint ax
        #g.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
        '''
        g = sns.JointGrid(x = 'bd1',y = 'bd2', data = df_bd_score, space=0, height = 6)
        g.plot_joint(sns.kdeplot, zorder=0, n_levels=10, shade = True, shade_lowest=True, color = method_color[method], clip = (-0.6,11), cbar = False) 
        sns.kdeplot(list(df_bd_score['bd1']), color= method_color[method], shade=True, linewidth=2, clip = (-0.6,11), ax=g.ax_marg_x)
        sns.kdeplot(list(df_bd_score['bd2']), color= method_color[method], shade=True, linewidth=2, clip = (-0.6,11), ax=g.ax_marg_y, vertical=True)
        plt.xlim([-0.6,11])
        plt.ylim([-0.6,11])
        #g7 = (sns.jointplot(x = 'bd1',y = 'bd2', data = df_bd_score, s = 0, color="k", marginal_kws=dict(rug = False, hist = False))
        #.plot_joint(sns.kdeplot, zorder=0, shade = True, shade_lowest=False, clip = (-1,11)))
        #g=sns.JointGrid(x = 'bd1',y = 'bd2', data = df_bd_score,space=0,ratio=5)
        #g = g.plot_joint(sns.kdeplot, color = method_color[method], shade = True)
        #g = g.plot_marginals(sns.kdeplot, color = method_color[method], shade=True)
        '''
        plt.title(method, FontSize = 15)
        #plt.xlabel('Boundary_1 score',  FontSize = 15)
        #plt.ylabel('Boundary_2 score',  FontSize = 15)
        ax=plt.gca()
        ax.spines['bottom'].set_linewidth(1.6)
        ax.spines['left'].set_linewidth(1.6)
        ax.spines['right'].set_linewidth(1.6)
        ax.spines['top'].set_linewidth(1.6)
        ax.tick_params(axis = 'y', length=7, width = 1.6)
        ax.tick_params(axis = 'x', length=3, width = 1.6)

        save_name = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/DpnII/method_evalute/TAD_two_bd_score/' + method + '_TAD_bd_score.svg'
        plt.savefig(save_name, format = 'svg', transparent = True) 
        plt.show()
        fig = plt.gcf() #获取当前figure
        plt.close(fig)
               
draw_tad_two_bd_score(method_tad_bd_score, method_list, method_color)



########################################### find case for method not good and plot the heatmap and TAD

def draw_heatmap_case_TAD(hic_mat_all_replicate, enzyme, datatype, method, Chr, start, end, TAD_list, color_use, region_list, save_name, resolution, bin_size = 10):
    region_color = 'black'
    dense_mat_fold_chr = hic_mat_all_replicate[enzyme][datatype]
    start = int(start / resolution)
    end = int(end / resolution)
    dense_matrix_part = dense_mat_fold_chr[start:end, start:end]
    #plt.figure(figsize=(10, 10))
    plt.figure(figsize=(4, 4))
    #plt.subplots(figsize=(4,4))
    ax = sns.heatmap(data = dense_matrix_part, square=True, yticklabels=False,  xticklabels=False, cmap='gist_gray_r',cbar_kws={"shrink": 0.8}, vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
    #cb.ax.tick_params(labelsize=16)
    if len(TAD_list) != 0:
        for TAD in TAD_list:
            ind_tad = TAD_list.index(TAD)
            color = color_use[ind_tad]
            st = int(TAD[0] / resolution) - start
            ed = int(TAD[1] / resolution) - start
            print(st, ed)
            draw_tad_region_upper_half(st, ed, color, size_v=3, size_h=3)
    else:
        print('No region to plot')
    
    if len(region_list) != 0:
        for region in region_list:
            st = int(region[0] / resolution) - start
            ed = int(region[1] / resolution) - start
            print(st, ed)
            draw_square_region(st, ed, region_color, size_v=3, size_h=3, row, col)
    else:
        print('No region to plot')
    st = start * resolution / 1000000
    ed = end * resolution / 1000000
    cord_list = []
    for i in range(end - start):
        if i % bin_size == 0:
            cord_list.append(i)
    plt.xticks(cord_list, FontSize = 20)
    ax.tick_params(axis='x',width=1.5, length = 5, colors='black')
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    region_name = Chr + ':' + str(st) + '-' + str(ed) + ' Mb'
    plt.title(region_name + ' TAD result of ' + method, FontSize = 10, pad = 20.0)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['top'].set_linewidth(1.2)
    ax.tick_params(axis = 'y', length=3, width = 1.2)
    ax.tick_params(axis = 'x', length=3, width = 1.2)  
    #plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

def draw_heatmap_case_TAD_new(hic_mat_all_replicate, bd_score_final, enzyme, datatype, method, Chr, start_o, end_o, TAD_list, color_use, region_list, save_name, resolution, bin_size = 10):
    st = int(start_o / resolution)
    ed = int(end_o / resolution)    
    
    plt.figure(figsize=(5,5.5))
    x_axis_range = range(len(bd_score_final['bd_score'][st:ed]))
    start_ =  st * resolution 
    end_ = ed * resolution
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
         
    ax1 = plt.subplot2grid((6, 6), (0, 0), rowspan=5,colspan=5)
    dense_mat_fold_chr = hic_mat_all_replicate[enzyme][datatype]
    start = int(start_ / resolution)
    end = int(end_ / resolution)
    dense_matrix_part = dense_mat_fold_chr[start:end, start:end]
    #plt.figure(figsize=(10, 10))
    #plt.figure(figsize=(4, 4))
    #ax = sns.heatmap(data = dense_matrix_part, square=True, yticklabels=False,  xticklabels=False, cmap='gist_gray_r', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
    #img = ax1.imshow(dense_matrix_part, cmap='gist_gray_r', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
    img = ax1.imshow(dense_matrix_part, cmap='gist_gray_r', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
    
    if len(TAD_list) != 0:
        for TAD in TAD_list:
            st_tad = int(TAD[0] / resolution) - start
            ed_tad = int(TAD[1] / resolution) - start
            ind_tad = TAD_list.index(TAD)
            color = color_use[ind_tad]
            print(st_tad, ed_tad)
            #draw_tad_region(st_tad, ed_tad, TAD_color, size_v=3, size_h=3)
            draw_tad_region_upper_half(st_tad, ed_tad, color, size_v=3, size_h=3)
    else:
        print('No TAD to plot')
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

    cax = plt.subplot2grid((6, 6), (0, 5), rowspan=5,colspan=1)
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes("right", size="1.5%", pad= 0.2)
    #cbar = plt.colorbar(img, cax=cax, ticks=MultipleLocator(2.0), format="%.1f",orientation='vertical',extendfrac='auto',spacing='uniform')
    cbaxes = inset_axes(cax, width="30%", height="80%", loc=6) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])

    ax5 = plt.subplot2grid((6, 6), (5, 0), rowspan=1,colspan=5,sharex=ax1)
    #ax5.plot(list(bd_score_final['bd_score'][st:ed]), color='black')
    ax5.bar(x_axis_range, list(bd_score_final['bd_score'][st:end]))
    ax5.spines['bottom'].set_linewidth(1.6)
    ax5.spines['left'].set_linewidth(1.6)
    ax5.spines['right'].set_linewidth(1.6)
    ax5.spines['top'].set_linewidth(1.6)
    ax5.tick_params(axis = 'y', length=5, width = 1.6)
    ax5.tick_params(axis = 'x', length=5, width = 1.6)
    ax5.set_ylabel('Bd score', FontSize = 10)
    
    #plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

def get_case_for_method_not_good(method_boundary_score, method_list, TAD_result_all, enzyme):
    TAD_score_dif_find_case = {}
    for method in method_list:
        df_method_score = copy.deepcopy(method_boundary_score[method])
        mean_score = []
        for i in range(len(df_method_score) - 1):
            if i % 2 == 0:
                score1 = df_method_score['score'][i]
                score2 = df_method_score['score'][i + 1]
                mean_score.append((score1 + score2) / 2)
            else:
                mean_score.append(0)
        mean_score.append(0)
        df_method_score['mean_score'] = mean_score
        score_diff = []
        for i in range(0, len(df_method_score)-1):
            if i == 0 :
                score_delta = df_method_score['mean_score'][i] - df_method_score['mean_score'][i+2] 
            elif i == len(df_method_score)-2:
                score_delta = df_method_score['mean_score'][i] - df_method_score['mean_score'][i-2]
            else:
                if i % 2 != 0:
                     score_delta = 50                   
                else:
                    score_delta1 = df_method_score['mean_score'][i] - df_method_score['mean_score'][i-2]
                    score_delta2 = df_method_score['mean_score'][i] - df_method_score['mean_score'][i+2] 
                    score_delta = score_delta1 + score_delta2
            score_diff.append(score_delta)
        score_diff.append(50)
        df_method_score_sort = copy.deepcopy(df_method_score)
        df_method_score_sort['score_diff'] = score_diff
        df_method_score_sort = df_method_score_sort.sort_values(by = ['score_diff'])  
        df_method_score_sort = df_method_score_sort.reset_index(drop = True)
        TAD_score_dif_find_case[method] = df_method_score_sort                
    return TAD_score_dif_find_case


TAD_score_dif_find_case = get_case_for_method_not_good(method_boundary_score, method_list, TAD_result_all, enzyme)


method_case_index = [(321, 559), [1176, 3362], [808, 266], [4770, 4151, 1026], [237, 4650], [3737, 3938, 3871],
                     [4324, 3322], [635], [4326, 2693, 4669], [3504, 1390], [857, 2035]]


i=15
method = method_list[i]
df_method_score_sort = copy.deepcopy(TAD_score_dif_find_case[method])

for j in range(240,270):
    print(j)  
    ind = df_method_score_sort['index'][j]
    
    st = ind * resolution
    ed = (ind+1) * resolution
    
    datatype = 'iced'
    TAD_color = method_color[method]
    df_tad = copy.deepcopy(TAD_result_all[enzyme][method]['TAD_domain'])
    Chr = 'chr2'
    start_o =  st - 800000
    end_o =  ed + 1000000
    #if start_o < 0 or end_o > chr_size:
        #continue
    TAD_region = get_related_tad(Chr, start_o, end_o, df_tad)
    if len(TAD_region) != 0:
        if TAD_region[0][0] < start_o:
            #TAD_region[0] = (start_o,TAD_region[0][-1])
            TAD_region = TAD_region[1:]
        if TAD_region[-1][-1] > end_o:
            #TAD_region[-1] = (TAD_region[-1][0], end_o)
            TAD_region = TAD_region[:-1]
       #TAD_region = TAD_region[1:-1]
    color_use = []
    for x in TAD_region:
        if x[0] != st and x[1] != ed:
            color_use.append(TAD_color)
        else:
            color_use.append('#FFFFFF')
        
    region_list = []
    save_name = 'E:/Users/dcdang/TAD_intergate/final_run/compare/Result/' + enzyme + '/method_evalute/case_not_good_and_missing' + '/' \
                + method + '_bad_case_' + str(ind) + '.svg' 
      
    draw_heatmap_case_TAD_new(hic_mat_all_replicate, bd_score_final, enzyme, datatype, method, Chr, start_o, end_o, TAD_region, color_use, region_list, save_name, resolution, bin_size = 10)
    #draw_heatmap_case_TAD(hic_mat_all_replicate, enzyme, datatype, method, Chr, start_o, end_o, TAD_region, color_use, region_list, save_name, resolution, bin_size = 10)

'''
# possible case index for 16 method
[31, 39]
[22, 84]
[211, 131]
[159, 99]
[0, 127]
[7,]
[868 , 1518]
[4, 273]
[, 120]
[702 , 179]
[807 , 78]
[190, 1540]
[11, 47]
[,177]
[24,25]
[263,]
'''
















