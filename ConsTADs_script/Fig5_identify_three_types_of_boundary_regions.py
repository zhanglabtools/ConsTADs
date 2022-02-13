# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 11:13:28 2022

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
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from matplotlib_venn import venn3, venn3_circles
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


###################### identify three types of boundary regions

def kmeans_SSE_with_max_Cnum(df_boundary_region_combine, K, save_name = ''):
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
    fig = plt.gcf() #获取当前figure
    plt.close(fig)
    

def kmeans_Score_with_max_Cnum(df_boundary_region_combine, K, save_name = ''):
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
 
    
def DBSCAN_cluster_with(df_boundary_region_combine, color_use = color_list16, permute=True):
    X = np.array(df_boundary_region_combine[['length', 'ave_score']])
    T = preprocessing.StandardScaler().fit(X)
    Xn=T.transform(X)
    # 试探不同参数下的聚类个数，没有得到合适的
    #for ep in [np.float(i / 20) for i in range(1, 20)]:
        #y_pred = DBSCAN(eps=ep).fit_predict(Xn)
        #print(len(np.unique(y_pred)))
    estimator = DBSCAN(eps=0.3, min_samples=5, metric='euclidean')
    estimator.fit(Xn)
    label_pred = estimator.labels_
    label_l = np.unique(label_pred)
    plt.figure(figsize=(5, 4.2))
    num = 0
    for l in label_l:
        num += 1
        x = X[label_pred == l]
        print(len(x))
        if permute == True:
            random_per = []
            for j in range(len(x)):
                random_per.append(random.uniform(-0.3,0.3))
            plt.scatter(x[:, 0] + np.array(random_per) , x[:, 1], c = color_use[num], marker='o', s=10, label=str(l))
        else:
            plt.scatter(x[:, 0], x[:, 1], c = color_use[l], marker='o', s=10, label=str(l))
    plt.legend(fontsize = 12)
    plt.xlabel('length',  FontSize = 12)
    plt.ylabel('Average Score',  FontSize = 12)
    plt.xticks(FontSize = 12)
    plt.yticks(FontSize = 12)
    #plt.ylim([0,1])
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)
    
    #plt.savefig(save_name, format = 'svg', transparent = True) 
    #plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)
 
def kmeans_cluster_with_fixed_Cnum(df_boundary_region_combine, K, color_use = color_list16, save_name = '', permute=True):
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
        print(len(x))
        length_r_l.append(np.mean(x[:, 0]))
        score_r_l.append(np.mean(x[:, 1]))    
    ind_len_max = np.argmax(length_r_l)
    ind_score_max = np.argmax(score_r_l)
    region_type = {}
    region_type[ind_len_max] = 'wide'
    region_type[ind_score_max] = 'sharp_strong'
    for k in range(K):
        if k == ind_len_max or k == ind_score_max:
            continue
        else:
            region_type[k] = 'sharp_weak'
    print(region_type)
    region_type_l = []
    for i in range(len(label_pred)):
        region_type_l.append(region_type[label_pred[i]])
    df_boundary_region_combine['region_type'] = list(region_type_l)
    
    plt.figure(figsize=(5, 4.2))
    type_list_r = ['sharp_strong', 'sharp_weak', 'wide']
    for i in range(len(type_list_r)) :
        type_r = type_list_r[i]
        x = X[df_boundary_region_combine['region_type'] == type_r]
        print(len(x))
        if permute == True:
            random_per = []
            for j in range(len(x)):
                random_per.append(random.uniform(-0.3,0.3))
            plt.scatter(x[:, 0] + np.array(random_per) , x[:, 1], c = color_use[i], marker='o', s=10, label=type_r)
        else:
            plt.scatter(x[:, 0], x[:, 1], c = color_use[i], marker='o', s=10, label=type_r)
    plt.legend(fontsize = 12, frameon = False)
    plt.xlabel('length',  FontSize = 12)
    plt.ylabel('Average Score',  FontSize = 12)
    plt.xticks(FontSize = 12)
    plt.yticks(FontSize = 12)
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
    plt.close(fig)    
    return df_boundary_region_combine 
  
    
def get_cut_off_adjust_region_label(df_boundary_region_with_type, color_use = color_list16, save_name = '', permute=True):    
    df_wide_part = df_boundary_region_with_type[df_boundary_region_with_type['region_type'] == 'wide']
    cut_length = np.percentile(df_wide_part['length'], 10)    
    df_sharp_strong_part = df_boundary_region_with_type[df_boundary_region_with_type['region_type'] == 'sharp_strong']    
    cut_score = np.percentile(df_sharp_strong_part['ave_score'], 10)
    cut_score = np.ceil(cut_score)
    print('Length cut-off:' + str(cut_length))
    print('Score cut-off:' + str(cut_score))
    region_type_adjust = []
    for i in range(len(df_boundary_region_with_type)):
        length = df_boundary_region_with_type['length'][i]
        score = df_boundary_region_with_type['ave_score'][i]
        if length >= cut_length:
            region_type_adjust.append('wide')
        elif score >= cut_score:
            region_type_adjust.append('sharp_strong')
        else:
            region_type_adjust.append('sharp_weak')
    df_boundary_region_with_type['region_type_adjust'] = region_type_adjust
    print('Adjust number:' + str(np.sum(df_boundary_region_with_type['region_type_adjust'] != df_boundary_region_with_type['region_type'])))
    
    plt.figure(figsize=(5, 4.2))
    X = np.array(df_boundary_region_with_type[['length', 'ave_score']])
    type_list_r = ['sharp_strong', 'sharp_weak', 'wide']
    for i in range(len(type_list_r)) :
        type_r = type_list_r[i]
        x = X[df_boundary_region_with_type['region_type_adjust'] == type_r]
        print(len(x))
        if permute == True:
            random_per = []
            for j in range(len(x)):
                random_per.append(random.uniform(-0.3,0.3))
            plt.scatter(x[:, 0] + np.array(random_per) , x[:, 1], c = color_use[i], marker='o', s=10, label=type_r)
        else:
            plt.scatter(x[:, 0], x[:, 1], c = color_use[i], marker='o', s=10, label=type_r)
    plt.legend(fontsize = 12, frameon = False)
    plt.xlabel('length',  FontSize = 12)
    plt.ylabel('Average Score',  FontSize = 12)
    plt.xticks(FontSize = 12)
    plt.yticks(FontSize = 12)
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
    plt.close(fig)  
    return df_boundary_region_with_type
            

def draw_bd_region_number(bd_region_type_record, color_bd_use, save_name):
    #type_ord = ['wide', 'sharp_weak', 'sharp_strong']
    type_ord = ['sharp_weak','sharp_strong','wide', ]
    df_number = pd.DataFrame(columns = ['cell_type', 'sharp_strong', 'sharp_weak', 'wide', 'number'])
    cell_l = list(bd_region_type_record.keys())
    sharp_sl = []
    sharp_wl = []
    wide_l = [] 
    num_l = []
    cell_l_use = []
    for sample in cell_l:
        if sample == 'GM12878_DpnII':
            continue
        df_cell_bd_region = copy.deepcopy(bd_region_type_record[sample])       
        sharp_s_num = np.sum(df_cell_bd_region['region_type_adjust'] == 'sharp_strong')
        sharp_w_num = np.sum(df_cell_bd_region['region_type_adjust'] == 'sharp_weak')
        wide_num = np.sum(df_cell_bd_region['region_type_adjust'] == 'wide')
        sharp_sl.append(sharp_s_num)
        sharp_wl.append(sharp_w_num)
        wide_l.append(wide_num)
        num_l.append(len(df_cell_bd_region))
        cell_l_use.append(sample)
    df_number['cell_type'] = cell_l_use
    df_number['sharp_strong'] = sharp_sl
    df_number['sharp_weak'] = sharp_wl
    df_number['wide'] = wide_l
    df_number['number'] = num_l
    
    df_number = df_number.sort_values(by = ['number'])
    
    plt.figure(figsize=(5,5))
    bottom_list = np.zeros(len(cell_l_use))
    for i in range(len(type_ord)):
        bd_type = type_ord[i]
        plt.bar(range(len(cell_l_use)), list(df_number[bd_type]), align="center", bottom=list(bottom_list), color=color_bd_use[bd_type], edgecolor = 'black', linewidth = 1.5)
        bottom_list += np.array(df_number[bd_type])
    
    plt.xticks(list(range(len(cell_l_use))), list(df_number['cell_type']), rotation= -30, FontSize = 12)
    #plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'], FontSize = 12)
    #plt.ylim([0,1])
    plt.ylabel('Number of boundary region',  FontSize = 12)
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
    
    

#### singel sample test 
             
cell_type = 'GM12878'
enzyme = 'MboI'
mat_dense = copy.deepcopy(hic_mat_all_cell_replicate[cell_type][enzyme]['iced'])
df_bd_insul_pvalue = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['pvalue'])
result_record = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['BD_region'])

bd_score_cell = copy.deepcopy(boundary_score_cell_all[cell_type][enzyme])
df_boundary_region_original = copy.deepcopy(result_record['Original']['bd_region'])
df_boundary_region_add = copy.deepcopy(result_record['Add']['bd_region'])
bd_score_cell_add = copy.deepcopy(result_record['Add']['TAD_score'])
df_boundary_region_adjust = copy.deepcopy(result_record['Adjust']['bd_region'])
bd_score_cell_adjust = copy.deepcopy(result_record['Adjust']['TAD_score'])
df_boundary_region_combine = copy.deepcopy(result_record['Combine']['bd_region'])
bd_score_cell_combine = copy.deepcopy(result_record['Combine']['TAD_score'])


## decide cluster number and clustering by K-means
save_name = ''
kmeans_SSE_with_max_Cnum(df_boundary_region_combine, K=9, save_name = '')
kmeans_Score_with_max_Cnum(df_boundary_region_combine, K=9, save_name = '')
df_boundary_region_with_type = kmeans_cluster_with_fixed_Cnum(df_boundary_region_combine, K = 3, color_use = color_list16, save_name = '', permute = True)
df_boundary_region_with_type = get_cut_off_adjust_region_label(df_boundary_region_with_type, color_use = color_list16, save_name = '', permute=True)


#### multi-cell deal
bd_region_save_add = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region'
color_bd = ['#D65F4D', '#459457', '#4392C3']

bd_region_type_record = {}
for cell_type in cell_type_list:
    print('This is ' + cell_type)
    cell_color_use = cell_color[cell_type]
    cell_save_add = bd_region_save_add + '/identify/' + cell_type
    if not os.path.exists(cell_save_add):
        os.makedirs(cell_save_add)
    if cell_type == 'GM12878':
        enzyme_list = ['DpnII', 'MboI']
    else:
        enzyme_list = ['MboI']
    for enzyme in enzyme_list:
        print('For ' + enzyme)
        result_record = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['BD_region'])
        df_boundary_region_combine = copy.deepcopy(result_record['Combine']['bd_region'])
        bd_score_cell_combine = copy.deepcopy(result_record['Combine']['TAD_score'])
        save_name1 = cell_save_add + '/' + cell_type + '_' + enzyme + '_SSE_plot.svg'
        kmeans_SSE_with_max_Cnum(df_boundary_region_combine, K=9, save_name = save_name1)
        save_name2 = cell_save_add + '/' + cell_type + '_' + enzyme + '_sh_score_plot.svg'
        kmeans_Score_with_max_Cnum(df_boundary_region_combine, K=9, save_name = save_name2)
        save_name3 = cell_save_add + '/' + cell_type + '_' + enzyme + '_kmeans_plot.svg'
        df_boundary_region_with_type = kmeans_cluster_with_fixed_Cnum(df_boundary_region_combine, 3, color_use = color_bd, save_name = save_name3, permute = True)
        save_name4 = cell_save_add + '/' + cell_type + '_' + enzyme + '_kmeans_adjust_plot.svg'
        df_boundary_region_with_type = get_cut_off_adjust_region_label(df_boundary_region_with_type, color_use = color_bd, save_name = save_name4, permute=True)
        bd_region_type_record[cell_type + '_' + enzyme] = df_boundary_region_with_type
        print('\n')


color_bd_use = {'sharp_strong':'#D65F4D', 'sharp_weak':'#459457', 'wide':'#4392C3'}
save_name_num = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\identify' + '/' + 'bd_region_number.svg'
df_cell_region_number = draw_bd_region_number(bd_region_type_record, color_bd_use, save_name = save_name_num)

save_data(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape' + '/' + 'boundary_region_type_for_all_cell_type.pkl', bd_region_type_record)



##### Analysis processing....

######### aggregate map and case map

def draw_contact_map_and_score_square(contact_map, bd_score_show, cell_line, region_color_use, save_name = '', type_ = 'center', pos_draw = '', resolution = 50000):
    if len(contact_map) == 41:
        x_range = [0,10,20,30,40]
    elif len(contact_map) == 21:
        x_range = [0,5,10,15,20]
    elif len(contact_map) == 19:
        x_range = [0, 4, 9, 14, 18]
    elif len(contact_map) == 15:
        x_range = [0, 4, 7, 10, 14]

    plt.figure(figsize=(5,5.5))    
    ax1 = plt.subplot2grid((5, 5), (0, 0), rowspan=4,colspan=4)    
    if type_ == 'single':
        img = ax1.imshow(contact_map, cmap='seismic', vmin = np.percentile(contact_map, 5), vmax = np.percentile(contact_map, 80))    
    else:
        vmin = np.percentile(contact_map, 0)
        vmax = np.percentile(contact_map, 100)    
        norm = mcolors.DivergingNorm(vmin=vmin, vcenter = 0, vmax=vmax)
        img = ax1.imshow(contact_map, cmap='seismic', vmin = vmin, vmax = vmax, norm = norm)      
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
    ax5.fill_between(list(range(len(bd_score_show))), 0, bd_score_show, color = region_color_use)
    #ax5.bar(list(range(len(bd_score_show))), bd_score_show)
    ##plt.ylim([-1,1])
    #if np.max(bd_score_show) >= 5:
        #plt.hlines(5, np.min(x_range), np.max(x_range), linestyles = '--', linewidth=1.6)
    #if np.min(bd_score_show) <= -5:
        #plt.hlines(-5, np.min(x_range), np.max(x_range), linestyles = '--', linewidth=1.6)
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

def max_norm_region_mat(mat_region):
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
    #mat_region_new = np.zeros([len(mat_region), len(mat_region)])
    #for k in range(len(mat_region)):
        #diag_vec = np.diag(mat_region, k=k)
        #diag_vec = diag_vec / np.mean(diag_vec)
        #mat_region_new += np.diag(diag_vec, k = k)
        #if k != 0:
            #mat_region_new += np.diag(diag_vec, k = -k)        
    return mat_region

def get_blank_row_in_matrix(mat_dense):
    df_row_sum = pd.DataFrame(np.sum(mat_dense, axis = 0))
    df_row_sum_part = df_row_sum[df_row_sum[0]==0]
    blank_row = list(df_row_sum_part.index)
    return blank_row

def aggregate_plot_heatmap_for_bd_region(df_bd_region_type_part, mat_dense1, bd_cell_score1, cell_type, region_color_use, save_name = '', mat_len = 10):    
    mat_dense_11 = np.zeros([2*mat_len+1, 2*mat_len+1])
    score_vec_11 = []
    for i in range(len(df_bd_region_type_part)):
        region = df_bd_region_type_part['region'][i]
        mid_ind = int((region[0] + region[-1]) / 2)
        if mid_ind < mat_len + 1 or mid_ind > len(mat_dense1) - mat_len -1:
            continue    
        mat_region1 = copy.deepcopy(mat_dense1[mid_ind - mat_len : mid_ind + mat_len+1, mid_ind - mat_len : mid_ind + mat_len+1])        
        score_11 = bd_cell_score1['bd_score'].iloc[mid_ind - mat_len : mid_ind + mat_len+1]
        score_vec_11.append(score_11)

        mat_region1_norm = max_norm_region_mat(mat_region1)
        mat_dense_11 += mat_region1_norm

    mat_dense_11 = mat_dense_11 / np.sum(mat_dense_11)   
    score_show_11 = np.array(score_vec_11)
    score_show_11 = np.mean(score_show_11, axis = 0)
    draw_contact_map_and_score_square(mat_dense_11, score_show_11, cell_type, region_color_use, save_name = save_name, type_ = 'single', pos_draw = '', resolution = 50000)


def compare_CTCF_peak_numbers_bd_region(df_bd_region_type, df_CTCF_peak_num_bin_chr2_part_norm, cell, mat_blank_row, region_type, region_color, save_name, expand_len = 15, random_num = (10, 300)):   
    df_ctcf_pos = pd.DataFrame(columns = ['ctcf_l', 'pos_l', 'type_l'])
    ctcf_l = []
    pos_l = []
    type_l = []
    count = 0
    for i in range(len(df_bd_region_type)):
        region = df_bd_region_type['region'][i]
        score1 = df_bd_region_type['score'][i]
        type_ = df_bd_region_type['region_type_adjust'][i]
        if type_ not in region_type:
            continue
        mid1 = region[np.argmax(score1)]
        #if type_ == 'wide':
            #mid1 = mid1 + 1
        #mid1 = int((region[0] + region[-1]) / 2)
        if mid1 < expand_len + 1 or mid1 > len(df_CTCF_peak_num_bin_chr2_part_norm) - expand_len -1:
            continue
        dist = []
        for site in range(mid1 - expand_len, mid1+expand_len+1):
            dist_s = np.abs(np.array(mat_blank_row) - site)
            dist.append(np.min(dist_s))
        if np.min(dist) <= 2:
            continue
        count += 1
        ctcf_vec1 = df_CTCF_peak_num_bin_chr2_part_norm[cell + '_CTCF_peak_num'][mid1 - expand_len : mid1+expand_len+1]
        
        ctcf_l += list(ctcf_vec1)
        pos_l += list(range(len(ctcf_vec1)))
        type_l += [type_ for j in range(len(ctcf_vec1))]
    print(count)    
    random_vec_all = {}
    for j in range(random_num[0]):
        random_index = []
        random_vec_l = []
        random_sample = 0
        while random_sample < random_num[-1]:
            mid2 = random.randint(0, len(df_CTCF_peak_num_bin_chr2_part_norm)-1)
            if mid2 < expand_len + 1 or mid2 > len(df_CTCF_peak_num_bin_chr2_part_norm) - expand_len -1:
                continue
            if len(random_index) == 0:
                dist = 100
            else:
                dist = np.abs(np.array(random_index) - mid2)
            if np.min(dist) <= 2:
                continue
            ctcf_vec2 = df_CTCF_peak_num_bin_chr2_part_norm[cell + '_CTCF_peak_num'][mid2 - expand_len : mid2+expand_len+1]
            random_vec_l.append(list(ctcf_vec2))
            random_sample += 1
        random_vec_all[j] = random_vec_l
    
    random_vec_mean = []
    for k in range(len(random_vec_all[0])):
        random_use = []
        for j in range(random_num[0]):
            random_use.append(random_vec_all[j][k])
        random_use = np.array(random_use)
        random_vec_mean.append(np.mean(random_use, axis = 0))
    
    for i in range(len(random_vec_mean)):
        ctcf_vec2 = random_vec_mean[i]
        ctcf_l += list(ctcf_vec2)
        pos_l += list(range(len(ctcf_vec2)))
        type_l += ['random' for j in range(len(ctcf_vec2)) ]
    
    df_ctcf_pos['ctcf_l'] = ctcf_l
    df_ctcf_pos['pos_l'] = pos_l
    df_ctcf_pos['type_l'] = type_l    
    
    list_type = region_type + ['random']        
    df_ctcf_pos['type_l'] = df_ctcf_pos['type_l'].astype('category')
    df_ctcf_pos['type_l'].cat.reorder_categories(list_type, inplace=True)
    df_ctcf_pos.sort_values('type_l', inplace=True)        
    color_l = [] 
    for type_ in region_type:
        color_l.append(region_color[type_])
    color_l.append('#B0B0B0')
                     
    plt.figure(figsize=(6.5,5))
    #sns.lineplot(x = 'pos_l', y = 'ctcf_l', data = df_ctcf_pos, hue = 'type_l', palette = color_l, markers=True, dashes=False, linewidth = 3)
    sns.lineplot(x = 'pos_l', y = 'ctcf_l', data = df_ctcf_pos, hue = 'type_l', style = 'type_l', palette = color_l, markers=True, dashes=False, linewidth = 3)
    if expand_len == 10: 
        plt.xticks([0, 5, 10, 15, 20], ['-500kb', '-250kb', 'boundary center', '250kb', '500kb'], FontSize = 10)
    elif expand_len == 7:
        plt.xticks([0, 3, 7, 11, 15], ['-350kb', '-200kb', 'boundary center', '200kb', '350kb'], FontSize = 10)
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

def compare_smc3_rad21_peak_numbers_bd_region(df_bd_region_type, df_bin_all_with_peak, peak_type, cell, mat_blank_row, region_type, region_color, save_name = '', expand_len = 15, random_num = (10, 300)):   
    df_ctcf_pos = pd.DataFrame(columns = ['ctcf_l', 'pos_l', 'type_l'])
    ctcf_l = []
    pos_l = []
    type_l = []
    count = 0
    for i in range(len(df_bd_region_type)):
        region = df_bd_region_type['region'][i]
        score1 = df_bd_region_type['score'][i]
        type_ = df_bd_region_type['region_type_adjust'][i]
        if type_ not in region_type:
            continue
        mid1 = region[np.argmax(score1)]
        if type_ == 'wide':
            mid1 = mid1 + 1
        #mid1 = int((region[0] + region[-1]) / 2)
        if mid1 < expand_len + 1 or mid1 > len(df_bin_all_with_peak) - expand_len -1:
            continue
        dist = []
        for site in range(mid1 - expand_len, mid1+expand_len+1):
            dist_s = np.abs(np.array(mat_blank_row) - site)
            dist.append(np.min(dist_s))
        if np.min(dist) <= 2:
            continue
        count += 1
        ctcf_vec1 = df_bin_all_with_peak[peak_type][mid1 - expand_len : mid1+expand_len+1]        
        ctcf_l += list(ctcf_vec1)
        pos_l += list(range(len(ctcf_vec1)))
        type_l += [type_ for j in range(len(ctcf_vec1))]
    print(count)    
    
    random_vec_all = {}
    for j in range(random_num[0]):
        random_index = []
        random_vec_l = []
        random_sample = 0
        while random_sample < random_num[-1]:
            mid2 = random.randint(0, len(df_bin_all_with_peak)-1)
            if mid2 < expand_len + 1 or mid2 > len(df_bin_all_with_peak) - expand_len -1:
                continue
            if len(random_index) == 0:
                dist = 100
            else:
                dist = np.abs(np.array(random_index) - mid2)
            if np.min(dist) <= 2:
                continue
            ctcf_vec2 = df_bin_all_with_peak[peak_type][mid2 - expand_len : mid2+expand_len+1]
            random_vec_l.append(list(ctcf_vec2))
            random_sample += 1
        random_vec_all[j] = random_vec_l    
    random_vec_mean = []
    for k in range(len(random_vec_all[0])):
        random_use = []
        for j in range(random_num[0]):
            random_use.append(random_vec_all[j][k])
        random_use = np.array(random_use)
        random_vec_mean.append(np.mean(random_use, axis = 0))    
    for i in range(len(random_vec_mean)):
        ctcf_vec2 = random_vec_mean[i]
        ctcf_l += list(ctcf_vec2)
        pos_l += list(range(len(ctcf_vec2)))
        type_l += ['random' for j in range(len(ctcf_vec2)) ]
    
    df_ctcf_pos['ctcf_l'] = ctcf_l
    df_ctcf_pos['pos_l'] = pos_l
    df_ctcf_pos['type_l'] = type_l
    
    list_type = region_type + ['random']        
    df_ctcf_pos['type_l'] = df_ctcf_pos['type_l'].astype('category')
    df_ctcf_pos['type_l'].cat.reorder_categories(list_type, inplace=True)
    df_ctcf_pos.sort_values('type_l', inplace=True)            
    color_l = [] 
    for type_ in region_type:
        color_l.append(region_color[type_])
    color_l.append('#B0B0B0')        
    plt.figure(figsize=(6.5,5))
    #sns.lineplot(x = 'pos_l', y = 'ctcf_l', data = df_ctcf_pos, hue = 'type_l', palette = color_l, markers=True, linewidth = 3)
    sns.lineplot(x = 'pos_l', y = 'ctcf_l', data = df_ctcf_pos, hue = 'type_l', style = 'type_l', palette = color_l, markers=True, dashes=False, linewidth = 3)
    if expand_len == 10: 
        plt.xticks([0, 5, 10, 15, 20], ['-500kb', '-250kb', 'boundary center', '250kb', '500kb'], FontSize = 10)
    elif expand_len == 7:
        plt.xticks([0, 3, 7, 11, 15], ['-350kb', '-200kb', 'boundary center', '200kb', '350kb'], FontSize = 10)
    plt.yticks(FontSize = 10)
    #plt.ylim([0.3, 0.95])
    plt.ylabel(peak_type + ' peaks fold change',  fontSize = 12)
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

def compare_indictor_bd_region(df_bd_region_type, indictor_sample, indictor_type, mat_blank_row, region_type, region_color, save_name = '', expand_len = 15, random_num = (10, 300)):   
    df_indictor_pos = pd.DataFrame(columns = ['indictor_l', 'pos_l', 'type_l'])
    indictor_l = []
    pos_l = []
    type_l = []
    df_indictor = indictor_sample[indictor_type]
    
    for i in range(len(df_bd_region_type)):
        region = df_bd_region_type['region'][i]
        score1 = df_bd_region_type['score'][i]
        type_ = df_bd_region_type['region_type_adjust'][i]
        if type_ not in region_type:
            continue
        mid1 = region[np.argmax(score1)]
        #if type_ == 'wide':
            #mid1 = mid1 + 1
        #mid1 = int((region[0] + region[-1]) / 2)
        if mid1 < expand_len + 1 or mid1 > len(df_indictor) - expand_len -1:
            continue                
        dist = []
        for site in range(mid1 - expand_len, mid1+expand_len+1):
            dist_s = np.abs(np.array(mat_blank_row) - site)
            dist.append(np.min(dist_s))
        if np.min(dist) <= 2:
            continue        
        ind_vec1 = df_indictor[indictor_type][mid1 - expand_len : mid1+expand_len+1]        
        indictor_l += list(ind_vec1)
        pos_l += list(range(len(ind_vec1)))
        type_l += [type_ for j in range(len(ind_vec1)) ]

    random_vec_all = {}
    for j in range(random_num[0]):
        random_index = []
        random_vec_l = []
        random_sample = 0
        while random_sample < random_num[-1]:
            mid2 = random.randint(0, len(df_indictor)-1)
            if mid2 < expand_len + 1 or mid2 > len(df_indictor) - expand_len -1:
                continue
            if len(random_index) == 0:
                dist_past = 100
            else:
                dist_past = np.abs(np.array(random_index) - mid2)
            if np.min(dist_past) <= 2:
                continue
            dist = []
            for site in range(mid2 - expand_len, mid2+expand_len+1):
                dist_s = np.abs(np.array(mat_blank_row) - site)
                dist.append(np.min(dist_s))
            if np.min(dist) <= 2:
                continue
            ind_vec2 = df_indictor[indictor_type][mid2 - expand_len : mid2+expand_len+1]
            random_vec_l.append(list(ind_vec2))
            random_sample += 1
        random_vec_all[j] = random_vec_l    
    random_vec_mean = []
    for k in range(len(random_vec_all[0])):
        random_use = []
        for j in range(random_num[0]):
            random_use.append(random_vec_all[j][k])
        random_use = np.array(random_use)
        random_vec_mean.append(np.mean(random_use, axis = 0))    
    for i in range(len(random_vec_mean)):
        ind_vec2 = random_vec_mean[i]
        indictor_l += list(ind_vec2)
        pos_l += list(range(len(ind_vec2)))
        type_l += ['random' for j in range(len(ind_vec2)) ]
   
    df_indictor_pos['indictor_l'] = indictor_l
    df_indictor_pos['pos_l'] = pos_l
    df_indictor_pos['type_l'] = type_l
    
    list_type = region_type + ['random']        
    df_indictor_pos['type_l'] = df_indictor_pos['type_l'].astype('category')
    df_indictor_pos['type_l'].cat.reorder_categories(list_type, inplace=True)
    df_indictor_pos.sort_values('type_l', inplace=True)            
    color_l = [] 
    for type_ in region_type:
        color_l.append(region_color[type_])
    color_l.append('#B0B0B0')    
    plt.figure(figsize=(6,5))
    sns.lineplot(x = 'pos_l', y = 'indictor_l', data = df_indictor_pos, hue = 'type_l',  style="type_l", palette = color_l, markers=True, dashes=False, linewidth = 3, ci = 95)
    if expand_len == 10: 
        plt.xticks([0, 5, 10, 15, 20], ['-500kb', '-250kb', 'boundary center', '250kb', '500kb'], FontSize = 10)
    elif expand_len == 7:
        plt.xticks([0, 3, 7, 11, 15], ['-350kb', '-200kb', 'boundary center', '200kb', '350kb'], FontSize = 10)
    plt.yticks(FontSize = 10)
    #if indictor_type == 'DI':
        #plt.ylim([-100, 100])
    plt.ylabel(indictor_type,  fontSize = 12)
    plt.xlabel('',  fontSize = 0)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    #plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)  
    plt.legend(loc = 'best', prop = {'size':10}, fancybox = None, edgecolor = 'white', facecolor = None, title = False, title_fontsize = 0)
    if save_name != '':    
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    plt.close(fig)   

### one cell type test

cell_type = 'GM12878'
enzyme = 'MboI'
mat_dense1 = copy.deepcopy(hic_mat_all_cell_replicate[cell_type][enzyme]['iced'])
mat_blank_row = get_blank_row_in_matrix(mat_dense1)

df_bd_insul_pvalue = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['pvalue'])
result_record = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['BD_region'])

df_boundary_region_combine = copy.deepcopy(result_record['Combine']['bd_region'])
bd_cell_score1 = copy.deepcopy(result_record['Combine']['TAD_score'])


df_bd_region_type = copy.deepcopy(bd_region_type_record[cell_type + '_' + enzyme])

region_type = ['sharp_strong', 'sharp_weak', 'wide']
region_color = {'sharp_strong':'#D65F4D', 'sharp_weak':'#459457', 'wide':'#4392C3'}

r_type = region_type[0]
region_color_use = region_color[r_type]
df_bd_region_type_part = df_bd_region_type[df_bd_region_type['region_type_adjust'] == r_type]
df_bd_region_type_part = df_bd_region_type_part.reset_index(drop = True)


aggregate_plot_heatmap_for_bd_region(df_bd_region_type_part, mat_dense1, bd_cell_score1, cell_type, region_color_use, descr = '', mat_len = 10)


### CTCF peak and mean score

df_CTCF_peak_num_bin_chr2 = pd.read_csv('E:/Users/dcdang/share/TAD_integrate/CTCF_peak_overlap_bin/human_chr2_50000_bin_with_CTCF_peak_multiple_cells.bed', sep = '\t', header = 0)

cell_with_CTCF_peak = ['GM12878_CTCF_peak_num', 'K562_CTCF_peak_num',
       'IMR90_CTCF_peak_num', 'HUVEC_CTCF_peak_num', 'HMEC_CTCF_peak_num',
       'NHEK_CTCF_peak_num']

df_CTCF_peak_num_bin_chr2_part = df_CTCF_peak_num_bin_chr2[cell_with_CTCF_peak]

CTCF_sum = np.sum(df_CTCF_peak_num_bin_chr2_part, axis = 0)
df_CTCF_peak_num_bin_chr2_part_norm = copy.deepcopy(df_CTCF_peak_num_bin_chr2_part)
for i in range(len(df_CTCF_peak_num_bin_chr2_part.columns)):
    col_name = list(df_CTCF_peak_num_bin_chr2_part.columns)[i]
    df_CTCF_peak_num_bin_chr2_part_norm[col_name] = df_CTCF_peak_num_bin_chr2_part_norm[col_name] / (CTCF_sum[i] / len(df_CTCF_peak_num_bin_chr2_part_norm)) 

region_type = ['sharp_strong', 'sharp_weak', 'wide']
compare_CTCF_peak_numbers_bd_region(df_bd_region_type, df_CTCF_peak_num_bin_chr2_part_norm, cell_type, mat_blank_row, region_type, region_color, save_name = '', expand_len = 7)

### three kinds of peaks  (This CTCF peaks is intersection of four peak files, above is one peak file result)
df_bin_all_with_peak = pd.read_csv('E:/Users/dcdang/share/TAD_integrate/peak_overlap_with_bd/bin_region_all_with_peak.bed', sep = '\t', header = 0)

peak_type_l = ['CTCF', 'RAD21', 'SMC3']
df_bin_all_with_peak_fold = copy.deepcopy(df_bin_all_with_peak)
for peak_type in peak_type_l:
    df_bin_all_with_peak_fold[peak_type] = df_bin_all_with_peak[peak_type] / (np.sum(df_bin_all_with_peak[peak_type]) / len(df_bin_all_with_peak))

region_type = ['sharp_strong', 'sharp_weak', 'wide']
peak_type = 'CTCF'
compare_smc3_rad21_peak_numbers_bd_region(df_bd_region_type, df_bin_all_with_peak_fold, peak_type, cell_type, mat_blank_row, region_type, region_color, save_name = '', expand_len = 7)


### indictor for bd region
indictor_sample = indictor_record_all_cell[cell_type][enzyme]
indictor_type = 'DI'
region_type = ['sharp_strong', 'sharp_weak', 'wide']

compare_indictor_bd_region(df_bd_region_type, indictor_sample, indictor_type, mat_blank_row, region_type, region_color, save_name = '', expand_len = 10)




### multi cell type run

Aggregrate_add = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\Aggregrate_map'
Indicator_add = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\Indicator'

region_type = ['sharp_strong', 'sharp_weak', 'wide']
region_color = {'sharp_strong':'#D65F4D', 'sharp_weak':'#459457', 'wide':'#4392C3'}

                
df_CTCF_peak_num_bin_chr2 = pd.read_csv('E:/Users/dcdang/share/TAD_integrate/CTCF_peak_overlap_bin/human_chr2_50000_bin_with_CTCF_peak_multiple_cells.bed', sep = '\t', header = 0)
cell_with_CTCF_peak = ['GM12878_CTCF_peak_num', 'K562_CTCF_peak_num',
       'IMR90_CTCF_peak_num', 'HUVEC_CTCF_peak_num', 'HMEC_CTCF_peak_num',
       'NHEK_CTCF_peak_num']
df_CTCF_peak_num_bin_chr2_part = df_CTCF_peak_num_bin_chr2[cell_with_CTCF_peak]
CTCF_sum = np.sum(df_CTCF_peak_num_bin_chr2_part, axis = 0)
df_CTCF_peak_num_bin_chr2_part_norm = copy.deepcopy(df_CTCF_peak_num_bin_chr2_part)
for i in range(len(df_CTCF_peak_num_bin_chr2_part.columns)):
    col_name = list(df_CTCF_peak_num_bin_chr2_part.columns)[i]
    df_CTCF_peak_num_bin_chr2_part_norm[col_name] = df_CTCF_peak_num_bin_chr2_part_norm[col_name] / (CTCF_sum[i] / len(df_CTCF_peak_num_bin_chr2_part_norm)) 

                
df_bin_all_with_peak = pd.read_csv('E:/Users/dcdang/share/TAD_integrate/peak_overlap_with_bd/bin_region_all_with_peak.bed', sep = '\t', header = 0)
peak_type_l = ['CTCF', 'RAD21', 'SMC3']
df_bin_all_with_peak_fold = copy.deepcopy(df_bin_all_with_peak)
for peak_type in peak_type_l:
    df_bin_all_with_peak_fold[peak_type] = df_bin_all_with_peak[peak_type] / (np.sum(df_bin_all_with_peak[peak_type]) / len(df_bin_all_with_peak))
 

                              
for cell_type in cell_type_list:
    print('This is ' + cell_type)
    if cell_type == 'GM12878':
        enzyme_list = ['DpnII', 'MboI']
    else:
        enzyme_list = ['MboI']
    for enzyme in enzyme_list:
        print('For ' + enzyme)        
        Aggregrate_save_add = Aggregrate_add + '/' + cell_type + '_' + enzyme        
        if not os.path.exists(Aggregrate_save_add):
            os.makedirs(Aggregrate_save_add)
        Indicator_peak_save_add = Indicator_add + '/peaks/' + cell_type + '_' + enzyme        
        if not os.path.exists(Indicator_peak_save_add):
            os.makedirs(Indicator_peak_save_add)
        Indicator_indicator_save_add = Indicator_add + '/DI_IS_CI/' + cell_type + '_' + enzyme        
        if not os.path.exists(Indicator_indicator_save_add):
            os.makedirs(Indicator_indicator_save_add)
            
        mat_dense1 = copy.deepcopy(hic_mat_all_cell_replicate[cell_type][enzyme]['iced'])
        mat_blank_row = get_blank_row_in_matrix(mat_dense1)
        df_bd_insul_pvalue = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['pvalue'])
        result_record = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['BD_region'])
        df_boundary_region_combine = copy.deepcopy(result_record['Combine']['bd_region'])
        bd_cell_score1 = copy.deepcopy(result_record['Combine']['TAD_score'])
        df_bd_region_type = copy.deepcopy(bd_region_type_record[cell_type + '_' + enzyme])

        for r_type in region_type:        
            region_color_use = region_color[r_type]           
            df_bd_region_type_part = df_bd_region_type[df_bd_region_type['region_type_adjust'] == r_type]
            df_bd_region_type_part = df_bd_region_type_part.reset_index(drop = True)
            
            save_name1 = Aggregrate_save_add + '/' + cell_type + '_' + enzyme + '_' + r_type + '_aggregrate_map.svg'
            aggregate_plot_heatmap_for_bd_region(df_bd_region_type_part, mat_dense1, bd_cell_score1, cell_type, region_color_use, save_name = save_name1, mat_len = 9)
        
        if cell_type != 'KBM7':
            save_name2 = Indicator_peak_save_add + '/' + cell_type + '_' + enzyme + '_CTCF_FC_compare.svg'
            compare_CTCF_peak_numbers_bd_region(df_bd_region_type, df_CTCF_peak_num_bin_chr2_part_norm, cell_type, mat_blank_row, region_type, region_color, save_name = save_name2, expand_len = 7)
        
        if cell_type == 'GM12878':
            for peak_type in ['CTCF', 'SMC3', 'RAD21']:
                save_name3 = Indicator_peak_save_add + '/' + cell_type + '_' + enzyme + '_' + peak_type + '_peaks_FC_compare.svg'           
                compare_smc3_rad21_peak_numbers_bd_region(df_bd_region_type, df_bin_all_with_peak_fold, peak_type, cell_type, mat_blank_row, region_type, region_color, save_name = save_name3, expand_len = 7)
            
        indictor_sample = indictor_record_all_cell[cell_type][enzyme]
        for indictor_type in ['DI', 'IS', 'CI']:           
            save_name4 = Indicator_indicator_save_add + '/' + cell_type + '_' + enzyme + '_' + indictor_type  + '_compare.svg'           
            compare_indictor_bd_region(df_bd_region_type, indictor_sample, indictor_type, mat_blank_row, region_type, region_color, save_name = save_name4, expand_len = 10)



##### bio data analysis
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



def draw_bd_region_bio_profile(df_bd_region_type, df_human_chr_bio_GM12878, bio_type, region_type, region_color, save_name = '', expand_len = 15, random_num = (10, 300)):
    df_biodata_pos = pd.DataFrame(columns = ['biodata_l', 'pos_l', 'type_l'])
    biodata_l = []
    pos_l = []
    type_l = []
    df_biodata = pd.DataFrame(copy.deepcopy(df_human_chr_bio_GM12878[bio_type]))
    df_biodata[bio_type] = df_biodata[bio_type] / np.mean(df_biodata[bio_type])

    for i in range(len(df_bd_region_type)):
        region = df_bd_region_type['region'][i]
        score1 = df_bd_region_type['score'][i]
        type_ = df_bd_region_type['region_type_adjust'][i]
        if type_ not in region_type:
            continue
        #mid1 = region[np.argmax(score1)]
        #if type_ == 'wide':
            #mid1 = mid1 + 1
        mid1 = int((region[0] + region[-1]) / 2)
        if mid1 < expand_len + 1 or mid1 > len(df_biodata) - expand_len -1:
            continue                
        dist = []
        for site in range(mid1 - expand_len, mid1+expand_len+1):
            dist_s = np.abs(np.array(mat_blank_row) - site)
            dist.append(np.min(dist_s))
        if np.min(dist) <= 2:
            continue        
        ind_vec1 = df_biodata[bio_type][mid1 - expand_len : mid1+expand_len+1]        
        biodata_l += list(ind_vec1)
        pos_l += list(range(len(ind_vec1)))
        type_l += [type_ for j in range(len(ind_vec1)) ]
    
    random_vec_all = {}
    for j in range(random_num[0]):
        random_index = []
        random_vec_l = []
        random_sample = 0
        while random_sample < random_num[-1]:
            mid2 = random.randint(0, len(df_biodata)-1)
            if mid2 < expand_len + 1 or mid2 > len(df_biodata) - expand_len -1:
                continue
            if len(random_index) == 0:
                dist_past = 100
            else:
                dist_past = np.abs(np.array(random_index) - mid2)
            if np.min(dist_past) <= 2:
                continue
            dist = []
            for site in range(mid2 - expand_len, mid2+expand_len+1):
                dist_s = np.abs(np.array(mat_blank_row) - site)
                dist.append(np.min(dist_s))
            if np.min(dist) <= 2:
                continue
            ind_vec2 = df_biodata[bio_type][mid2 - expand_len : mid2+expand_len+1]
            random_vec_l.append(list(ind_vec2))
            random_sample += 1
        random_vec_all[j] = random_vec_l    
    random_vec_mean = []
    for k in range(len(random_vec_all[0])):
        random_use = []
        for j in range(random_num[0]):
            random_use.append(random_vec_all[j][k])
        random_use = np.array(random_use)
        random_vec_mean.append(np.mean(random_use, axis = 0))    
    for i in range(len(random_vec_mean)):
        ind_vec2 = random_vec_mean[i]
        biodata_l += list(ind_vec2)
        pos_l += list(range(len(ind_vec2)))
        type_l += ['random' for j in range(len(ind_vec2)) ]
        
    df_biodata_pos['biodata_l'] = biodata_l
    df_biodata_pos['pos_l'] = pos_l
    df_biodata_pos['type_l'] = type_l
    
    list_type = region_type + ['random']        
    df_biodata_pos['type_l'] = df_biodata_pos['type_l'].astype('category')
    df_biodata_pos['type_l'].cat.reorder_categories(list_type, inplace=True)
    df_biodata_pos.sort_values('type_l', inplace=True)            
    color_l = [] 
    for type_ in region_type:
        color_l.append(region_color[type_])
    color_l.append('#B0B0B0')    
        
    plt.figure(figsize=(6,5))
    sns.lineplot(x = 'pos_l', y = 'biodata_l', data = df_biodata_pos, hue = 'type_l',  style="type_l", palette = color_l, markers=True, dashes=False, linewidth = 3, ci = 95)
    if expand_len == 10: 
        plt.xticks([0, 5, 10, 15, 20], ['-500kb', '-250kb', 'boundary center', '250kb', '500kb'], FontSize = 10)
    elif expand_len == 7:
        plt.xticks([0, 3, 7, 11, 15], ['-350kb', '-200kb', 'boundary center', '200kb', '350kb'], FontSize = 10)
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

for bio_type in list(df_human_chr_bio_GM12878.columns):       
    draw_bd_region_bio_profile(df_bd_region_type, df_human_chr_bio_GM12878, bio_type, region_type, region_color, save_name = '', expand_len = 15, random_num = (10, 300))       
        
       
bio_target_l = ['RepG1',
 'RepG2',
 'RepS1',
 'RepS2',
 'RepS3',
 'RepS4',
 'RepWave']

bio_target_l = ['YY1']
bio_target_l = ['CTCF', 'SMC3', 'RAD21']
bio_target_l = ['H3K36me3', 'H3K79me2']
bio_target_l = bio_name_list


cell_type = 'GM12878'
enzyme = 'MboI'
df_bd_region_type = copy.deepcopy(bd_region_type_record[cell_type + '_' + enzyme])

for bio_type in bio_target_l:
    save_name = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\repli-seq_bd_region' + '/' + cell_type + '/' + cell_type + '_' + enzyme + '_' + bio_type + '_bd_region_profile.svg'
    draw_bd_region_bio_profile(df_bd_region_type, df_human_chr_bio_GM12878, bio_type, region_type, region_color, save_name = save_name, expand_len = 30, random_num = (20, 300))       
        
       
def get_bd_region_bio_fold_change(df_bd_region_type, df_human_chr_bio, bio_name_type_list):
    bio_data_list = list(df_human_chr_bio.columns)
    df_bd_bio_value = pd.DataFrame(columns = bio_data_list)
    for bio_data in bio_data_list:
        value_result = []
        for i in range(len(df_bd_region_type)):
            region = df_bd_region_type['region'][i]
            target = df_human_chr_bio[bio_data].iloc[region]
            value_result.append(np.mean(target))  
        df_bd_bio_value[bio_data] = value_result
    df_bd_bio_value['bd_type'] = df_bd_region_type['region_type_adjust']

    label_all = ['wide', 'sharp_strong', 'sharp_weak']
    df_bd_bio_FC = pd.DataFrame(columns = label_all)
    
    for label in label_all:
        FC_result = []
        df_bd_bio_value_part = df_bd_bio_value[df_bd_bio_value['bd_type'] == label]
        df_bd_bio_value_part = df_bd_bio_value_part.reset_index(drop = True)
        for bio_data in bio_data_list:
            #median_bg = np.median(df_bd_bio_value[bio_data])
            median_bg = np.median(df_human_chr_bio[bio_data])
            median_bd = np.median(df_bd_bio_value_part[bio_data])
            FC_result.append(np.log2(median_bd / median_bg) )
            #FC_result.append(median_bd / median_bg)
        df_bd_bio_FC[label] = FC_result
    df_bd_bio_FC.index = bio_data_list
    df_bd_bio_FC['data_type'] = bio_name_type_list
    sns.clustermap(df_bd_bio_FC[label_all], center = 0, row_cluster=False, col_cluster=False, cmap = 'coolwarm')
    return df_bd_bio_FC


bio_save_add = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\bio_enrich'

cell_type = 'GM12878'
enzyme = 'MboI'
df_bd_region_type = copy.deepcopy(bd_region_type_record[cell_type + '_' + enzyme])

df_bd_bio_FC_GM12878 = get_bd_region_bio_fold_change(df_bd_region_type, df_human_chr_bio_GM12878, bio_name_type_list)


df_bd_bio_FC_GM12878 = df_bd_bio_FC_GM12878.reindex(index=['H2A.Z', 'H3K27me3', 'H3K9me3', 'H3K27ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K36me3', 'H3K79me2', 'H4K20me1', 
           'Pol2', 'CTCF', 'RAD21', 'SMC3', 'YY1', 'Methylation', 'DNase','neg', 'pos', 
            'RepG1', 'RepS1', 'RepS2', 'RepS3', 'RepS4', 'RepG2', 'RepWave'])

df_bd_bio_FC_GM12878.to_csv(bio_save_add+ '/' + cell_type + '_' + enzyme + '_bd_region_bio_data_fold_change.bed', sep = '\t', header = True, index = True)



bio_data_add = 'E:/Users/dcdang/share/TAD_integrate/K562_data_for_use/bigwig_bed_file'
bio_type_list = ['epigenetics', 'RNA-seq', 'Repli-seq']

df_human_chr_bio_K562 = pd.DataFrame()
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
                df_human_chr_bio_K562[bio_name] = df_bio_data[0]
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
            df_human_chr_bio_K562[bio_name] = df_bio_data[0]
            bio_name_list.append(bio_name)
            bio_name_type_list.append(bio_name_type)


cell_type = 'K562'
enzyme = 'MboI'
df_bd_region_type = copy.deepcopy(bd_region_type_record[cell_type + '_' + enzyme])

df_bd_bio_FC_K562 = get_bd_region_bio_fold_change(df_bd_region_type, df_human_chr_bio_K562, bio_name_type_list)



df_bd_bio_FC_K562 = df_bd_bio_FC_K562.reindex(index=['H2A.Z', 'H3K27me3', 'H3K9me3', 'H3K27ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K36me3', 'H3K79me2', 'H4K20me1', 
           'Pol2', 'CTCF', 'RAD21', 'SMC3', 'YY1', 'Methylation', 'DNase','neg', 'pos', 
            'RepG1', 'RepS1', 'RepS2', 'RepS3', 'RepS4', 'RepG2', 'RepWave'])
df_bd_bio_FC_K562.to_csv(bio_save_add+ '/' + cell_type + '_' + enzyme + '_bd_region_bio_data_fold_change.bed', sep = '\t', header = True, index = True)


#### gene deal

Genecode_v10_add = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\data_use\Roadmap_gene_RNA_seq'

names = list(range(10))
df_gene_info = pd.read_csv(Genecode_v10_add + '/' + 'Gencode_V10_gene_information.txt', sep = '\t', header = None, names=names)
df_gene_info = df_gene_info.fillna('None')
df_gene_info.columns = ['gene_id', 'chr', 'start', 'end', 'strand', 'type', 'gene_name', 'info', 'None1', 'None2']
df_gene_info['chr'] = 'chr' + np.array(df_gene_info['chr'])
df_gene_info = df_gene_info.sort_values(by = ['chr', 'start'])
df_gene_info = df_gene_info.reset_index(drop = True)


df_gene_reads_cell_lines = pd.read_csv(Genecode_v10_add + '/' + '57epigenomes.N.pc', sep = '\t',  header = 0)

df_gene_FPKM_cell_lines = pd.read_csv(Genecode_v10_add + '/' + '57epigenomes.RPKM.pc', sep = '\t',  header = 0)

col_old = list(df_gene_reads_cell_lines.columns)
col_name = ['gene_id'] + col_old[2:]

df_gene_reads_cell_lines = df_gene_reads_cell_lines[col_old[:-1]]
df_gene_reads_cell_lines.columns = col_name

df_gene_FPKM_cell_lines = df_gene_FPKM_cell_lines[col_old[:-1]]
df_gene_FPKM_cell_lines.columns = col_name


Road_map_cell_label = {'GM12878':'E116', 'HMEC':'E119', 'HUVEC':'E122', 'IMR90':'E017', 'K562':'E123', 'KBM7':'None', 'NHEK':'E127'}
df_cell_gene_reads = pd.DataFrame(columns = ['GM12878', 'HMEC', 'HUVEC', 'K562', 'NHEK']) 
df_cell_gene_FPKM = pd.DataFrame(columns = ['GM12878', 'HMEC', 'HUVEC', 'K562', 'NHEK'])
for cell_type in ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'K562', 'NHEK']:
    if cell_type == 'IMR90':
        continue
    cell_label = Road_map_cell_label[cell_type]   
    df_cell_gene_reads[cell_type] = copy.deepcopy(df_gene_reads_cell_lines[cell_label])
    df_cell_gene_FPKM[cell_type] = copy.deepcopy(df_gene_FPKM_cell_lines[cell_label])

df_cell_gene_reads['gene_id'] = df_gene_reads_cell_lines['gene_id']
df_cell_gene_FPKM['gene_id'] = df_gene_FPKM_cell_lines['gene_id']




gene_info_index_l = []
for i in range(len(df_cell_gene_reads)):
    gene_id = df_cell_gene_reads['gene_id'][i]
    gene_index = list(df_gene_info['gene_id']).index(gene_id)
    gene_info_index_l.append(gene_index)    

df_gene_info_part = copy.deepcopy(df_gene_info.iloc[gene_info_index_l])
df_gene_info_part = df_gene_info_part.reset_index(drop = True)
df_gene_info_part = df_gene_info_part[['gene_id', 'chr', 'start', 'end', 'strand', 'type', 'gene_name', 'info']]
df_gene_info_part.columns = ['gene_id_check', 'chr', 'start', 'end', 'strand', 'type', 'gene_name', 'info']


df_cell_gene_reads_combine = pd.concat([df_gene_info_part, df_cell_gene_reads], axis = 1)
df_cell_gene_reads_combine = df_cell_gene_reads_combine.sort_values(by = ['chr', 'start'])
df_cell_gene_reads_combine = df_cell_gene_reads_combine.reset_index(drop = True)
print(np.sum(df_cell_gene_reads_combine['gene_id'] == df_cell_gene_reads_combine['gene_id_check']))

df_cell_gene_FPKM_combine = pd.concat([df_gene_info_part, df_cell_gene_FPKM], axis = 1)
df_cell_gene_FPKM_combine = df_cell_gene_FPKM_combine.sort_values(by = ['chr', 'start'])
df_cell_gene_FPKM_combine = df_cell_gene_FPKM_combine.reset_index(drop = True)
print(np.sum(df_cell_gene_FPKM_combine['gene_id'] == df_cell_gene_FPKM_combine['gene_id_check']))

save_add = r'E:\Users\dcdang\share\TAD_integrate\bd_region_overlap_gene'
df_cell_gene_FPKM_combine[['chr', 'start', 'end', 'strand', 'type', 'gene_name',
        'GM12878', 'HMEC', 'HUVEC', 'K562', 'NHEK', 'gene_id']].to_csv(save_add + '/' + 'Gencode_V10_gene_info.bed', sep = '\t', header = None, index = None)



cell_type = 'NHEK'
enzyme = 'MboI'
df_bd_region_type = copy.deepcopy(bd_region_type_record[cell_type + '_' + enzyme])

df_bd_region_type_over_gene = copy.deepcopy(df_bd_region_type)
df_bd_region_type_over_gene['start'] = df_bd_region_type_over_gene['start'] * resolution 
df_bd_region_type_over_gene['end'] = df_bd_region_type_over_gene['end'] * resolution + resolution
df_bd_region_type_over_gene.to_csv(save_add + '/' + cell_type + '_' + enzyme + '_bd_region_with_type.bed', sep = '\t', header = None, index = None )


df_bd_region_with_gene = pd.read_csv(save_add + '/' + cell_type + '_' + enzyme + '_bd_region_with_gene.bed', sep = '\t', header = None)

df_bd_region_with_gene.columns = list(df_bd_region_type.columns) + ['chr_g', 'start_g', 'end_g', 'strand', 'type', 'gene_name',
        'GM12878', 'HMEC', 'HUVEC', 'K562', 'NHEK', 'gene_id'] + ['overlap']

    
def compare_bd_region_gene(df_bd_region_with_gene, cell_type, region_type, region_color, save_name = ''):    
    df_gene_in_region = pd.DataFrame(columns = ['gene_name', 'FPKM', 'region_type'])
    gene_l = []
    fpkm_l = []
    r_type_l = []
    for i in range(len(df_bd_region_with_gene)):
        if df_bd_region_with_gene['chr_g'][i] == '.':
            continue
        gene_l.append(df_bd_region_with_gene['gene_name'][i])
        fpkm_l.append(np.float(df_bd_region_with_gene[cell_type][i]))
        r_type_l.append(df_bd_region_with_gene['region_type_adjust'][i])
    df_gene_in_region['gene_name'] = gene_l
    df_gene_in_region['FPKM'] = fpkm_l
    df_gene_in_region['region_type'] = r_type_l
    
    list_type = ['sharp_weak', 'sharp_strong', 'wide']       
    df_gene_in_region['region_type'] = df_gene_in_region['region_type'].astype('category')
    df_gene_in_region['region_type'].cat.reorder_categories(list_type, inplace=True)
    df_gene_in_region.sort_values('region_type', inplace=True)        
        
    SS_value = np.array(df_gene_in_region[df_gene_in_region['region_type'] == 'sharp_strong']['FPKM'])
    SW_value = np.array(df_gene_in_region[df_gene_in_region['region_type'] == 'sharp_weak']['FPKM'])
    Wi_value = np.array(df_gene_in_region[df_gene_in_region['region_type'] == 'wide']['FPKM'])
    
    sta1, pvalue1 = scipy.stats.mannwhitneyu(SS_value, SW_value)
    print('sharp_strong and sharp weak:')
    print(sta1, pvalue1)
    sta2, pvalue2 = scipy.stats.mannwhitneyu(Wi_value, SS_value)
    print('wide and sharp strong:')
    print(sta2, pvalue2)
    sta3, pvalue3 = scipy.stats.mannwhitneyu(Wi_value, SW_value)
    print('wide and sharp weak:')
    print(sta3, pvalue3)
    
    color_l = [] 
    for type_ in list_type:
        color_l.append(region_color[type_])
    
    plt.figure(figsize=(3,3))
    ax1 = sns.barplot(x = 'region_type', y = 'FPKM', data = df_gene_in_region, 
                capsize = 0.2, saturation = 8,             
            errcolor = 'black', errwidth = 1.5,  
            ci = 95, edgecolor='black', palette = color_l, alpha = 1)
    
    plt.yticks(FontSize = 10)
    plt.xticks(FontSize = 10)
    plt.title(cell_type)
    plt.ylabel('FPKM',  fontSize = 10)
    plt.xlabel('',  fontSize = 0)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    #plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)  
    #plt.legend(loc = 'best', prop = {'size':10}, fancybox = None, edgecolor = 'white', facecolor = None, title = False, title_fontsize = 0)
    if save_name != '':    
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)   
        


cell_type = 'K562'
df_bd_region_with_gene = pd.read_csv(save_add + '/' + cell_type + '_' + enzyme + '_bd_region_with_gene.bed', sep = '\t', header = None)

df_bd_region_with_gene.columns = list(df_bd_region_type.columns) + ['chr_g', 'start_g', 'end_g', 'strand', 'type', 'gene_name',
        'GM12878', 'HMEC', 'HUVEC', 'K562', 'NHEK', 'gene_id'] + ['overlap']
    
compare_bd_region_gene(df_bd_region_with_gene, cell_type, region_type, region_color)


save_fig_add = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\bd_region_gene'
for cell_type in ['GM12878', 'K562', 'HMEC', 'HUVEC', 'NHEK']:
    print(cell_type)
    df_bd_region_with_gene = pd.read_csv(save_add + '/' + cell_type + '_' + enzyme + '_bd_region_with_gene.bed', sep = '\t', header = None)
    
    df_bd_region_with_gene.columns = list(df_bd_region_type.columns) + ['chr_g', 'start_g', 'end_g', 'strand', 'type', 'gene_name',
            'GM12878', 'HMEC', 'HUVEC', 'K562', 'NHEK', 'gene_id'] + ['overlap']
    save_name = save_fig_add + '/' + cell_type + '_' + enzyme + '_bd_region_gene_expression_2.svg'   
    compare_bd_region_gene(df_bd_region_with_gene, cell_type, region_type, region_color, save_name = save_name)
    print('\n')


### chromHMM state enrich
state_add = r'E:\Users\dcdang\share\TAD_integrate\bd_region_overlap_gene\bd_region_chrom_state'


for cell_type in ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'K562', 'NHEK', 'KBM7']:
    enzyme = 'MboI'
    df_bd_region_type = copy.deepcopy(bd_region_type_record[cell_type + '_' + enzyme])
    
    df_bd_region_type_chrom_state = copy.deepcopy(df_bd_region_type)
    df_bd_region_type_chrom_state['start'] = df_bd_region_type_chrom_state['start'] * resolution 
    df_bd_region_type_chrom_state['end'] = df_bd_region_type_chrom_state['end'] * resolution + resolution
    df_bd_region_type_chrom_state.to_csv(state_add + '/' + cell_type + '_' + enzyme + '_bd_region_with_type.bed', sep = '\t', header = None, index = None )

### 中间过程在虚拟机内完成了

state_result_add = r'E:\Users\dcdang\share\TAD_integrate\bd_region_overlap_gene\bd_region_chrom_state\result_add'

bd_region_state_15_cell_all = {}
bd_region_state_18_cell_all = {}
for cell_type in ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'K562', 'NHEK']:
    enzyme = 'MboI'
    file_15 = state_result_add + '/' + cell_type + '_' + enzyme + '__15_state_FC_result.pkl'
    cell_state_15 = read_save_data(file_15)
    
    file_18 = state_result_add + '/' + cell_type + '_' + enzyme + '__18_state_FC_result.pkl'
    cell_state_18 = read_save_data(file_18)
    
    bd_region_state_15_cell_all[cell_type + '_' + enzyme] = cell_state_15
    bd_region_state_18_cell_all[cell_type + '_' + enzyme] = cell_state_18
    


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


#### segway result
    
    
def get_bd_region_state_FC_segway(cell_type, enzyme, bd_region_state_5_cell_all, region_type = ['sharp_strong', 'sharp_weak', 'wide']):        
    cell_FC_result = bd_region_state_5_cell_all[cell_type + '_' + enzyme]    
    state_all_order = ['QUI', 'CON', 'FAC', 'BRD', 'SPC']
    #state_all_order = ['BRD', 'SPC', 'FAC', 'CON','QUI',]
    region_type_ord = ['wide', 'sharp_strong', 'sharp_weak']
    df_bd_region_state_FC = pd.DataFrame(columns = region_type_ord)
    for r_type in region_type_ord:
        state_FC = []
        bd_region_state_dic = cell_FC_result[r_type]
        for state in state_all_order:
            state_FC.append(bd_region_state_dic[state])      
        df_bd_region_state_FC[r_type] = state_FC    
    df_bd_region_state_FC.index = state_all_order    
    sns.clustermap(df_bd_region_state_FC, center = 0, row_cluster=False, col_cluster=False, cmap = 'coolwarm')
    return df_bd_region_state_FC      
    
    
      
state_result_add = r'E:\Users\dcdang\share\TAD_integrate\bd_region_overlap_gene\bd_region_chrom_state\result_segway'

bd_region_state_5_cell_all = {}
for cell_type in ['GM12878', 'HUVEC', 'IMR90', 'K562']:
    enzyme = 'MboI'
    file_5 = state_result_add + '/' + cell_type + '_' + enzyme + '__5_segway_state_FC_result.pkl'
    cell_state_5 = read_save_data(file_5)
    
    bd_region_state_5_cell_all[cell_type + '_' + enzyme] = cell_state_5

      
save_add = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\State_enrich\Segway'

for cell_type in ['GM12878', 'HUVEC', 'IMR90', 'K562']:
    enzyme = 'MboI'    
    df_bd_region_5_state_FC = get_bd_region_state_FC_segway(cell_type, enzyme, bd_region_state_5_cell_all, region_type = ['sharp_strong', 'sharp_weak', 'wide'])  
    save_name5 = save_add + '/' + cell_type + '_' + enzyme + '_Segway_5_state_FC.bed'
    df_bd_region_5_state_FC.to_csv(save_name5, sep = '\t', header = True, index = True)


### subcompartment enrich  
def get_bd_region_state_FC_subcompartment(cell_type, enzyme, bd_region_state_5_cell_all, region_type = ['sharp_strong', 'sharp_weak', 'wide']):        
    cell_FC_result = bd_region_state_5_cell_all[cell_type + '_' + enzyme]    
    state_all_order = ['A1', 'A2', 'B1', 'B2', 'B3']
    region_type_ord = ['wide', 'sharp_strong', 'sharp_weak']
    df_bd_region_state_FC = pd.DataFrame(columns = region_type_ord)
    for r_type in region_type_ord:
        state_FC = []
        bd_region_state_dic = cell_FC_result[r_type]
        for state in state_all_order:
            state_FC.append(bd_region_state_dic[state])      
        df_bd_region_state_FC[r_type] = state_FC    
    df_bd_region_state_FC.index = state_all_order    
    sns.clustermap(df_bd_region_state_FC, center = 0, row_cluster=False, col_cluster=False, cmap = 'coolwarm')
    return df_bd_region_state_FC      
    
 
state_result_add = r'E:\Users\dcdang\share\TAD_integrate\bd_region_overlap_gene\bd_region_chrom_state\result_subcom'

bd_region_subcompartment_cell_all = {}
for cell_type in ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'K562']:
    enzyme = 'MboI'
    file_sc = state_result_add + '/' + cell_type + '_' + enzyme + '__subcompartment_FC_result.pkl'
    cell_state_sc = read_save_data(file_sc)
    
    bd_region_subcompartment_cell_all[cell_type + '_' + enzyme] = cell_state_sc

 
    
save_add = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\State_enrich\Subcompartment'

for cell_type in ['GM12878', 'HMEC','HUVEC', 'IMR90', 'K562']:
    enzyme = 'MboI'    
    df_bd_region_sc_FC = get_bd_region_state_FC_subcompartment(cell_type, enzyme, bd_region_subcompartment_cell_all, region_type = ['sharp_strong', 'sharp_weak', 'wide'])  
    save_name_sc = save_add + '/' + cell_type + '_' + enzyme + '_subcompartment_FC.bed'
    df_bd_region_sc_FC.to_csv(save_name_sc, sep = '\t', header = True, index = True)


############## repeat element

def get_bd_repeat_zscore(df_bd_region_type, df_cell_bd_repeat_num, dict_random_bd_repeat, type_, col_all):
    zscore_l = []
    for i in range(len(df_cell_bd_repeat_num)):
        if i !=0  and i % 50 == 0:
            print('50 region done!')
        record_region = []
        for j in range(len(dict_random_bd_repeat)):
            df_random_bd_repeat = dict_random_bd_repeat[j][type_]
            record_region.append(list(df_random_bd_repeat.iloc[i]))
        record_region = np.array(record_region)
      
        mean_vec = np.mean(record_region, axis = 0)
        std_vec = np.std(record_region, axis = 0)
        zscore_vec = (np.array(df_cell_bd_repeat_num.iloc[i]) - mean_vec) / std_vec
        zscore_l.append(zscore_vec)
        
    df_cell_bd_repeat_zscore = pd.DataFrame(np.array(zscore_l))
    df_cell_bd_repeat_zscore.columns = col_all 
    df_cell_bd_repeat_zscore['bd_region_type'] = df_bd_region_type['region_type_adjust']
    return df_cell_bd_repeat_zscore

def draw_repeat_enrich_in_bd_region_class(col_all, df_cell_bd_repeat_zscore, region_color, save_name = ''):
    df_bd_repeat_for_draw = pd.DataFrame(columns = ['zscore', 'type', 'bd_type'])
    zscore_l = []
    type_l = []
    bd_type_l = []
    for col in col_all:
        zscore_l += list(df_cell_bd_repeat_zscore[col])
        type_l += [col for i in range(len(df_cell_bd_repeat_zscore))]
        bd_type_l += list(df_cell_bd_repeat_zscore['bd_region_type'])
        
    df_bd_repeat_for_draw['zscore'] = zscore_l
    df_bd_repeat_for_draw['type'] = type_l
    df_bd_repeat_for_draw['bd_type'] = bd_type_l
    
    type_order = ['sharp_weak', 'sharp_strong', 'wide']
    color_use = []
    for type_ in type_order:
        color_use.append(region_color[type_])
            
    #plt.figure(figsize= (12, 5))
    #ax = sns.barplot(x = 'type', y = 'zscore', data = df_bd_repeat_for_draw, hue = 'bd_type', hue_order = type_order, palette = color_use, errwidth = 0,  ci = 'sd', saturation=1)
    #plt.hlines(0, -0.5, 13.5, linestyle = '-')
    #plt.xticks(rotation = -30)
               
    plt.figure(figsize= (12, 5))
    ax = sns.boxplot(x = 'type', y = 'zscore', data = df_bd_repeat_for_draw, hue = 'bd_type', hue_order = type_order, palette = color_use, saturation=1, fliersize = 0)
    plt.ylim([-3, 4])
    plt.hlines(0, -0.5, 13.5, linestyle = '--')
    plt.xticks(rotation = -30)
       
    #plt.figure(figsize= (9, 5))
    #ax = sns.boxplot(x = 'type', y = 'zscore', data = df_bd_repeat_for_draw, hue_order = type_order, palette = color_use, saturation=1, fliersize = 0)
    #plt.ylim([-3, 3])
    #plt.hlines(0, -0.5, 13.5, linestyle = '--')
    #plt.xticks(rotation = -30)
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True)  
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)
    

def draw_repeat_enrich_in_bd_region_family(col_all, df_cell_bd_repeat_zscore, region_color, save_name = ''):
    df_bd_repeat_for_draw = pd.DataFrame(columns = ['zscore', 'type', 'bd_type'])
    zscore_l = []
    type_l = []
    bd_type_l = []
    for col in col_all:
        zscore_l += list(df_cell_bd_repeat_zscore[col])
        type_l += [col for i in range(len(df_cell_bd_repeat_zscore))]
        bd_type_l += list(df_cell_bd_repeat_zscore['bd_region_type'])
        
    df_bd_repeat_for_draw['zscore'] = zscore_l
    df_bd_repeat_for_draw['type'] = type_l
    df_bd_repeat_for_draw['bd_type'] = bd_type_l
    
    type_order = ['sharp_weak', 'sharp_strong', 'wide']
    color_use = []
    for type_ in type_order:
        color_use.append(region_color[type_])
    #plt.figure(figsize= (12, 5))
    #ax = sns.barplot(x = 'type', y = 'zscore', data = df_bd_repeat_for_draw, hue = 'bd_type', hue_order = type_order, palette = color_use, errwidth = 0,  ci = 'sd', saturation=1)
    #plt.hlines(0, -0.5, 13.5, linestyle = '-')
    #plt.xticks(rotation = -30)               
    plt.figure(figsize= (16, 5))
    ax = sns.boxplot(x = 'type', y = 'zscore', data = df_bd_repeat_for_draw, hue = 'bd_type', hue_order = type_order, palette = color_use, saturation=1, fliersize = 0)
    plt.ylim([-3, 4.5])
    plt.hlines(0, -0.5, 41.5, linestyle = '--')
    plt.xticks(rotation = -50)
       
    #plt.figure(figsize= (9, 5))
    #ax = sns.boxplot(x = 'type', y = 'zscore', data = df_bd_repeat_for_draw, hue_order = type_order, palette = color_use, saturation=1, fliersize = 0)
    #plt.ylim([-3, 3])
    #plt.hlines(0, -0.5, 13.5, linestyle = '--')
    #plt.xticks(rotation = -30)
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True)  
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)
    
    

 
#hg19_repeat_file = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\data_use\hg19_repeat_element' + '/' + 'rmsk.txt'

#df_hg19_repeat_file = pd.read_csv(hg19_repeat_file, sep = '\t', header = None)

#df_hg19_repeat_file.columns = [0, 1, 2, 3, 4, 'chr', 'start', 'end', 5, 'chain', 'name', 'class', 'family', 6, 7, 8, 9]

#df_hg19_repeat_file_part = df_hg19_repeat_file[['chr', 'start', 'end', 'chain', 'name', 'class', 'family']]
#df_hg19_repeat_file_part.to_csv(r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\data_use\hg19_repeat_element' + '/' + 'Hg19_repeat_element.bed', sep = '\t', header = True, index = None)

df_hg19_repeat_file = pd.read_csv(r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\data_use\hg19_repeat_element' + '/' + 'Hg19_repeat_element.bed', sep = '\t', header = 0)


repeat_class = []
for repeat_c in list(np.unique(df_hg19_repeat_file['class'])):
    if '?' not in repeat_c:
        repeat_class.append(repeat_c)
    
repeat_type = {}
for repeat_c in repeat_class:
    print('This is ' + repeat_c)
    repeat_family = []
    df_hg19_repeat_file_part = df_hg19_repeat_file[df_hg19_repeat_file['class'] == repeat_c] 
    for repeat_f in list(np.unique(df_hg19_repeat_file_part['family'])):
        if '?' not in repeat_f:
            repeat_family.append(repeat_f)
            repeat_type[repeat_c] = repeat_family
    print(repeat_family)
            


repeat_add = r'E:\Users\dcdang\share\TAD_integrate\bd_region_overlap_gene\bd_region_with_repeat'

bd_overlap_repeat_result = read_save_data(repeat_add + '/' + 'All_cell_type_repeat_class_family_overlap.pkl')

random_result_record_cell = {}
for cell_type in ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'K562', 'NHEK', 'KBM7']:
    enzyme = 'MboI'
    file = repeat_add + '/' + 'random_add' + '/' + cell_type + '_' + enzyme + '_random_500_record.pkl'
    record_file = read_save_data(file)
    random_result_record_cell[cell_type + '_' + enzyme] = record_file[cell_type + '_' + enzyme]
    


save_add_repeat = r'E:\Users\dcdang\share\TAD_integrate\bd_region_overlap_gene\bd_region_with_repeat\mid_result'

repeat_zscore_record = {}
for cell_type in ['GM12878', 'HMEC', 'HUVEC', 'IMR90', 'K562', 'NHEK', 'KBM7']:    
    print(cell_type)
    enzyme = 'MboI'
    repeat_zscore_record[cell_type + '_' + enzyme] = {}
    df_bd_region_type = copy.deepcopy(bd_region_type_record[cell_type + '_' + enzyme])

    type_ = 'class'
    print('Dealing with class....')
    df_cell_bd_repeat_num = bd_overlap_repeat_result[cell_type + '_' + enzyme][type_]
    col_all = list(df_cell_bd_repeat_num.columns)
    dict_random_bd_repeat = random_result_record_cell[cell_type + '_' + enzyme]
    df_cell_bd_repeat_zscore = get_bd_repeat_zscore(df_bd_region_type, df_cell_bd_repeat_num, dict_random_bd_repeat, type_, col_all)    
    save_name1 = save_add_repeat + '/' + cell_type + '_' + enzyme + '_bd_repeat_class_zscore.svg' 
    draw_repeat_enrich_in_bd_region_class(col_all, df_cell_bd_repeat_zscore, region_color, save_name1)
    repeat_zscore_record[cell_type + '_' + enzyme][type_] = df_cell_bd_repeat_zscore
    
    
    type_ = 'family'
    print('Dealing with family....')
    df_cell_bd_repeat_num = bd_overlap_repeat_result[cell_type + '_' + enzyme][type_]
    col_all = list(df_cell_bd_repeat_num.columns)
    dict_random_bd_repeat = random_result_record_cell[cell_type + '_' + enzyme]
    df_cell_bd_repeat_zscore = get_bd_repeat_zscore(df_bd_region_type, df_cell_bd_repeat_num, dict_random_bd_repeat, type_, col_all)    
    save_name2 = save_add_repeat + '/' + cell_type + '_' + enzyme + '_bd_repeat_family_zscore.svg' 
    draw_repeat_enrich_in_bd_region_family(col_all, df_cell_bd_repeat_zscore, region_color, save_name2)
    repeat_zscore_record[cell_type + '_' + enzyme][type_] = df_cell_bd_repeat_zscore
    
    


### re-draw these result


def get_all_cell_type_repeat_zscore_draw(repeat_zscore_record, cell_type_list, repeat_type, repeat_subtype, region_color, range_y = [], save_name = ''):
    enzyme = 'MboI'
    
    df_repeat_zscore_draw = pd.DataFrame(columns = ['zscore', 'bd_type', 'cell_type'])
    zscore_l = []
    bd_type_l = []
    cell_l = []
    
    for cell_type in cell_type_list:
        df_cell_result = copy.deepcopy(repeat_zscore_record[cell_type + '_' + enzyme][repeat_type])
        df_cell_result = df_cell_result.fillna(0)
        zscore_l += list(df_cell_result[repeat_subtype])
        bd_type_l += list(df_cell_result['bd_region_type'])
        cell_l += [cell_type for i in range(len(df_cell_result))]
    df_repeat_zscore_draw['zscore'] = zscore_l
    df_repeat_zscore_draw['bd_type'] = bd_type_l
    df_repeat_zscore_draw['cell_type'] = cell_l
    
    for cell_type in cell_type_list:
        print('This is ' + cell_type)
        df_repeat_zscore_draw_part = df_repeat_zscore_draw[df_repeat_zscore_draw['cell_type'] == cell_type]
        SS_value = np.array(df_repeat_zscore_draw_part[df_repeat_zscore_draw_part['bd_type'] == 'sharp_strong']['zscore'])
        SW_value = np.array(df_repeat_zscore_draw_part[df_repeat_zscore_draw_part['bd_type'] == 'sharp_weak']['zscore'])
        WI_value = np.array(df_repeat_zscore_draw_part[df_repeat_zscore_draw_part['bd_type'] == 'wide']['zscore'])        
        pvalue_ss_sw = scipy.stats.ttest_ind(SS_value, SW_value)[1] 
        pvalue_ss_wi = scipy.stats.ttest_ind(SS_value, WI_value)[1] 
        pvalue_sw_wi = scipy.stats.ttest_ind(SW_value, WI_value)[1]              
        print('Sharp strong and sharp weak: ' + str(pvalue_ss_sw))
        print('Sharp strong and wide: ' + str(pvalue_ss_wi))
        print('Sharp weak and wide: ' + str(pvalue_sw_wi))
    
    type_order = ['sharp_weak', 'sharp_strong', 'wide']
    color_use = []
    for type_ in type_order:
        color_use.append(region_color[type_])
   
    plt.figure(figsize= (8, 5))
    ax = sns.boxplot(x = 'cell_type', y = 'zscore', data = df_repeat_zscore_draw, hue = 'bd_type', hue_order = type_order, palette = color_use, saturation=1, fliersize = 0)
    #ax = sns.violinplot(x = 'cell_type', y = 'zscore', data = df_repeat_zscore_draw, hue = 'bd_type', hue_order = type_order, palette = color_use, saturation=1, cut=0)
    if range_y != []:
        plt.ylim(range_y)
    plt.hlines(0, -0.45, 6.45, linestyle = '--')
    plt.xticks(rotation = 0)
    plt.xlabel('cell type', FontSize = 0)
    plt.ylabel(repeat_subtype + ' enrich zscore', FontSize = 12)    
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    ax.legend_ = None
    #plt.figure(figsize= (9, 5))
    #ax = sns.boxplot(x = 'type', y = 'zscore', data = df_bd_repeat_for_draw, hue_order = type_order, palette = color_use, saturation=1, fliersize = 0)
    #plt.ylim([-3, 3])
    #plt.hlines(0, -0.5, 13.5, linestyle = '--')
    #plt.xticks(rotation = -30)
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True)  
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)


def get_certain_cell_type_all_repeat_zscore_draw(repeat_zscore_record, cell_type, repeat_type, target_repeat, region_color, range_y = [], save_name = ''):
    enzyme = 'MboI'   
    df_repeat_zscore_draw = pd.DataFrame(columns = ['zscore', 'bd_type', 'repeat_type'])
    zscore_l = []
    bd_type_l = []
    repeat_l = []
    
    df_cell_result = copy.deepcopy(repeat_zscore_record[cell_type + '_' + enzyme][repeat_type])
    df_cell_result = df_cell_result.fillna(0)
    for repeat_subtype in target_repeat:
        zscore_l += list(df_cell_result[repeat_subtype])
        bd_type_l += list(df_cell_result['bd_region_type'])
        repeat_l += [repeat_subtype for i in range(len(df_cell_result['bd_region_type']))]
        
    df_repeat_zscore_draw['zscore'] = zscore_l
    df_repeat_zscore_draw['bd_type'] = bd_type_l
    df_repeat_zscore_draw['repeat_type'] = repeat_l
    
    type_order = ['sharp_weak', 'sharp_strong', 'wide']
    color_use = []
    for type_ in type_order:
        color_use.append(region_color[type_])
   
    plt.figure(figsize= (12, 5))
    ax = sns.boxplot(x = 'repeat_type', y = 'zscore', data = df_repeat_zscore_draw, hue = 'bd_type', hue_order = type_order, palette = color_use, saturation=1, fliersize = 0)
    #ax = sns.violinplot(x = 'cell_type', y = 'zscore', data = df_repeat_zscore_draw, hue = 'bd_type', hue_order = type_order, palette = color_use, saturation=1, cut=0)
    if range_y != []:
        plt.ylim(range_y)
    plt.hlines(0, -0.45, 27.45, linestyle = '--')
    plt.xticks(rotation = 30)
    plt.xlabel('cell type', FontSize = 0)
    plt.ylabel('Repeat element enrich zscore', FontSize = 12)    
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    ax.legend_ = None
    #plt.figure(figsize= (9, 5))
    #ax = sns.boxplot(x = 'type', y = 'zscore', data = df_bd_repeat_for_draw, hue_order = type_order, palette = color_use, saturation=1, fliersize = 0)
    #plt.ylim([-3, 3])
    #plt.hlines(0, -0.5, 13.5, linestyle = '--')
    #plt.xticks(rotation = -30)
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True)  
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)




repeat_type = 'class'
repeat_subtype = 'SINE'
range_y = [-2, 4]
print('For ' + repeat_subtype)
save_name1 = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\bd_region_repeat\cell_enriched_repeat' + '/' + 'All_cell_bd_region_' + repeat_subtype + '_enrich_zscore.svg'
get_all_cell_type_repeat_zscore_draw(repeat_zscore_record, cell_type_list, repeat_type, repeat_subtype, region_color, range_y, save_name = save_name1)
    

repeat_type = 'family'
repeat_subtype = 'Alu'
range_y = [-1.5, 4.5]
print('For ' + repeat_subtype)
save_name1 = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\bd_region_repeat\cell_enriched_repeat' + '/' + 'All_cell_bd_region_' + repeat_subtype + '_enrich_zscore.svg'
get_all_cell_type_repeat_zscore_draw(repeat_zscore_record, cell_type_list, repeat_type, repeat_subtype, region_color, range_y, save_name = save_name1)
    

repeat_type = 'family'
repeat_subtype = 'TcMar-Tigger'
range_y = [-2, 3.5]
print('For ' + repeat_subtype)
save_name1 = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\bd_region_repeat\cell_enriched_repeat' + '/' + 'All_cell_bd_region_' + repeat_subtype + '_enrich_zscore.svg'
get_all_cell_type_repeat_zscore_draw(repeat_zscore_record, cell_type_list, repeat_type, repeat_subtype, region_color, range_y, save_name = save_name1)
    


repeat_dict = {'SINE':['Alu', 'MIR'], 
               'LINE':['L1', 'L2', 'CR1', 'RTE'],
               'LTR':['ERV', 'ERV1', 'ERVK', 'ERVL', 'ERVL-MaLR', 'Gypsy'],
               'DNA':[  'Merlin', 'MuDR', 'TcMar-Mariner', 'TcMar-Tc1',
                      'TcMar-Tc2', 'TcMar-Tigger', 'hAT-Blackjack', 'hAT-Charlie',
                      'hAT-Tag1', 'hAT-Tip100'],
               'Low_complexity':['Low_complexity'],
               'Satellite':['Satellite'],
               'RNA':['RNA', 'rRNA', 'scRNA', 'snRNA', 'srpRNA', 'tRNA']}

target_repeat = ['Alu', 'MIR', 'L1', 'L2', 'CR1', 'RTE',
                 'ERV', 'ERV1', 'ERVK', 'ERVL', 'ERVL-MaLR', 'Gypsy',
                   'Merlin', 'MuDR', 'TcMar-Mariner',
                   'TcMar-Tc2', 'TcMar-Tigger', 'hAT-Blackjack',
                   'hAT-Charlie', 'hAT-Tip100',
                   'Low_complexity', 'Satellite', 
                   'RNA', 'rRNA', 'scRNA', 'snRNA', 'srpRNA', 'tRNA']

range_y_use = {'GM12878':[-3, 3.5], 'K562':[-2.5, 4.5], 'HMEC':[-2.5, 4],
               'NHEK':[-2.5, 4], 'HUVEC':[-2.5, 4], 'IMR90':[-3, 4.5], 
               'KBM7':[-2.5, 3.5]}

for cell_type in cell_type_list:
    print('This is ' + cell_type)    
    range_y = range_y_use[cell_type]
    save_name = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\bd_region_repeat\all_repeat_zscore_for_every_cell' + '/' + cell_type + '_bd_region_enrich_all_repeat_zscore.svg'
    get_certain_cell_type_all_repeat_zscore_draw(repeat_zscore_record, cell_type, repeat_type, target_repeat, region_color, range_y, save_name = save_name)
    
 
    
## focus on Alu
from collections import Counter
 
def get_alu_cpm_compare_barplot(alu_type_number, df_bd_overlap_alu, df_bd_region_type, dict_random_bd_alu, region_color, save_name = '', type_order = ['sharp_weak', 'sharp_strong', 'wide']):
    # boundary region
    df_bd_overlap_alu_new = copy.deepcopy(df_bd_overlap_alu)
    for alu in ['AluS', 'AluJ', 'AluY']:
        df_bd_overlap_alu_new[alu] = df_bd_overlap_alu[alu] / alu_type_number[alu] * 10**6 / np.array(df_bd_region_type['length'])
    # random
    alus_random = []
    aluj_random = []
    aluy_random = []
    for i in dict_random_bd_alu.keys():
        for alu in ['AluS', 'AluJ', 'AluY']:
            target = dict_random_bd_alu[i][alu] / alu_type_number[alu] * 10**6 / np.array(df_bd_region_type['length'])
            if alu == 'AluS':
                alus_random.append(list(target))
            if alu == 'AluJ':
                aluj_random.append(list(target))
            if alu == 'AluY':
                aluy_random.append(list(target))
    random_bd_alu_mean = {}
    random_bd_alu_mean['AluS'] = np.mean(np.array(alus_random), axis = 0)
    random_bd_alu_mean['AluJ'] = np.mean(np.array(aluj_random), axis = 0)
    random_bd_alu_mean['AluY'] = np.mean(np.array(aluy_random), axis = 0)

    df_alu_draw = pd.DataFrame(columns = ['number', 'Alu_type', 'bd_type'])
    number_l = []
    alu_type_l = []
    bd_type_l = []
    for alu in ['AluS', 'AluJ', 'AluY']:
        number_l += list(df_bd_overlap_alu_new[alu])
        alu_type_l += [alu for i in range(len(df_bd_overlap_alu_new))]
        bd_type_l += list(df_bd_region_type['region_type_adjust'])
        
        number_l += list(random_bd_alu_mean[alu])
        alu_type_l += [alu for i in range(len(random_bd_alu_mean[alu]))]
        bd_type_l += ['random' for i in range(len(random_bd_alu_mean[alu]))]
    df_alu_draw['number'] = number_l
    df_alu_draw['Alu_type'] = alu_type_l
    df_alu_draw['bd_type'] = bd_type_l
    df_alu_draw = df_alu_draw.fillna(0)
    for alu in ['AluS', 'AluJ', 'AluY']:
        print('For ' + alu)
        df_alu_draw_part = df_alu_draw[df_alu_draw['Alu_type'] == alu]
        SS_value = np.array(df_alu_draw_part[df_alu_draw_part['bd_type'] == 'sharp_strong']['number'])
        SW_value = np.array(df_alu_draw_part[df_alu_draw_part['bd_type'] == 'sharp_weak']['number'])
        WI_value = np.array(df_alu_draw_part[df_alu_draw_part['bd_type'] == 'wide']['number'])
        random_value = np.array(df_alu_draw_part[df_alu_draw_part['bd_type'] == 'random']['number'])    
        pvalue_ss_r = scipy.stats.ttest_ind(SS_value, random_value)[1] 
        pvalue_sw_r = scipy.stats.ttest_ind(SW_value, random_value)[1] 
        pvalue_wi_r = scipy.stats.ttest_ind(WI_value, random_value)[1]              
        #plt.figure(figsize= (6, 5))
        #sns.distplot(SS_value, color = '#D65F4D')
        #sns.distplot(SW_value, color = '#459457')
        #sns.distplot(WI_value, color = '#4392C3')
        #sns.distplot(random_value, color = 'grey')        
        #print(np.mean(SS_value))
        #print(np.mean(SW_value))
        #print(np.mean(WI_value))
        #print(np.mean(random_value))
        print('Sharp_strong and random :' + str(pvalue_ss_r))
        print('Sharp_weak and random :' + str(pvalue_sw_r))
        print('Wide and random :' + str(pvalue_wi_r))

    color_use = ['#B0B0B0']
    for type_ in type_order:
        color_use.append(region_color[type_])
    #print(np.max(df_alu_draw['number']))
    plt.figure(figsize= (6, 5))
    ax = sns.barplot(x = 'Alu_type', y = 'number', hue = 'bd_type', data = df_alu_draw, hue_order = ['random', 'sharp_weak', 'sharp_strong', 'wide'], palette = color_use, saturation = 1,             
            errcolor = 'black', errwidth = 1.6, capsize = 0.08, ci = 95, edgecolor='black', alpha = 1)
    #plt.legend(fontsize = 12, fancybox = False,shadow = False, frameon=False, title = '', loc = 'upper left')
    plt.yticks(FontSize = 12)
    plt.xticks(FontSize = 10, rotation = 0)
    plt.xlabel('type', FontSize = 0)
    plt.ylabel('Counts per million / 50kb', FontSize = 12)    
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    ax.legend_ = None
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True)  
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

   
df_hg19_repeat_file_chr_alu = df_hg19_repeat_file[(df_hg19_repeat_file['chr'] == 'chr2') & (df_hg19_repeat_file['family'] == 'Alu')]   
df_hg19_repeat_file_chr_alu = df_hg19_repeat_file_chr_alu.reset_index(drop = True)


# count alu number of 3 types    
alu_num_info = Counter(list(df_hg19_repeat_file_chr_alu['name']))
num_s = 0
num_j = 0
num_y = 0
for class_alu in list(alu_num_info.keys()):
    if 'AluS' in class_alu:
        num_s += alu_num_info[class_alu]
    elif 'AluJ' in class_alu:
        num_j += alu_num_info[class_alu]
    elif 'AluY' in class_alu:
        num_y += alu_num_info[class_alu]

alu_type_number = {}
alu_type_number['AluS'] = num_s
alu_type_number['AluJ'] = num_j
alu_type_number['AluY'] = num_y


# load bd region overlap alu numbers from unbuntu 
bd_overlap_alu_result = read_save_data(repeat_add + '/' + 'All_cell_type_alu_overlap_number.pkl')

bd_overlap_alu_result_random = read_save_data(repeat_add + '/' + 'random_cell_type_alu_overlap_number.pkl')



for cell_type in cell_type_list:
    print('This is ' + cell_type)
    enzyme = 'MboI'
    df_bd_overlap_alu = copy.deepcopy(bd_overlap_alu_result[cell_type])
    df_bd_region_type = copy.deepcopy(bd_region_type_record[cell_type + '_' + enzyme])
    dict_random_bd_alu = copy.deepcopy(bd_overlap_alu_result_random[cell_type + '_' + enzyme])
    
    save_name = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\bd_region_repeat\alu_repeat' + '/' + cell_type + '_' + enzyme + '_alu_CPM_in_bd_region.svg'
    get_alu_cpm_compare_barplot(alu_type_number, df_bd_overlap_alu, df_bd_region_type, dict_random_bd_alu, region_color, save_name = save_name)
    print('\n')



    
######## bd region conserved     


region_save_add = r'E:\Users\dcdang\share\TAD_integrate\bd_region_overlap_gene\bd_region_conserved'

for cell in cell_type_list:
    enzyme = 'MboI'
    df_bd_region_cell = copy.deepcopy(bd_region_type_record[cell + '_' + enzyme])
    df_bd_region_cell['index'] = list(range(len(df_bd_region_cell)))
    df_bd_region_cell.to_csv(region_save_add + '/' + cell + '_' + enzyme + '_bd_region_with_type.bed', sep = '\t', header = None, index = None)
    
    
bd_region_overlap_all_cell = read_save_data(region_save_add + '/' + 'bd_region_overlap_all_cell.pkl')



cell_num = 7
df_cell_bd_region_conserved = pd.DataFrame(columns = ['cons_num', 'cell_type', 'bd_type', 'bd_type_switch'])
cons_num_l = []
cell_l = []
bd_type_l = []
bd_type_switch_l = []

for cell_type in cell_type_list:
    print('This is ' + cell_type)
    df_bd_region_overlap = bd_region_overlap_all_cell[cell_type + '_' + enzyme]
    
    target_col1 = []
    target_col2 = []
    for cell_t in cell_type_list:
        if cell_t == cell_type:
            continue
        target_col1.append(cell_t + '_' + 'overlap')
        target_col2.append(cell_t + '_' + 'judge')
    cons_num = np.sum(df_bd_region_overlap[target_col1], axis = 1)
    cons_num_l += list((np.array(cons_num) + 1) / cell_num)
    cell_l += [cell_type for i in range(len(df_bd_region_overlap))]
    bd_type_l += list(df_bd_region_overlap['region_type_adjust'])
    for i in range(len(df_bd_region_overlap)):
        stable_num = 0 
        for col in target_col2:
            judge = df_bd_region_overlap[col][i]
            if judge == 'stable':
                stable_num += 1
        bd_type_switch_l.append((stable_num+1) / cell_num)
        #bd_type_switch_l.append((stable_num + 1) / (cons_num[i] + 1))
        
df_cell_bd_region_conserved['cons_num'] = cons_num_l
df_cell_bd_region_conserved['cell_type'] = cell_l
df_cell_bd_region_conserved['bd_type'] = bd_type_l
df_cell_bd_region_conserved['bd_type_switch'] = bd_type_switch_l





def draw_bd_region_cell_conserved(df_cell_bd_region_conserved, ylabel, type_order, region_color, save_name = ''):   
    for cell_type in cell_type_list:
        print('For ' + cell_type)
        df_cell_bd_region_conserved_part = df_cell_bd_region_conserved[df_cell_bd_region_conserved['cell_type'] == cell_type]        
        SS_value = np.array(df_cell_bd_region_conserved_part[df_cell_bd_region_conserved_part['bd_type'] == 'sharp_strong'][ylabel])
        SW_value = np.array(df_cell_bd_region_conserved_part[df_cell_bd_region_conserved_part['bd_type'] == 'sharp_weak'][ylabel])
        WI_value = np.array(df_cell_bd_region_conserved_part[df_cell_bd_region_conserved_part['bd_type'] == 'wide'][ylabel])    
        pvalue_ss_sw = scipy.stats.mannwhitneyu(SS_value, SW_value)[1] 
        pvalue_ss_wi = scipy.stats.mannwhitneyu(SS_value, WI_value)[1] 
        pvalue_wi_sw = scipy.stats.mannwhitneyu(WI_value, SW_value)[1] 
        print('Sharp_strong and sharp_weak :' + str(pvalue_ss_sw))
        print('Sharp_strong and wide :' + str(pvalue_ss_wi))
        print('Sharp_weak and wide :' + str(pvalue_wi_sw))
    up_bound = np.max(df_cell_bd_region_conserved[ylabel])
    color_use = []
    for type_ in type_order:
        color_use.append(region_color[type_])
    plt.figure(figsize= (6, 5))
    ax = sns.barplot(x = 'cell_type', y = ylabel, data = df_cell_bd_region_conserved, hue = 'bd_type', hue_order = type_order, palette = color_use, saturation = 1,             
            errcolor = 'black', errwidth = 1.6, capsize = 0.1, ci = 95, edgecolor='black', alpha = 1)
    #plt.legend(fontsize = 12, fancybox = False,shadow = False, frameon=False, title = '', loc = 'upper left')
    plt.yticks(FontSize = 12)
    plt.xticks(FontSize = 10, rotation = 0)
    if ylabel == 'cons_num':
        plt.ylabel('Cell line conservation',  FontSize = 12)
        plt.ylim([0,up_bound + 0.1])
    elif ylabel == 'bd_type_switch':
        plt.ylabel('region type conservation',  FontSize = 12)
        plt.ylim([0,0.85])
    plt.xlabel('type', FontSize = 0)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    ax.legend_ = None
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True)  
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)


result_save_add = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\region_conserved'

#type_order = ['sharp_weak', 'sharp_strong', 'wide']
type_order = ['sharp_weak', 'sharp_strong', 'wide']
ylabel = 'cons_num'
save_name1 = result_save_add + '/' + 'bd_region_cell_type_conservation_all.svg' 
draw_bd_region_cell_conserved(df_cell_bd_region_conserved, ylabel, type_order, region_color, save_name = save_name1)


type_order = ['sharp_weak', 'wide', 'sharp_strong']
ylabel = 'bd_type_switch'
save_name2 = result_save_add + '/' + 'bd_region_bd_type_conservation_all.svg' 
draw_bd_region_cell_conserved(df_cell_bd_region_conserved, ylabel, type_order, region_color, save_name = save_name2)

###### bd region DNase peak motif analysis
def draw_bar_plot_motif_enrich(df_cell_DNase_peak_motif_top, b_color, save_name = ''):    
    plt.figure(figsize= (6, 5))
    plt.bar(list(range(len(df_cell_DNase_peak_motif_top))), list(df_cell_DNase_peak_motif_top['-log10(P-value)']), color = b_color)

    plt.xticks(list(range(len(df_cell_DNase_peak_motif_top))), list(df_cell_DNase_peak_motif_top['Motif_Name']), rotation= 90, FontSize = 10)
    #plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'], FontSize = 12)
    #plt.ylim([0,1])
    plt.ylabel('-log10(P-value)',  FontSize = 12)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.tick_params(axis = 'y', length=7, width = 1.6)
    ax.tick_params(axis = 'x', length=3, width = 1.6)
    plt.gcf().subplots_adjust(bottom = 0.3)
    if save_name != '':
        plt.savefig(save_name, format = 'svg', transparent = True)  
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)


    
motif_file = r'E:\Users\dcdang\share\TAD_integrate\DNase_peak\DNase_peak_TF' + '/' + 'multi_cell_bd_DNase_motif_enrich.pkl'

cell_bd_DNase_motif_enrich = read_save_data(motif_file)

bd_tf_add = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\bd_region_TF'
top_cut = 25
region_type = ['sharp_strong', 'sharp_weak', 'wide']
motif_result_record = {}
for cell_type in ['IMR90', 'GM12878', 'HMEC', 'HUVEC', 'K562', 'NHEK']:
    color_l = []
    r_type_l = []
    motif_l = []
    motif_result_record[cell_type] = {}
    if not os.path.exists(bd_tf_add + '/' + cell_type):
        os.makedirs(bd_tf_add + '/' + cell_type)
    cell_motif_enrich = copy.deepcopy(cell_bd_DNase_motif_enrich[cell_type])
    for r_type in region_type:
        df_cell_DNase_peak_motif = cell_motif_enrich[r_type]
        df_cell_DNase_peak_motif_top = df_cell_DNase_peak_motif.iloc[:top_cut]
        b_color = region_color[r_type]
        save_name_motif = bd_tf_add + '/' + cell_type + '/' + cell_type + '_' + r_type + '_' + 'bd_region_enrich_TF_top' + str(top_cut) + '.svg'
        draw_bar_plot_motif_enrich(df_cell_DNase_peak_motif_top, b_color, save_name_motif)
        motif = list(df_cell_DNase_peak_motif_top['Motif_Name'])
        motif_l.append(set(motif))
        color_l.append(b_color)
        r_type_l.append(r_type)
    
    plt.figure(figsize= (6, 8))
    vd3 = venn3(subsets = [motif_l[0], motif_l[1], motif_l[2]], 
                     set_labels=(r_type_l[0], r_type_l[1], r_type_l[2]), 
                     set_colors=(color_l[0], color_l[1], color_l[2]), 
                     alpha=1, normalize_to=1.0, ax=None, 
                     subset_label_formatter=None,
                     )
    venn3_circles([motif_l[0], motif_l[1], motif_l[2]], linestyle='-', linewidth=1.5, color='black')
    for text in vd3.set_labels:
        text.set_fontsize(16)
    for text in vd3.subset_labels:
        text.set_fontsize(16)
    #save_name_venn = ''
    save_name_venn = bd_tf_add + '/' + cell_type + '/' + cell_type + '3_bd_region_top_TF_overlap.svg'
    if save_name_venn != '':
        plt.savefig(save_name_venn, format = 'svg', transparent = True)  
       
    A = motif_l[0]
    B = motif_l[1]
    C = motif_l[2]     
    motif_result_record[cell_type]['SS-SW-WI'] = A-B-C 
    motif_result_record[cell_type]['SW-SS-WI'] = B-A-C  
    motif_result_record[cell_type]['WI-SS-SW'] = C-A-B 
    motif_result_record[cell_type]['SS&SW-WI'] = A&B-C
    motif_result_record[cell_type]['SS&WI-SW'] = A&C-B
    motif_result_record[cell_type]['SW&WI-SS'] = B&C-A 
    motif_result_record[cell_type]['SS&SW&WI'] = A&B&C
        
    motif_unio = motif_l[0] | motif_l[1] | motif_l[2]
    motif_unio = set(list(cell_motif_enrich[r_type]['Motif_Name']))
    
def get_motif_enrich_ratio_diff(cell_motif_enrich, motif_unio, r_type1, r_type2):
       
    df_DNase_peak_motif_cell1 = cell_motif_enrich[r_type1]
    df_DNase_peak_motif_cell2 = cell_motif_enrich[r_type2]
       
    df_DNase_peak_motif_cell_part1 = df_DNase_peak_motif_cell1[df_DNase_peak_motif_cell1['Motif_Name'].isin(motif_unio)]
    df_DNase_peak_motif_cell_part2 = df_DNase_peak_motif_cell2[df_DNase_peak_motif_cell2['Motif_Name'].isin(motif_unio)]

    motif_order = CategoricalDtype(list(motif_unio), ordered=True)

    df_DNase_peak_motif_cell_part1['Motif_Name'] = df_DNase_peak_motif_cell_part1['Motif_Name'].astype(motif_order)
    df_DNase_peak_motif_cell_part1 = df_DNase_peak_motif_cell_part1.sort_values('Motif_Name')
    df_DNase_peak_motif_cell_part1 = df_DNase_peak_motif_cell_part1.reset_index(drop = True)
    
    df_DNase_peak_motif_cell_part2['Motif_Name'] = df_DNase_peak_motif_cell_part2['Motif_Name'].astype(motif_order)
    df_DNase_peak_motif_cell_part2 = df_DNase_peak_motif_cell_part2.sort_values('Motif_Name')
    df_DNase_peak_motif_cell_part2 = df_DNase_peak_motif_cell_part2.reset_index(drop = True)
      
    ratio1 = df_DNase_peak_motif_cell_part1['% of Target Sequences with Motif']
    ratio2 = df_DNase_peak_motif_cell_part2['% of Target Sequences with Motif']
    
    ratio1_ = [np.float(x.split('%')[0]) for x in ratio1]
    ratio2_ = [np.float(x.split('%')[0]) for x in ratio2]
    
    enrich1 = df_DNase_peak_motif_cell_part1['-log10(P-value)']
    enrich2 = df_DNase_peak_motif_cell_part2['-log10(P-value)']   
    
    plt.figure(figsize= (6, 6))
    plt.scatter((np.array(ratio1_) - np.array(ratio2_)), (np.array(enrich1) - np.array(enrich2)))    


from pandas.api.types import CategoricalDtype

r_type1 = 'sharp_strong'
r_type2 = 'sharp_weak'

get_motif_enrich_ratio_diff(cell_motif_enrich, motif_unio, r_type1, r_type2)

###################  single cell image
def get_chr_bin(hg38_chr2_bin_list):
    df_bin_chr2_hg38 = pd.DataFrame(columns = ['chr', 'start', 'end', 'index'])
    chr_l = []
    st_l = []
    ed_l = []
    index_l = []
    for i in range(len(hg38_chr2_bin_list)):
        bin_ = hg38_chr2_bin_list[i]
        chr_l.append(bin_.split(':')[0])
        st_l.append(int(bin_.split(':')[-1].split('-')[0]))
        ed_l.append(int(bin_.split(':')[-1].split('-')[-1]))
        index_l.append(i)
    
    df_bin_chr2_hg38 = pd.DataFrame(columns = ['chr', 'start', 'end', 'index'])
    df_bin_chr2_hg38['chr'] = chr_l
    df_bin_chr2_hg38['start'] = st_l
    df_bin_chr2_hg38['end'] = ed_l
    df_bin_chr2_hg38['index'] = index_l
    return df_bin_chr2_hg38

def hg38_bin_adjust(df_bin_chr2_hg38_to_hg19, resolution):
    length_l = []
    st_l_hg19 = []
    ed_l_hg19 = []
    ind_l_hg19 = []
    
    for i in range(len(df_bin_chr2_hg38_to_hg19)):
        st = df_bin_chr2_hg38_to_hg19['start'][i]
        ed = df_bin_chr2_hg38_to_hg19['end'][i]
        length_l.append(ed - st)
        mid = (ed + st) / 2
        mid_index = int(mid / resolution)
        st_hg19 = mid_index * resolution
        ed_hg19 = mid_index * resolution + resolution
        st_l_hg19.append(st_hg19)
        ed_l_hg19.append(ed_hg19)
        ind_l_hg19.append(mid_index)
    df_bin_chr2_hg38_to_hg19['length'] = length_l
    df_bin_chr2_hg38_to_hg19['st_hg19'] = st_l_hg19
    df_bin_chr2_hg38_to_hg19['ed_hg19'] = ed_l_hg19
    df_bin_chr2_hg38_to_hg19['index_hg19'] = ind_l_hg19
    return df_bin_chr2_hg38_to_hg19

# get other information
cell_type = 'IMR90'
enzyme = 'MboI'
mat_dense = copy.deepcopy(hic_mat_all_cell_replicate[cell_type][enzyme]['iced'])
mat_blank_row = get_blank_row_in_matrix(mat_dense)
df_bd_insul_pvalue = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['pvalue'])
result_record = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['BD_region'])
df_boundary_region_combine = copy.deepcopy(result_record['Combine']['bd_region'])
bd_score_cell_combine = copy.deepcopy(result_record['Combine']['TAD_score'])
df_bd_region_type = copy.deepcopy(bd_region_type_record[cell_type + '_' + enzyme])
# df_bd_region_type save
df_bd_region_type_save = copy.deepcopy(df_bd_region_type)
df_bd_region_type_save['start'] = np.array(df_bd_region_type_save['start']*resolution).astype('int32')
df_bd_region_type_save['end'] = np.array(df_bd_region_type_save['end']*resolution).astype('int32') + resolution
df_bd_region_type_save.to_csv(r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\single_cell_image' + '/' + 'IMR90_bd_region_type.bed', sep = '\t', header = None, index = None)


##### 250kb hg38 to hg19

hg38_chr2_size = 242193529
res_250 = 250000
Chr = 'chr2'
hg38_chr2_bin_list = chr_cut(hg38_chr2_size, Chr, res_250, chr_ref)
df_bin_chr2_hg38 = get_chr_bin(hg38_chr2_bin_list)
df_bin_chr2_hg38.to_csv(r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\single_cell_image' + '/' + 'human_hg38_bin_list.txt', sep = '\t', header = None, index = None)
df_bin_chr2_hg38_250 = copy.deepcopy(df_bin_chr2_hg38)

# Liftover done, liftover regions overlap with bd region
df_bin_chr2_hg38_to_hg19 = pd.read_csv(r'E:\Users\dcdang\share\TAD_integrate\single_cell_image' + '/' + 'human_hg38_to_hg19_250_lift.bed', sep = '\t', header = None)
df_bin_chr2_hg38_to_hg19.columns = ['chr', 'start', 'end', 'index']
# adjust liftover bin
#df_bin_chr2_hg38_to_hg19 = hg38_bin_adjust(df_bin_chr2_hg38_to_hg19, resolution)

df_lift_bin_with_bd_region = pd.read_csv(r'E:\Users\dcdang\share\TAD_integrate\single_cell_image' + '/' + 'hg38_chr2_250k_bin_in_hg19_with_bd_region.bed', sep = '\t', header = None)

stats_l = []
bd_type_l = []
for i in range(len(df_bin_chr2_hg38_250)):
    index = df_bin_chr2_hg38_250['index'][i]
    if index not in list(df_bin_chr2_hg38_to_hg19['index']):
        bd_type_l.append('Unmapped')
        stats_l.append('None')
        continue
    if index not in list(df_lift_bin_with_bd_region[3]):
        bd_type_l.append('No_overlap')
        stats_l.append('None')
    else:
        df_lift_bin_with_bd_region_part = df_lift_bin_with_bd_region[df_lift_bin_with_bd_region[3] == index]
        bd_type_target = list(df_lift_bin_with_bd_region_part[16])
        overlap_target = list(df_lift_bin_with_bd_region_part[17])
        ind = np.argmax(overlap_target)
        bd_type = bd_type_target[ind]
        overlap = overlap_target[ind]
        stats_l.append((bd_type, overlap))
        if overlap < 50000:
            bd_type_l.append('No_overlap')
        else:
            bd_type_l.append(bd_type)

df_bin_chr2_hg38_250['stats'] = stats_l       
df_bin_chr2_hg38_250['bd_type'] = bd_type_l


#### the image data are built from hg38
info_bk_zhuang = r'E:\Users\scfan\data\Zhuang_2020_Cell_data\chromosome2' + '\data.pkl'
dic_info_zhuang = read_save_data(info_bk_zhuang)

IMR90_bd_single_chrom = r'E:\Users\scfan\data\Zhuang_2020_Cell_data\chromosome2\analysis\call_domain' + '\domain_2021_06_23.pkl'
dic_IMR90_domain_single_chrom = read_save_data(IMR90_bd_single_chrom)
IMR90_domain_l = dic_IMR90_domain_single_chrom['domain_starts']

# cell number for each bin
dom_all = np.array([dom for doms in IMR90_domain_l for dom in doms[1:-1]])
unk_,cts_=np.unique(dom_all,return_counts=True)
cts = np.zeros(935)
cts[unk_]=cts_

# probe 50kb
probe_name_l = dic_info_zhuang['region_names']

#######
probe_judge = []
probe_l = []
for i in range(len(df_bin_chr2_hg38_250)):
    Chr = df_bin_chr2_hg38_250['chr'][i] 
    st = df_bin_chr2_hg38_250['start'][i]
    p_name = Chr + ':' + str(st+1) + '-' + str(st+resolution+1)
    if p_name in probe_name_l:
        probe_judge.append('Yes')
        probe_l.append(p_name)
    else:
        probe_judge.append('No')
        probe_l.append('None')
        
df_bin_chr2_hg38_250['probe_judge'] =probe_judge         
df_bin_chr2_hg38_250['probe_l'] = probe_l       
                
df_bin_chr2_hg38_250_part = df_bin_chr2_hg38_250[df_bin_chr2_hg38_250['probe_judge'] == 'Yes']        
df_bin_chr2_hg38_250_part = df_bin_chr2_hg38_250_part.reset_index(drop = True)        

cell_num = 3029  
df_bin_chr2_hg38_250_part['cell_ratio'] =  cts / cell_num       

cell_ratio_l = []
bd_type_l = []
df_bd_cell_ratio = pd.DataFrame(columns = ['cell_ratio', 'bd_type'])
for i in range(len(df_bin_chr2_hg38_250_part)-2):
    bd_type = df_bin_chr2_hg38_250_part['bd_type'][i]
    if bd_type == 'Unmapped':
        continue
    bd_type_l.append(bd_type)
    cell_ratio_l.append(cts[i] / cell_num)

df_bd_cell_ratio['cell_ratio'] = cell_ratio_l
df_bd_cell_ratio['bd_type'] = bd_type_l


for bd_type in list(np.unique(df_bd_cell_ratio['bd_type'])):
    print('This is ' + bd_type)
    df_bd_cell_ratio_part = df_bd_cell_ratio[df_bd_cell_ratio['bd_type'] == bd_type]
    print(np.mean(df_bd_cell_ratio_part['cell_ratio']))
    
plt.figure(figsize= (5, 5))
#sns.boxplot(x = 'bd_type', y = 'cell_ratio', data = df_bd_cell_ratio, fliersize = 0, order = ['No_overlap', 'sharp_weak', 'sharp_strong', 'wide'])
sns.violinplot(x = 'bd_type', y = 'cell_ratio', data = df_bd_cell_ratio, cut = 2, order = ['No_overlap', 'sharp_weak', 'sharp_strong', 'wide'])
plt.ylim([-0.01, 0.30])


#### hg19 to hg38 50kb (above fail and use this final)

hg19_chr2_size = 243199373
res_50 = 50000
Chr = 'chr2'
hg19_chr2_bin_list = chr_cut(hg19_chr2_size, Chr, res_50, chr_ref)
df_bin_chr2_hg19 = get_chr_bin(hg19_chr2_bin_list)
df_bin_chr2_hg19.to_csv(r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\single_cell_image' + '/' + 'human_hg19_bin_list.txt', sep = '\t', header = None, index = None)
df_bin_chr2_hg38_250_new = copy.deepcopy(df_bin_chr2_hg38)


df_hg19_50_to_hg38_lift = pd.read_csv(r'E:\Users\dcdang\share\TAD_integrate\single_cell_image' + '/' + 'human_hg19_to_hg38_50_lift.bed', sep = '\t', header = None)
df_hg19_50_to_hg38_lift.columns = ['chr', 'start', 'end', 'index']
    

bd_type_list = ['No-bd' for i in range(len(df_bin_chr2_hg19))]

for i in range(len(df_bd_region_type)):
    region = df_bd_region_type['region'][i]
    bd_type = df_bd_region_type['region_type_adjust'][i]
    for ind in region:
        bd_type_list[ind] = bd_type
df_bin_chr2_hg19['bd_type'] = bd_type_list

type_lift = []
for i in range(len(df_hg19_50_to_hg38_lift)):
    ind = df_hg19_50_to_hg38_lift['index'][i]
    bd_type = df_bin_chr2_hg19['bd_type'][ind]
    type_lift.append(bd_type)    
df_hg19_50_to_hg38_lift['bd_type'] = type_lift    

df_hg19_50_to_hg38_lift.to_csv(r'E:\Users\dcdang\share\TAD_integrate\single_cell_image' + '/' + 'human_hg19_to_hg38_lift_with_bd.bed', sep = '\t', header =None, index = None)


df_hg38_250_bin_with_bd = pd.read_csv(r'E:\Users\dcdang\share\TAD_integrate\single_cell_image' + '/' + 'hg38_chr2_250k_bin_overlap_bd_region.bed', sep = '\t', header = None)

stats_l = []
bd_judge_l = []
for i in range(len(df_bin_chr2_hg38_250_new)):
    ind = df_bin_chr2_hg38_250_new['index'][i]
    if ind not in df_hg38_250_bin_with_bd[3]:
        stats_l.append('None')
        bd_judge_l.append('No_overlap')
    df_hg38_250_bin_with_bd_part = df_hg38_250_bin_with_bd[df_hg38_250_bin_with_bd[3] == ind]
    df_hg38_250_bin_with_bd_part = df_hg38_250_bin_with_bd_part.reset_index(drop = True)
    target = {}
    bd_length = []
    for type_ in ['No-bd', 'sharp_weak', 'sharp_strong', 'wide']:
        df_hg38_250_bin_with_bd_part_1 = df_hg38_250_bin_with_bd_part[df_hg38_250_bin_with_bd_part[8] == type_]
        if len(df_hg38_250_bin_with_bd_part_1) == 0:
            length_type = 0
        else:
            length_type = np.sum(df_hg38_250_bin_with_bd_part_1[9])
        target[type_] = length_type
        bd_length.append(length_type)
    stats_l.append(target)
    max_l = np.max(bd_length[1:])
    if max_l < 50000:
        bd_judge_l.append('No-bd')
    else:
        ind_max = np.argmax(bd_length[1:])
        bd_target = ['sharp_weak', 'sharp_strong', 'wide'][ind_max]
        bd_judge_l.append(bd_target)
        
df_bin_chr2_hg38_250_new['bd_stats'] = stats_l
df_bin_chr2_hg38_250_new['bd_type'] = bd_judge_l
        

#######     
df_bin_chr2_hg38_250_new['probe_judge'] =probe_judge         
df_bin_chr2_hg38_250_new['probe_l'] = probe_l       
                
df_bin_chr2_hg38_250_new_part = df_bin_chr2_hg38_250_new[df_bin_chr2_hg38_250['probe_judge'] == 'Yes']        
df_bin_chr2_hg38_250_new_part = df_bin_chr2_hg38_250_new_part.reset_index(drop = True)        

cell_num = 3029  
df_bin_chr2_hg38_250_new_part['cell_ratio'] =  cts / cell_num       

bd_type_adjust = list( df_bin_chr2_hg38_250_new_part['bd_type'])
for i in range(1, len(df_bin_chr2_hg38_250_new_part)-1):
    c_ratio = df_bin_chr2_hg38_250_new_part['cell_ratio'][i]
    bd_type_la = df_bin_chr2_hg38_250_new_part['bd_type'][i-1]
    bd_type = df_bin_chr2_hg38_250_new_part['bd_type'][i]
    bd_type_ne = df_bin_chr2_hg38_250_new_part['bd_type'][i+1]
    if bd_type != 'No-bd':
        continue
    else:
        if c_ratio >= np.mean(df_bin_chr2_hg38_250_new_part['cell_ratio']):
            if bd_type_la != 'No-bd':
                bd_type_adjust[i] = bd_type_la
            elif bd_type_ne != 'No-bd':
                bd_type_adjust[i] = bd_type_ne

        
df_bin_chr2_hg38_250_new_part['bd_type_adjust'] = bd_type_adjust



cell_ratio_l = []
bd_type_l = []
df_bd_cell_ratio = pd.DataFrame(columns = ['cell_ratio', 'bd_type'])
for i in range(len(df_bin_chr2_hg38_250_new_part)):
    bd_type = df_bin_chr2_hg38_250_new_part['bd_type_adjust'][i]
    if bd_type == 'No-overlap':
        continue
    bd_type_l.append(bd_type)
    cell_ratio_l.append(cts[i] / cell_num)

df_bd_cell_ratio['cell_ratio'] = cell_ratio_l
df_bd_cell_ratio['bd_type'] = bd_type_l


for bd_type in list(np.unique(df_bd_cell_ratio['bd_type'])):
    print('This is ' + bd_type)
    df_bd_cell_ratio_part = df_bd_cell_ratio[df_bd_cell_ratio['bd_type'] == bd_type]
    print(len(df_bd_cell_ratio_part))
    print(np.median(df_bd_cell_ratio_part['cell_ratio']))
  
df_bd_cell_ratio_part = df_bd_cell_ratio[df_bd_cell_ratio['cell_ratio'] != 0]

bd_type_use = ['No-bd', 'sharp_strong', 'sharp_weak', 'wide']
for i in range(len(bd_type_use)):
    bd_type1 = bd_type_use[i]
    for j in range(i, len(bd_type_use)):
        bd_type2 = bd_type_use[j]
        if bd_type2 == bd_type1:
            continue
        df_cell_ratio1 = df_bd_cell_ratio_part[df_bd_cell_ratio_part['bd_type'] == bd_type1]
        df_cell_ratio2 = df_bd_cell_ratio_part[df_bd_cell_ratio_part['bd_type'] == bd_type2]
        sta, pvalue = scipy.stats.mannwhitneyu(df_cell_ratio1['cell_ratio'], df_cell_ratio2['cell_ratio'])
        print('This is ' + bd_type1 + ' and ' + bd_type2)
        print(pvalue)
    print('\n')


region_color = {'sharp_strong': '#D65F4D', 'sharp_weak': '#459457', 'wide': '#4392C3', 'No-bd': '#999999'}
order_use = ['No-bd', 'sharp_weak', 'sharp_strong', 'wide']
color_use = []
for bd_type in order_use:
    color_use.append(region_color[bd_type])
    
    
plt.figure(figsize= (5, 5))
sns.boxplot(x = 'bd_type', y = 'cell_ratio', data = df_bd_cell_ratio_part, fliersize = 0, order = order_use, palette = color_use, saturation = 1)
plt.ylim([-0.01, 0.22])
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.6)
ax.spines['left'].set_linewidth(1.6)
ax.spines['right'].set_linewidth(1.6)
ax.spines['top'].set_linewidth(1.6)
ax.tick_params(axis = 'y', length=5, width = 1.6)
ax.tick_params(axis = 'x', length=5, width = 1.6)
plt.savefig(r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\single_cell_image\result' + '/' + 'bd_type_cell_ratio_compare.svg', format = 'svg', transparent = True) 
plt.show()
#fig = plt.gcf() #获取当前figure
#plt.close(fig)   


def draw_single_cell_contact_map_with_cell_ratio_bd_type(st, ed, single_cell_contact_map, cts, type_symbol, probe_name_l, bin_size = 20):
    x_axis_range = range(len(cts[st:ed]))
    st_prob = probe_name_l[st]
    ed_prob = probe_name_l[ed]
    Chr = st_prob.split(':')[0]
    st_name = int(st_prob.split(':')[-1].split('-')[0]) - 1
    ed_name = int(ed_prob.split(':')[-1].split('-')[-1]) - 1
    region_name = Chr + ':' + str(st_name) + '-' + str(ed_name)
    x_ticks_l = []
    y_ticks_l = []
    cord_list = []
    for i in range(ed - st):
        if i % bin_size == 0:
            cord_list.append(i)
            mid_prob = probe_name_l[st + i]
            pos = int(mid_prob.split(':')[-1].split('-')[0]) - 1
            x_ticks_l.append(str(pos / 1000000))
            y_ticks_l.append(str(pos / 1000000))
    
    plt.figure(figsize=(8, 8))     
    ax0 = plt.subplot2grid((8, 6), (0, 0), rowspan=5,colspan=5)
    dense_matrix_part = single_cell_contact_map[st:ed+1, st:ed+1]
    img = ax0.imshow(dense_matrix_part, cmap='seismic', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
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

    cax = plt.subplot2grid((8, 6), (0, 5), rowspan=5,colspan=1)
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes("right", size="1.5%", pad= 0.2)
    #cbar = plt.colorbar(img, cax=cax, ticks=MultipleLocator(2.0), format="%.1f",orientation='vertical',extendfrac='auto',spacing='uniform')
    cbaxes = inset_axes(cax, width="30%", height="60%", loc=3) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])
  
    ax1 = plt.subplot2grid((8, 6), (5, 0), rowspan=1, colspan=5, sharex=ax0)    
    ax1.plot(x_axis_range, cts[st:ed], marker = '.', linewidth = 2, c = 'black')
    plt.ylabel('cell_ratio')
    ax1.set_xticks([])
    #ax1.set_yticks([])
    ax1.spines['bottom'].set_linewidth(1.6)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(1.6)
    ax1.spines['top'].set_linewidth(1.6)
    ax1.tick_params(axis = 'y', length=5, width = 1.6)
    ax1.tick_params(axis = 'x', length=5, width = 1.6)
    
    ax2 = plt.subplot2grid((8, 6), (6, 0), rowspan=1,colspan=5, sharex=ax0)    
    bd_data = []
    cmap=['#B0B0B0','#459457','#D65F4D','#4392C3']
    my_cmap = ListedColormap(cmap)
    bounds=[0,0.9,1.9,2.9,3.9]
    norm = matplotlib.colors.BoundaryNorm(bounds, my_cmap.N)      
    for i in range(10):
        bd_data.append(type_symbol[st:ed])
    #bd_data_expand = np.reshape(np.array(bd_data), (10, len(bd_data[0])))
    ax2.imshow(bd_data, cmap = my_cmap, norm = norm)
    plt.ylabel('bd type')
    ax2.spines['bottom'].set_linewidth(1.6)
    ax2.spines['left'].set_linewidth(1.6)
    ax2.spines['right'].set_linewidth(1.6)
    ax2.spines['top'].set_linewidth(1.6)
    ax2.tick_params(axis = 'y', length=0, width = 0)
    ax2.tick_params(axis = 'x', length=0, width = 0)    
    #plt.savefig(save_name, format = 'svg', transparent = True) 
    #plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)   

single_cell_contact_map = np.load(r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\single_cell_image' + '/' + 'All_single_cell_freq_contact_map.npy')

type_symbol = []
for i in range(len(df_bin_chr2_hg38_250_new_part)):
    bd_type = df_bin_chr2_hg38_250_new_part['bd_type_adjust'][i]
    if bd_type == 'No-bd':
        type_symbol.append(0)
    elif bd_type == 'None':
        type_symbol.append(0)
    else:
        ind = ['sharp_weak', 'sharp_strong', 'wide'].index(bd_type)
        type_symbol.append(ind+1)
   
st = 180
ed = 265
draw_single_cell_contact_map_with_cell_ratio_bd_type(st, ed, single_cell_contact_map, cts, type_symbol, probe_name_l, bin_size = 20)


def compare_single_cell_image_bd_probablity_250(df_bin_chr2_hg38_250_part, indictor_type, region_type_use, region_color_use, save_name = '', expand_len = 25, random_num = (10, 300)):
    df_indictor_pos = pd.DataFrame(columns = ['indictor_l', 'pos_l', 'type_l'])
    indictor_l = []
    pos_l = []
    type_l = []
    df_indictor = pd.DataFrame(df_bin_chr2_hg38_250_part[indictor_type])   
    for i in range(len(df_bin_chr2_hg38_250_part)):
        type_ = df_bin_chr2_hg38_250_part['bd_type'][i]
        if type_ not in region_type_use:
            continue
        mid1 = i+1
        if mid1 < expand_len + 1 or mid1 > len(df_indictor) - expand_len -1:
            continue                      
        ind_vec1 = df_indictor[indictor_type][mid1 - expand_len : mid1+expand_len+1]        
        indictor_l += list(ind_vec1)
        pos_l += list(range(len(ind_vec1)))
        type_l += [type_ for j in range(len(ind_vec1)) ]
    
    random_vec_all = {}
    for j in range(random_num[0]):
        random_vec_l = []
        random_sample = 0
        while random_sample < random_num[-1]:
            mid2 = random.randint(0, len(df_indictor)-1)
            if mid2 < expand_len + 1 or mid2 > len(df_indictor) - expand_len -1:
                continue
            ind_vec2 = df_indictor[indictor_type][mid2 - expand_len : mid2+expand_len+1]
            random_vec_l.append(list(ind_vec2))
            random_sample += 1
        random_vec_all[j] = random_vec_l    
    random_vec_mean = []
    for k in range(len(random_vec_all[0])):
        random_use = []
        for j in range(random_num[0]):
            random_use.append(random_vec_all[j][k])
        random_use = np.array(random_use)
        random_vec_mean.append(np.mean(random_use, axis = 0))    
    for i in range(len(random_vec_mean)):
        ind_vec2 = random_vec_mean[i]
        indictor_l += list(ind_vec2)
        pos_l += list(range(len(ind_vec2)))
        type_l += ['random' for j in range(len(ind_vec2)) ]

    df_indictor_pos['indictor_l'] = indictor_l
    df_indictor_pos['pos_l'] = pos_l
    df_indictor_pos['type_l'] = type_l
    
    list_type = region_type_use + ['random']        
    df_indictor_pos['type_l'] = df_indictor_pos['type_l'].astype('category')
    df_indictor_pos['type_l'].cat.reorder_categories(list_type, inplace=True)
    df_indictor_pos.sort_values('type_l', inplace=True)            
    color_l = [] 
    for type_ in region_type_use:
        color_l.append(region_color_use[type_])
    color_l.append('#B0B0B0')    
    plt.figure(figsize=(6,5))
    sns.lineplot(x = 'pos_l', y = 'indictor_l', data = df_indictor_pos, hue = 'type_l',  style="type_l", palette = color_l, markers=True, dashes=False, linewidth = 3, ci = 95)
    if expand_len == 10: 
        plt.xticks([0, 5, 10, 15, 20], ['-500kb', '-250kb', 'boundary center', '250kb', '500kb'], FontSize = 10)
    elif expand_len == 7:
        plt.xticks([0, 3, 7, 11, 15], ['-350kb', '-200kb', 'boundary center', '200kb', '350kb'], FontSize = 10)
    plt.yticks(FontSize = 10)
    #if indictor_type == 'DI':
        #plt.ylim([-100, 100])
    plt.ylabel(indictor_type,  fontSize = 12)
    plt.xlabel('',  fontSize = 0)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.6)
    ax.spines['left'].set_linewidth(1.6)
    ax.spines['right'].set_linewidth(1.6)
    ax.spines['top'].set_linewidth(1.6)
    ax.tick_params(axis = 'y', length=5, width = 1.6)
    ax.tick_params(axis = 'x', length=5, width = 1.6)
    #plt.subplots_adjust(left=0.17, right=0.9, top=0.9, bottom=0.1)  
    plt.legend(loc = 'best', prop = {'size':10}, fancybox = None, edgecolor = 'white', facecolor = None, title = False, title_fontsize = 0)
    if save_name != '':    
        plt.savefig(save_name, format = 'svg', transparent = True) 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)   

indictor_type = 'cell_ratio'
region_type_use = ['sharp_strong', 'sharp_weak', 'wide' , 'No_overlap']
region_color_use = {'sharp_strong': '#D65F4D', 'sharp_weak': '#459457', 'wide': '#4392C3', 'No_overlap':'#999999'}
compare_single_cell_image_bd_probablity_250(df_bin_chr2_hg38_250_part, indictor_type, region_type_use, region_color_use, save_name = '', expand_len = 6, random_num = (10, 300))



##### boundary region case find

symbol_dic = {'No-bd':0, 'sharp_weak':1, 'sharp_strong':2, 'wide':3}
def get_bd_type_symbol(df_bd_region_type, bd_score_cell_combine):
    bd_symbol = np.zeros(len(bd_score_cell_combine))
    for i in range(len(df_bd_region_type)):
        region = df_bd_region_type['region'][i]
        bd_type = df_bd_region_type['region_type_adjust'][i]
        symbol = symbol_dic[bd_type]
        for x in region:
            bd_symbol[x] = symbol
    return bd_symbol

def draw_bd_region_case(st, ed, contact_map, bd_score_cell_combine, bd_symbol, Chr, save_name = '', bin_size = 8, resolution = 50000):
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



# get other information
cell_type = 'GM12878'
enzyme = 'MboI'
mat_dense = copy.deepcopy(hic_mat_all_cell_replicate[cell_type][enzyme]['iced'])
mat_blank_row = get_blank_row_in_matrix(mat_dense)
df_bd_insul_pvalue = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['pvalue'])
result_record = copy.deepcopy(result_record_all[cell_type + '_' + enzyme]['BD_region'])
df_boundary_region_combine = copy.deepcopy(result_record['Combine']['bd_region'])
bd_score_cell_combine = copy.deepcopy(result_record['Combine']['TAD_score'])
df_bd_region_type = copy.deepcopy(bd_region_type_record[cell_type + '_' + enzyme])



bd_symbol = get_bd_type_symbol(df_bd_region_type, bd_score_cell_combine)

Chr = 'chr2'
st = 1422 - 30
ed = 1422 + 30
save_name = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\case_plot' + '/' + 'bd_region_heatmap_1.svg'
draw_bd_region_case(st, ed, mat_dense, bd_score_cell_combine, bd_symbol, Chr, save_name = save_name, bin_size = 10, resolution = 50000)


Chr = 'chr2'
st = 2734 - 32
ed = 2734 + 22
save_name = r'E:\Users\dcdang\TAD_intergate\final_run\TAD_seperation_landscape\boundary_region\analysis_new\case_plot' + '/' + 'bd_region_heatmap_2.svg'
draw_bd_region_case(st, ed, mat_dense, bd_score_cell_combine, bd_symbol, Chr, save_name = save_name, bin_size = 10, resolution = 50000)










