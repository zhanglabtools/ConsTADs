# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 17:13:13 2022

@author: dcdang
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 21:36:58 2022

@author: dcdang
"""

import pandas as pd
import numpy as np
import scipy.sparse 
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import scipy


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


###########################  TAD number and size comparison
  
def compare_TAD_number_between_method_or_datasets(TAD_result_all_cell_type, method_list, data_type_list, save_add):
    df_num_record_across_method = pd.DataFrame(columns = ['data', 'method1', 'method2', 'm1_num', 'm2_num', 'num_dif', 'type'])
    data_l = []
    m_l_1 = []
    m_l_2 = []
    m_num_1 = []
    m_num_2 = []
    num_dif_l = []
    type_l = []
    for i in range(len(data_type_list)):
        data_t = data_type_list[i]
        cell_t = data_t.split('_')[0]
        enzyme_t = data_t.split('_')[-1]        
        TAD_res = copy.deepcopy(TAD_result_all_cell_type[cell_t][enzyme_t])
        for j in range(len(method_list)):
            m1 = method_list[j]
            df_tad_1 = TAD_res[m1]['TAD_domain']
            for k in range(j+1, len(method_list)):
                m2 = method_list[k]
                df_tad_2 = TAD_res[m2]['TAD_domain']
                data_l.append(data_t)
                m_l_1.append(m1)
                m_l_2.append(m2)
                m_num_1.append(len(df_tad_1))
                m_num_2.append(len(df_tad_2))
                num_dif_l.append(np.abs(len(df_tad_1) - len(df_tad_2)))
                type_l.append('across method')
    df_num_record_across_method['data'] = data_l
    df_num_record_across_method['method1'] = m_l_1
    df_num_record_across_method['method2'] = m_l_2
    df_num_record_across_method['m1_num'] = m_num_1
    df_num_record_across_method['m2_num'] = m_num_2 
    df_num_record_across_method['num_dif'] = num_dif_l
    df_num_record_across_method['type'] = type_l

    df_num_record_across_dataset = pd.DataFrame(columns = ['method', 'data1', 'data2', 'd1_num', 'd2_num', 'num_dif', 'type'])
    data_l_1 = []
    data_l_2 = []
    m_l = []
    d_num_1 = []
    d_num_2 = []
    num_dif_l = []
    type_l = []
    for i in range(len(method_list)):
        method = method_list[i]
        for j in range(len(data_type_list)):
            data_1 = data_type_list[j]
            cell_1 = data_1.split('_')[0]
            enzyme_1 = data_1.split('_')[1]
            TAD_res1 = copy.deepcopy(TAD_result_all_cell_type[cell_1][enzyme_1])
            df_tad_1 = TAD_res1[method]['TAD_domain']
            for k in range(j+1, len(data_type_list)):
                data_2 = data_type_list[k]
                cell_2 = data_2.split('_')[0]
                enzyme_2 = data_2.split('_')[1]
                TAD_res2 = copy.deepcopy(TAD_result_all_cell_type[cell_2][enzyme_2])
                df_tad_2 = TAD_res2[method]['TAD_domain']
                m_l.append(method)
                data_l_1.append(data_1)
                data_l_2.append(data_2)
                d_num_1.append(len(df_tad_1))
                d_num_2.append(len(df_tad_2))
                num_dif_l.append(np.abs(len(df_tad_1) - len(df_tad_2)))
                type_l.append('across dataset')
    df_num_record_across_dataset['method'] = m_l
    df_num_record_across_dataset['data1'] = data_l_1
    df_num_record_across_dataset['data2'] = data_l_2
    df_num_record_across_dataset['d1_num'] = d_num_1
    df_num_record_across_dataset['d2_num'] = d_num_2 
    df_num_record_across_dataset['num_dif'] = num_dif_l
    df_num_record_across_dataset['type'] = type_l
    
    df_draw_record = pd.DataFrame(columns = ['method', 'type', 'num_dif'])
    m_l = []
    type_l = []
    num_l = []
    order_use = []
    for method in method_list:
        df_num_method_m = df_num_record_across_method[(df_num_record_across_method['method1'] == method) | (df_num_record_across_method['method2'] == method)]   
        m_l += [method for i in range(len(df_num_method_m))]
        type_l += ['across method' for i in range(len(df_num_method_m))]
        num_l += list(df_num_method_m['num_dif'])
        dif_m = np.array(df_num_method_m['num_dif'])
        order_use.append(np.mean(dif_m))
        
        df_num_data_m = df_num_record_across_dataset[(df_num_record_across_dataset['method'] == method)]
        m_l += [method for i in range(len(df_num_data_m))]
        type_l += ['across dataset' for i in range(len(df_num_data_m))]
        num_l += list(df_num_data_m['num_dif'])
        dif_d = np.array(df_num_data_m['num_dif'])
    
        sta, pvalue = scipy.stats.mannwhitneyu(dif_m, dif_d)
        print(method)
        print(sta, pvalue)
    df_draw_record['method'] = m_l
    df_draw_record['type'] = type_l
    df_draw_record['num_dif'] = num_l
    ord_ = np.argsort(order_use)
    method_ord = np.array(method_list)[ord_]
    color_face = ['#4392C3', '#D65F4D']
    plt.figure(figsize= (8, 3))
    ax = sns.barplot(x="method", y="num_dif", hue = 'type', order = method_ord,   
                data=df_draw_record, capsize = 0.2, saturation = 8, 
                errcolor = 'black', errwidth = 2,
                ci = 95, edgecolor='black', palette = color_face)    
    plt.ylim([0, 630])
    plt.xticks(rotation= -27, FontSize = 8)
    plt.yticks(FontSize = 8)
    plt.ylabel('Difference in TAD number',  FontSize = 8)
    plt.xlabel('method', FontSize = 0)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['right'].set_linewidth(1.8)
    ax.spines['top'].set_linewidth(1.8)
    ax.tick_params(axis = 'y', length=4, width = 1.8)
    ax.tick_params(axis = 'x', length=4, width = 1.8)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    plt.savefig(save_add + '/' + 'Method_TAD_number_compare_version.svg', format = 'svg', transparent = True) 
    plt.show()
    fig = plt.gcf() #获取当前figure
    #plt.close(fig)
    return df_num_record_across_method, df_num_record_across_dataset
       
def compare_TAD_length_between_method_or_datasets(TAD_result_all_cell_type, method_list, data_type_list, save_add):
    df_num_record_across_method = pd.DataFrame(columns = ['data', 'method1', 'method2', 'm1_len', 'm2_len', 'len_dif', 'type'])
    data_l = []
    m_l_1 = []
    m_l_2 = []
    m_num_1 = []
    m_num_2 = []
    num_dif_l = []
    type_l = []
    for i in range(len(data_type_list)):
        data_t = data_type_list[i]
        cell_t = data_t.split('_')[0]
        enzyme_t = data_t.split('_')[-1]        
        TAD_res = copy.deepcopy(TAD_result_all_cell_type[cell_t][enzyme_t])
        for j in range(len(method_list)):
            m1 = method_list[j]
            df_tad_1 = TAD_res[m1]['TAD_domain']
            for k in range(j+1, len(method_list)):
                m2 = method_list[k]
                df_tad_2 = TAD_res[m2]['TAD_domain']
                len1 = np.mean(df_tad_1['end'] - df_tad_1['start'])  
                len2 = np.mean(df_tad_2['end'] - df_tad_2['start'])  
                data_l.append(data_t)
                m_l_1.append(m1)
                m_l_2.append(m2)
                m_num_1.append(len1)
                m_num_2.append(len2)
                num_dif_l.append(np.abs(len1 - len2))
                type_l.append('across method')
    df_num_record_across_method['data'] = data_l
    df_num_record_across_method['method1'] = m_l_1
    df_num_record_across_method['method2'] = m_l_2
    df_num_record_across_method['m1_len'] = m_num_1
    df_num_record_across_method['m2_len'] = m_num_2 
    df_num_record_across_method['len_dif'] = num_dif_l
    df_num_record_across_method['type'] = type_l

    df_num_record_across_dataset = pd.DataFrame(columns = ['method', 'data1', 'data2', 'd1_len', 'd2_len', 'len_dif', 'type'])
    data_l_1 = []
    data_l_2 = []
    m_l = []
    d_num_1 = []
    d_num_2 = []
    num_dif_l = []
    type_l = []
    for i in range(len(method_list)):
        method = method_list[i]
        for j in range(len(data_type_list)):
            data_1 = data_type_list[j]
            cell_1 = data_1.split('_')[0]
            enzyme_1 = data_1.split('_')[1]
            TAD_res1 = copy.deepcopy(TAD_result_all_cell_type[cell_1][enzyme_1])
            df_tad_1 = TAD_res1[method]['TAD_domain']
            for k in range(j+1, len(data_type_list)):
                data_2 = data_type_list[k]
                cell_2 = data_2.split('_')[0]
                enzyme_2 = data_2.split('_')[1]
                TAD_res2 = copy.deepcopy(TAD_result_all_cell_type[cell_2][enzyme_2])
                df_tad_2 = TAD_res2[method]['TAD_domain']
                len1 = np.mean(df_tad_1['end'] - df_tad_1['start'])  
                len2 = np.mean(df_tad_2['end'] - df_tad_2['start'])             
                m_l.append(method)
                data_l_1.append(data_1)
                data_l_2.append(data_2)
                d_num_1.append(len1)
                d_num_2.append(len2)
                num_dif_l.append(np.abs(len1- len2))
                type_l.append('across dataset')
    df_num_record_across_dataset['method'] = m_l
    df_num_record_across_dataset['data1'] = data_l_1
    df_num_record_across_dataset['data2'] = data_l_2
    df_num_record_across_dataset['d1_len'] = d_num_1
    df_num_record_across_dataset['d2_len'] = d_num_2 
    df_num_record_across_dataset['len_dif'] = num_dif_l
    df_num_record_across_dataset['type'] = type_l

    df_draw_record = pd.DataFrame(columns = ['method', 'type', 'len_dif'])
    m_l = []
    type_l = []
    num_l = []
    order_use = []
    for method in method_list:
        df_num_method_m = df_num_record_across_method[(df_num_record_across_method['method1'] == method) | (df_num_record_across_method['method2'] == method)]   
        m_l += [method for i in range(len(df_num_method_m))]
        type_l += ['across method' for i in range(len(df_num_method_m))]
        num_l += list(df_num_method_m['len_dif'])
        dif_m = np.array(df_num_method_m['len_dif'])
        order_use.append(np.mean(dif_m))
        
        df_num_data_m = df_num_record_across_dataset[(df_num_record_across_dataset['method'] == method)]
        m_l += [method for i in range(len(df_num_data_m))]
        type_l += ['across dataset' for i in range(len(df_num_data_m))]
        num_l += list(df_num_data_m['len_dif'])
        dif_d = np.array(df_num_data_m['len_dif'])
    
        sta, pvalue = scipy.stats.mannwhitneyu(dif_m, dif_d)
        print(method)
        print(sta, pvalue)
    df_draw_record['method'] = m_l
    df_draw_record['type'] = type_l
    df_draw_record['len_dif'] = np.array(num_l) / 1000
    
    ord_ = np.argsort(order_use)
    method_ord = np.array(method_list)[ord_]
    color_face = ['#4392C3', '#D65F4D']
    plt.figure(figsize= (8, 3))
    ax = sns.barplot(x="method", y="len_dif", hue = 'type', order = method_ord,   
                data=df_draw_record, capsize = 0.2, saturation = 8, 
                errcolor = 'black', errwidth = 2,
                ci = 95, edgecolor='black', palette = color_face)    
    plt.ylim([0, 1600])
    plt.xticks(rotation= -27, FontSize = 8)
    plt.yticks(FontSize = 8)
    plt.ylabel('Difference in TAD size (kb)',  FontSize = 8)
    plt.xlabel('method', FontSize = 0)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['right'].set_linewidth(1.8)
    ax.spines['top'].set_linewidth(1.8)
    ax.tick_params(axis = 'y', length=4, width = 1.8)
    ax.tick_params(axis = 'x', length=4, width = 1.8)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)

    plt.savefig(save_add + '/' + 'Method_TAD_length_compare_version.svg', format = 'svg', transparent = True) 
    plt.show()  
    fig = plt.gcf() #获取当前figure
    #plt.close(fig)    
    return df_num_record_across_method, df_num_record_across_dataset
 
    

save_add = r'E:/Users/dcdang/TAD_intergate/final_run/compare/Result_new/single_method_type' 

data_type_list = []
for cell in list(TAD_result_all_cell_type.keys()):
    enzyme_list = list(TAD_result_all_cell_type[cell].keys())
    for enzyme in enzyme_list:
        data_type_list.append(cell + '_' + enzyme)
     

df_num_across_method, df_num_across_dataset = compare_TAD_number_between_method_or_datasets(TAD_result_all_cell_type, method_list, data_type_list, save_add)

df_len_across_method, df_len_across_dataset = compare_TAD_length_between_method_or_datasets(TAD_result_all_cell_type, method_list, data_type_list, save_add)


###### jaccard index for TAD boundary

def get_jaccard_index(ind1, ind1_expand, ind2, ind2_expand):
     insersection1 = list(set(ind1).intersection(set(ind2_expand)))
     insersection2 = list(set(ind2).intersection(set(ind1_expand)))
     union = list(set(ind1).union(set(ind2)))
     jaccard_index = np.max([len(insersection1), len(insersection2)]) / len(union)
     if jaccard_index >= 0.99:
          jaccard_index = 1
     return jaccard_index

def get_ind_expand(method_list, result_all, bin_name_list, Chr):
     ind_all = {}
     ind_all_expand = {}
     for i in range(len(method_list)):
         method = method_list[i]
         print('Dealing with ' + method)
         df_TAD = result_all[method]['TAD_domain']
         bd_list_all = list(df_TAD['boundary_st']) + list(df_TAD['boundary_ed'])
         bd_list_all = list(set(bd_list_all))
         ind1 = []
         ind1_expand = []
         for i in range(len(bd_list_all)):
               name = bd_list_all[i]
               ind = list(bin_name_list[Chr]).index(name)
               ind1.append(ind)
               ind1_expand.append(ind)
               ind1_expand.append(ind-1)
               ind1_expand.append(ind+1)
         ind_all[method] = ind1
         ind_all_expand[method] = ind1_expand
     return ind_all, ind_all_expand

def compare_TAD_jaccard_index_between_method_or_datasets(Chr, bin_name_list, TAD_result_all_cell_type, method_list, data_type_list, save_add):    
    index_record = {}
    for data in data_type_list:
        print('For ' + data)
        index_record[data] = {}
        cell = data.split('_')[0]
        enzyme = data.split('_')[-1]
        result_target = copy.deepcopy(TAD_result_all_cell_type[cell][enzyme])
        ind_all_t, ind_all_expand_t = get_ind_expand(method_list, result_target, bin_name_list, Chr)
        index_record[data]['index'] = ind_all_t
        index_record[data]['index_expand'] = ind_all_expand_t
    
    df_JI_record_across_method = pd.DataFrame(columns = ['data', 'method1', 'method2', 'jaccard_index', 'type'])
    data_l = []
    m_l_1 = []
    m_l_2 = []
    Jaccard_ind_l = []
    type_l = []
    for i in range(len(data_type_list)):
        data_t = data_type_list[i]
        cell_t = data_t.split('_')[0]
        enzyme_t = data_t.split('_')[-1]               
        for j in range(len(method_list)):
            m1 = method_list[j]
            ind1 = index_record[data_t]['index'][m1]
            ind1_expand = index_record[data_t]['index_expand'][m1]
            for k in range(j+1, len(method_list)):
                m2 = method_list[k]
                ind2 = index_record[data_t]['index'][m2]
                ind2_expand = index_record[data_t]['index_expand'][m2]
                jaccard_index = get_jaccard_index(ind1, ind1_expand, ind2, ind2_expand)
                data_l.append(data_t)
                m_l_1.append(m1)
                m_l_2.append(m2)
                Jaccard_ind_l.append(jaccard_index)
                type_l.append('across method')
    df_JI_record_across_method['data'] = data_l
    df_JI_record_across_method['method1'] = m_l_1
    df_JI_record_across_method['method2'] = m_l_2
    df_JI_record_across_method['jaccard_index'] = Jaccard_ind_l
    df_JI_record_across_method['type'] = type_l

    df_JI_record_across_dataset = pd.DataFrame(columns = ['method', 'data1', 'data2', 'jaccard_index', 'type'])
    data_l_1 = []
    data_l_2 = []
    m_l = []
    Jaccard_ind_l = []
    type_l = []
    for i in range(len(method_list)):
        method = method_list[i]
        for j in range(len(data_type_list)):
            data_1 = data_type_list[j]
            cell_1 = data_1.split('_')[0]
            enzyme_1 = data_1.split('_')[1]
            ind1 = index_record[data_1]['index'][method]
            ind1_expand = index_record[data_1]['index_expand'][method]           
            for k in range(j+1, len(data_type_list)):
                data_2 = data_type_list[k]
                cell_2 = data_2.split('_')[0]
                enzyme_2 = data_2.split('_')[1]
                ind2 = index_record[data_2]['index'][method]
                ind2_expand = index_record[data_2]['index_expand'][method]           
                jaccard_index = get_jaccard_index(ind1, ind1_expand, ind2, ind2_expand)                
                m_l.append(method)
                data_l_1.append(data_1)
                data_l_2.append(data_2)
                Jaccard_ind_l.append(jaccard_index)
                type_l.append('across dataset')
    df_JI_record_across_dataset['method'] = m_l
    df_JI_record_across_dataset['data1'] = data_l_1
    df_JI_record_across_dataset['data2'] = data_l_2
    df_JI_record_across_dataset['jaccard_index'] = Jaccard_ind_l 
    df_JI_record_across_dataset['type'] = type_l

    df_draw_record = pd.DataFrame(columns = ['method', 'type', 'jaccard_index'])
    m_l = []
    type_l = []
    num_l = []
    order_use = []
    for method in method_list:
        df_num_method_m = df_JI_record_across_method[(df_JI_record_across_method['method1'] == method) | (df_JI_record_across_method['method2'] == method)]   
        m_l += [method for i in range(len(df_num_method_m))]
        type_l += ['across method' for i in range(len(df_num_method_m))]
        num_l += list(df_num_method_m['jaccard_index'])
        dif_m = np.array(df_num_method_m['jaccard_index'])
        order_use.append(np.mean(dif_m))
        
        df_num_data_m = df_JI_record_across_dataset[(df_JI_record_across_dataset['method'] == method)]
        m_l += [method for i in range(len(df_num_data_m))]
        type_l += ['across dataset' for i in range(len(df_num_data_m))]
        num_l += list(df_num_data_m['jaccard_index'])
        dif_d = np.array(df_num_data_m['jaccard_index'])
    
        sta, pvalue = scipy.stats.mannwhitneyu(dif_m, dif_d)
        print(method)
        print(sta, pvalue)
    df_draw_record['method'] = m_l
    df_draw_record['type'] = type_l
    df_draw_record['jaccard_index'] = num_l
    
    ord_ = np.argsort(order_use)
    method_ord = np.array(method_list)[ord_]
    color_face = ['#4392C3', '#D65F4D']
    plt.figure(figsize= (8, 3))
    ax = sns.barplot(x="method", y="jaccard_index", hue = 'type', order = method_ord,   
                data=df_draw_record, capsize = 0.2, saturation = 8, 
                errcolor = 'black', errwidth = 2,
                ci = 95, edgecolor='black', palette = color_face)    
    plt.ylim([0, 0.8])
    plt.xticks(rotation= -27, FontSize = 8)
    plt.yticks(FontSize = 8)
    plt.ylabel('Jaccard index between boundary sets',  FontSize = 8)
    plt.xlabel('method', FontSize = 0)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['right'].set_linewidth(1.8)
    ax.spines['top'].set_linewidth(1.8)
    ax.tick_params(axis = 'y', length=4, width = 1.8)
    ax.tick_params(axis = 'x', length=4, width = 1.8)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    ax.legend_ = None
    plt.savefig(save_add + '/' + 'Jaccard_index_compare_version.svg', format = 'svg', transparent = True) 
    plt.show()  
    fig = plt.gcf() #获取当前figure
    #plt.close(fig)    
    return df_JI_record_across_method, df_JI_record_across_dataset
 
ref_add = 'E:/Users/dcdang/multi_sepcies_project/multi_species_hic_results/reference_size'
species = 'human'
resolution = 50000
chr_ref = get_chr_ref(species)
bin_name_list = get_bin_name_list_for_chr(species, resolution, chr_ref, ref_add)
Chr = 'chr2'
   
df_JI_record_across_method, df_JI_record_across_dataset = compare_TAD_jaccard_index_between_method_or_datasets(Chr, bin_name_list, TAD_result_all_cell_type, method_list, data_type_list, save_add)


#################################### Measure of Cordance for TAD domain

def inter_domain_region_add(df_TAD_m1, chr_length):
    df_TAD_m1_complete = pd.DataFrame(columns = ['chr', 'start', 'end'])
    st_list = []
    ed_list = []
    if df_TAD_m1['start'][0] != 0:
        st_list.append(0)
        ed_list.append(df_TAD_m1['start'][0])
    for i in range(len(df_TAD_m1) - 1):
        st = df_TAD_m1['start'][i]
        ed = df_TAD_m1['end'][i]
        st_list.append(st)
        ed_list.append(ed)
        st_n = df_TAD_m1['start'][i+1]
        if st_n != ed:
            st_list.append(ed)
            ed_list.append(st_n)
    
    st_list.append(df_TAD_m1['start'].iloc[-1])
    ed_list.append(df_TAD_m1['end'].iloc[-1])
    if df_TAD_m1['end'].iloc[-1] != chr_length:
        st_list.append(df_TAD_m1['end'].iloc[-1])
        ed_list.append(chr_length)
    df_TAD_m1_complete['start'] = st_list
    df_TAD_m1_complete['end'] = ed_list
    df_TAD_m1_complete['chr'] = [df_TAD_m1['chr'][0] for i in range(len(df_TAD_m1_complete))]
    return df_TAD_m1_complete

def get_overlap_length(st1, ed1, st2, ed2):
    if ed1 <= st2 or st1 >= ed2:
        overlap = 0
    elif st2 >= st1 and st2 <= ed1 and ed2 > ed1:
        overlap = ed1 - st2
    elif st1 >= st2 and st1 <= ed2 and ed1 > ed2:
        overlap = ed2 - st1
    elif st1 >= st2 and ed1 <= ed2:
        overlap = ed1 - st1
    elif st2 >= st1 and ed2 <= ed1:
        overlap = ed2 - st2
    return overlap

def calculate_MoC_of_TAD_domain(df_TAD_m1, df_TAD_m2):
    record = []
    moc_value = 0
    n1 = len(df_TAD_m1)
    n2 = len(df_TAD_m2)
    for i in range(len(df_TAD_m1)):
        st1 = df_TAD_m1['start'][i]
        ed1 = df_TAD_m1['end'][i]
        length1 = int((ed1 - st1) / 1000)
        for j in range(len(df_TAD_m2)):
            st2 = df_TAD_m2['start'][j]
            ed2 = df_TAD_m2['end'][j]
            length2 = int((ed2 - st2) / 1000)
            overlap = get_overlap_length(st1, ed1, st2, ed2)
            overlap = int(overlap / 1000)
            add_value = int(overlap)**2 / (length1 * length2) 
            moc_value += add_value  
            record.append(add_value)
    MoC_value = (1/(np.sqrt(n1*n2) - 1)) * (moc_value - 1)
    return MoC_value
   
def compare_TAD_MoC_between_method_or_datasets(chr_length, TAD_result_all_cell_type, method_list, data_type_list, save_add):       
    df_moc_record_across_method = pd.DataFrame(columns = ['data', 'method1', 'method2', 'moc', 'type'])
    data_l = []
    m_l_1 = []
    m_l_2 = []
    moc_l = []
    type_l = []
    print('Same data, different method')
    for i in range(len(data_type_list)):
        data_t = data_type_list[i]
        cell_t = data_t.split('_')[0]
        enzyme_t = data_t.split('_')[-1]               
        TAD_res = copy.deepcopy(TAD_result_all_cell_type[cell_t][enzyme_t])
        print('For ' + data_t)
        for j in range(len(method_list)):
            m1 = method_list[j]
            df_tad_m1 = TAD_res[m1]['TAD_domain']
            df_domain_m1 = inter_domain_region_add(df_tad_m1, chr_length)                
            for k in range(j+1, len(method_list)):
                m2 = method_list[k]
                print('Method1 is ' + m1 + ' ; ' + 'Method2 is ' + m2)
                df_tad_m2 = TAD_res[m2]['TAD_domain']
                df_domain_m2 = inter_domain_region_add(df_tad_m2, chr_length)                
                if len(df_domain_m1) == 1 and len(df_domain_m2) == 1:
                    MoC_value = 1
                else:
                    MoC_value = calculate_MoC_of_TAD_domain(df_domain_m1, df_domain_m2)                                
                data_l.append(data_t)
                m_l_1.append(m1)
                m_l_2.append(m2)
                moc_l.append(MoC_value)
                type_l.append('across method')
    df_moc_record_across_method['data'] = data_l
    df_moc_record_across_method['method1'] = m_l_1
    df_moc_record_across_method['method2'] = m_l_2
    df_moc_record_across_method['moc'] = moc_l
    df_moc_record_across_method['type'] = type_l

    df_moc_record_across_dataset = pd.DataFrame(columns = ['method', 'data1', 'data2', 'moc', 'type'])
    data_l_1 = []
    data_l_2 = []
    m_l = []
    moc_l = []
    type_l = []
    print('Same method, different data')
    for i in range(len(method_list)):
        method = method_list[i]
        print('For ' + method)
        for j in range(len(data_type_list)):
            data_1 = data_type_list[j]
            cell_1 = data_1.split('_')[0]
            enzyme_1 = data_1.split('_')[1]
            TAD_res1 = copy.deepcopy(TAD_result_all_cell_type[cell_1][enzyme_1])
            df_tad_1 = TAD_res1[method]['TAD_domain']
            df_domain_1 = inter_domain_region_add(df_tad_1, chr_length)                
            for k in range(j+1, len(data_type_list)):
                data_2 = data_type_list[k]
                print('Data1 is ' + data_1 + ' ; ' + 'Data2 is ' + data_2)
                cell_2 = data_2.split('_')[0]
                enzyme_2 = data_2.split('_')[1]
                TAD_res2 = copy.deepcopy(TAD_result_all_cell_type[cell_2][enzyme_2])
                df_tad_2 = TAD_res2[method]['TAD_domain']
                df_domain_2 = inter_domain_region_add(df_tad_2, chr_length)                
                if len(df_domain_m1) == 1 and len(df_domain_m2) == 1:
                    MoC_value = 1
                else:
                    MoC_value = calculate_MoC_of_TAD_domain(df_domain_1, df_domain_2)
                m_l.append(method)
                data_l_1.append(data_1)
                data_l_2.append(data_2)
                moc_l.append(MoC_value)
                type_l.append('across dataset')
    df_moc_record_across_dataset['method'] = m_l
    df_moc_record_across_dataset['data1'] = data_l_1
    df_moc_record_across_dataset['data2'] = data_l_2
    df_moc_record_across_dataset['moc'] = moc_l 
    df_moc_record_across_dataset['type'] = type_l

    df_draw_record = pd.DataFrame(columns = ['method', 'type', 'moc'])
    m_l = []
    type_l = []
    num_l = []
    order_use = []
    for method in method_list:
        df_num_method_m = df_moc_record_across_method[(df_moc_record_across_method['method1'] == method) | (df_moc_record_across_method['method2'] == method)]   
        m_l += [method for i in range(len(df_num_method_m))]
        type_l += ['across method' for i in range(len(df_num_method_m))]
        num_l += list(df_num_method_m['moc'])
        dif_m = np.array(df_num_method_m['moc'])
        order_use.append(np.mean(dif_m))
        
        df_num_data_m = df_moc_record_across_dataset[(df_moc_record_across_dataset['method'] == method)]
        m_l += [method for i in range(len(df_num_data_m))]
        type_l += ['across dataset' for i in range(len(df_num_data_m))]
        num_l += list(df_num_data_m['moc'])
        dif_d = np.array(df_num_data_m['moc'])
    
        sta, pvalue = scipy.stats.mannwhitneyu(dif_m, dif_d)
        print(method)
        print(sta, pvalue)
    df_draw_record['method'] = m_l
    df_draw_record['type'] = type_l
    df_draw_record['moc'] = num_l
    
    ord_ = np.argsort(order_use)
    method_ord = np.array(method_list)[ord_]
    color_face = ['#4392C3', '#D65F4D']
    plt.figure(figsize= (8, 3))
    ax = sns.barplot(x="method", y="moc", hue = 'type', order = method_ord,   
                data=df_draw_record, capsize = 0.2, saturation = 8, 
                errcolor = 'black', errwidth = 2,
                ci = 95, edgecolor='black', palette = color_face)    
    plt.ylim([0, 1.0])
    plt.xticks(rotation= -27, FontSize = 8)
    plt.yticks(FontSize = 8)
    plt.ylabel('MoC between domain sets',  FontSize = 8)
    plt.xlabel('method', FontSize = 0)
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['right'].set_linewidth(1.8)
    ax.spines['top'].set_linewidth(1.8)
    ax.tick_params(axis = 'y', length=4, width = 1.8)
    ax.tick_params(axis = 'x', length=4, width = 1.8)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    ax.legend_ = None
    
    plt.savefig(save_add + '/' + 'MoC_compare_version.svg', format = 'svg', transparent = True) 
    plt.show()  
    fig = plt.gcf() #获取当前figure
    #plt.close(fig)    
    return df_moc_record_across_method, df_moc_record_across_dataset
 
    
chr_length = 243199373
df_moc_record_across_method, df_moc_record_across_dataset = compare_TAD_MoC_between_method_or_datasets(chr_length, bin_name_list, TAD_result_all_cell_type, method_list, data_type_list, save_add)


#save_data(r'E:/Users/dcdang/TAD_intergate/final_run/compare/Result_new/All_cell_type_method/df_moc_record_across_method.pkl', df_moc_record_across_method)
#save_data(r'E:/Users/dcdang/TAD_intergate/final_run/compare/Result_new/All_cell_type_method/df_moc_record_across_dataset.pkl', df_moc_record_across_dataset)
#df_moc_record_across_method = read_save_data(r'E:/Users/dcdang/TAD_intergate/final_run/compare/Result_new/All_cell_type_method/df_moc_record_across_method.pkl')
#df_moc_record_across_dataset = read_save_data(r'E:/Users/dcdang/TAD_intergate/final_run/compare/Result_new/All_cell_type_method/df_moc_record_across_dataset.pkl')








