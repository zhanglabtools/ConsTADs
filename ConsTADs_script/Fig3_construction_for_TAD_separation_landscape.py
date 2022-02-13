# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 23:37:36 2022

@author: dcdang

"""
import pandas as pd
import numpy as np
import scipy.sparse 
import copy
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pickle
import scipy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
        

########## Refining boundary score profile to construct the TAD separation landscape

def draw_bd_score_combine_ms_pvalue(mat_dense, Chr, st, ed, bd_score_final, df_bd_pvalue_result, peak_list, resolution, save_name, p_cut, target_site = [], bin_size = 10):   
    plt.figure(figsize=(6,8.7))
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
         
    ax1 = plt.subplot2grid((11, 7), (0, 0), rowspan=6,colspan=6)
    start = int(start_ / resolution)
    end = int(end_ / resolution)
    dense_matrix_part = mat_dense[start:end, start:end]
    #plt.figure(figsize=(10, 10))
    #plt.figure(figsize=(4, 4))
    #ax = sns.heatmap(data = dense_matrix_part, square=True, yticklabels=False,  xticklabels=False, cmap='gist_gray_r', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
    #img = ax1.imshow(dense_matrix_part, cmap='gist_gray_r', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
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
    ax1.set_title('TAD landscape of region:' + region_name, fontsize=12, pad = 15.0)
    cax = plt.subplot2grid((11, 7), (0, 6), rowspan=6,colspan=1)
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes("right", size="1.5%", pad= 0.2)
    #cbar = plt.colorbar(img, cax=cax, ticks=MultipleLocator(2.0), format="%.1f",orientation='vertical',extendfrac='auto',spacing='uniform')
    cbaxes = inset_axes(cax, width="30%", height="100%", loc=3) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])

    ax1_5 = plt.subplot2grid((11, 7), (6, 0), rowspan=1,colspan=6,sharex=ax1)
    ax1_5.plot(list(df_bd_pvalue_result[peak_list[-1]][st:ed]), color='black')
    ax1_5.bar(x_axis_range, list(df_bd_pvalue_result[peak_list[-1]][st:ed]), label=peak_list[-1], color='red')
    if p_cut != 0:
        ax1_5.hlines(p_cut, x_axis_range[0], x_axis_range[-1], color = 'black', linestyles = '--')
    ax1_5.spines['bottom'].set_linewidth(1.6)
    ax1_5.spines['left'].set_linewidth(1.6)
    ax1_5.spines['right'].set_linewidth(1.6)
    ax1_5.spines['top'].set_linewidth(1.6)
    ax1_5.tick_params(axis = 'y', length=5, width = 1.6)
    ax1_5.tick_params(axis = 'x', length=5, width = 1.6)
    ax1_5.set_ylabel(peak_list[-1], FontSize = 5)

    ax2 = plt.subplot2grid((11, 7), (7, 0), rowspan=1,colspan=6,sharex=ax1)
    ax2.plot(list(df_bd_pvalue_result[peak_list[-2]][st:ed]), color='black')
    ax2.bar(x_axis_range, list(df_bd_pvalue_result[peak_list[-2]][st:ed]), label=peak_list[-2], color='red')
    ax2.hlines(p_cut, x_axis_range[0], x_axis_range[-1], color = 'black', linestyles = '--')
    ax2.spines['bottom'].set_linewidth(1.6)
    ax2.spines['left'].set_linewidth(1.6)
    ax2.spines['right'].set_linewidth(1.6)
    ax2.spines['top'].set_linewidth(1.6)
    ax2.tick_params(axis = 'y', length=5, width = 1.6)
    ax2.tick_params(axis = 'x', length=5, width = 1.6)
    ax2.set_ylabel(peak_list[-2], FontSize = 5)

    ax3 = plt.subplot2grid((11, 7), (8, 0), rowspan=1,colspan=6,sharex=ax1)
    ax3.plot(list(df_bd_pvalue_result[peak_list[-3]][st:ed]), color='black')
    ax3.bar(x_axis_range, list(df_bd_pvalue_result[peak_list[-3]][st:ed]), label=peak_list[-3], color='red')
    if p_cut != 0:
        ax3.hlines(p_cut, x_axis_range[0], x_axis_range[-1], color = 'black', linestyles = '--')
    ax3.spines['bottom'].set_linewidth(1.6)
    ax3.spines['left'].set_linewidth(1.6)
    ax3.spines['right'].set_linewidth(1.6)
    ax3.spines['top'].set_linewidth(1.6)
    ax3.tick_params(axis = 'y', length=5, width = 1.6)
    ax3.tick_params(axis = 'x', length=5, width = 1.6)
    ax3.set_ylabel(peak_list[-3], FontSize = 5)

    ax4 = plt.subplot2grid((11, 7), (9, 0), rowspan=1,colspan=6,sharex=ax1)
    ax4.plot(list(df_bd_pvalue_result[peak_list[-4]][st:ed]), color='black')
    ax4.bar(x_axis_range, list(df_bd_pvalue_result[peak_list[-4]][st:ed]), label=peak_list[-4], color='red')
    if p_cut != 0:
        ax4.hlines(p_cut, x_axis_range[0], x_axis_range[-1], color = 'black', linestyles = '--')
    ax4.spines['bottom'].set_linewidth(1.6)
    ax4.spines['left'].set_linewidth(1.6)
    ax4.spines['right'].set_linewidth(1.6)
    ax4.spines['top'].set_linewidth(1.6)
    ax4.tick_params(axis = 'y', length=5, width = 1.6)
    ax4.tick_params(axis = 'x', length=5, width = 1.6)
    ax4.set_ylabel(peak_list[-4], FontSize = 5)

    ax5 = plt.subplot2grid((11, 7), (10, 0), rowspan=1,colspan=6,sharex=ax4)
    ax5.plot(list(bd_score_final['bd_score'][st:ed]), color='black')
    ax5.bar(x_axis_range, list(bd_score_final['bd_score'][st:ed]))
    ax5.spines['bottom'].set_linewidth(1.6)
    ax5.spines['left'].set_linewidth(1.6)
    ax5.spines['right'].set_linewidth(1.6)
    ax5.spines['top'].set_linewidth(1.6)
    ax5.tick_params(axis = 'y', length=5, width = 1.6)
    ax5.tick_params(axis = 'x', length=5, width = 1.6)
    ax5.set_ylabel('Bd score', FontSize = 5)
    if len(target_site) != 0:
        site_use = []
        for i in range(len(target_site)):
            if target_site[i] < st:
                pass
            elif target_site[i] <= end:
                site_use.append(target_site[i])
            else:
                break
        plt.vlines(np.array(site_use) - st, 0, 14)
    #plt.savefig(save_name, format = 'svg', transparent = True) 
    #plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

def draw_bd_score_combine_ms_pvalue_2_bd_score(mat_dense, Chr, st, ed, bd_cell_score_original, bd_cell_score_refined, df_bd_pvalue_result, peak_list, resolution, save_name, p_cut, target_site = [], bin_size = 10):
    
    plt.figure(figsize=(6,9.5))
    x_axis_range = range(len(bd_cell_score_original['bd_score'][st:ed]))
    
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
         
    ax1 = plt.subplot2grid((12, 7), (0, 0), rowspan=6,colspan=6)
    start = int(start_ / resolution)
    end = int(end_ / resolution)
    dense_matrix_part = mat_dense[start:end, start:end]
    #plt.figure(figsize=(10, 10))
    #plt.figure(figsize=(4, 4))
    #ax = sns.heatmap(data = dense_matrix_part, square=True, yticklabels=False,  xticklabels=False, cmap='gist_gray_r', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
    #img = ax1.imshow(dense_matrix_part, cmap='gist_gray_r', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
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
    ax1.set_title('TAD landscape of region:' + region_name, fontsize=12, pad = 15.0)

    cax = plt.subplot2grid((12, 7), (0, 6), rowspan=6,colspan=1)
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes("right", size="1.5%", pad= 0.2)
    #cbar = plt.colorbar(img, cax=cax, ticks=MultipleLocator(2.0), format="%.1f",orientation='vertical',extendfrac='auto',spacing='uniform')
    cbaxes = inset_axes(cax, width="30%", height="100%", loc=3) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])

    ax1_5 = plt.subplot2grid((12, 7), (6, 0), rowspan=1,colspan=6,sharex=ax1)
    ax1_5.plot(list(df_bd_pvalue_result[peak_list[-1]][st:ed]), color='black')
    ax1_5.bar(x_axis_range, list(df_bd_pvalue_result[peak_list[-1]][st:ed]), label=peak_list[-1], color='#B4E0F9')
    if p_cut != 0:
        ax1_5.hlines(p_cut, x_axis_range[0], x_axis_range[-1], color = 'black', linestyles = '--')
    ax1_5.spines['bottom'].set_linewidth(1.6)
    ax1_5.spines['left'].set_linewidth(1.6)
    ax1_5.spines['right'].set_linewidth(1.6)
    ax1_5.spines['top'].set_linewidth(1.6)
    ax1_5.tick_params(axis = 'y', length=5, width = 1.6)
    ax1_5.tick_params(axis = 'x', length=5, width = 1.6)
    ax1_5.set_ylabel(peak_list[-1], FontSize = 5)

    ax2 = plt.subplot2grid((12, 7), (7, 0), rowspan=1,colspan=6,sharex=ax1)
    ax2.plot(list(df_bd_pvalue_result[peak_list[-2]][st:ed]), color='black')
    ax2.bar(x_axis_range, list(df_bd_pvalue_result[peak_list[-2]][st:ed]), label=peak_list[-2], color='#B4E0F9')
    ax2.hlines(p_cut, x_axis_range[0], x_axis_range[-1], color = 'black', linestyles = '--')
    ax2.spines['bottom'].set_linewidth(1.6)
    ax2.spines['left'].set_linewidth(1.6)
    ax2.spines['right'].set_linewidth(1.6)
    ax2.spines['top'].set_linewidth(1.6)
    ax2.tick_params(axis = 'y', length=5, width = 1.6)
    ax2.tick_params(axis = 'x', length=5, width = 1.6)
    ax2.set_ylabel(peak_list[-2], FontSize = 5)

    ax3 = plt.subplot2grid((12, 7), (8, 0), rowspan=1,colspan=6,sharex=ax1)
    ax3.plot(list(df_bd_pvalue_result[peak_list[-3]][st:ed]), color='black')
    ax3.bar(x_axis_range, list(df_bd_pvalue_result[peak_list[-3]][st:ed]), label=peak_list[-3], color='#B4E0F9')
    if p_cut != 0:
        ax3.hlines(p_cut, x_axis_range[0], x_axis_range[-1], color = 'black', linestyles = '--')
    ax3.spines['bottom'].set_linewidth(1.6)
    ax3.spines['left'].set_linewidth(1.6)
    ax3.spines['right'].set_linewidth(1.6)
    ax3.spines['top'].set_linewidth(1.6)
    ax3.tick_params(axis = 'y', length=5, width = 1.6)
    ax3.tick_params(axis = 'x', length=5, width = 1.6)
    ax3.set_ylabel(peak_list[-3], FontSize = 5)

    ax4 = plt.subplot2grid((12, 7), (9, 0), rowspan=1,colspan=6,sharex=ax1)
    ax4.plot(list(df_bd_pvalue_result[peak_list[-4]][st:ed]), color='black')
    ax4.bar(x_axis_range, list(df_bd_pvalue_result[peak_list[-4]][st:ed]), label=peak_list[-4], color='#B4E0F9')
    if p_cut != 0:
        ax4.hlines(p_cut, x_axis_range[0], x_axis_range[-1], color = 'black', linestyles = '--')
    ax4.spines['bottom'].set_linewidth(1.6)
    ax4.spines['left'].set_linewidth(1.6)
    ax4.spines['right'].set_linewidth(1.6)
    ax4.spines['top'].set_linewidth(1.6)
    ax4.tick_params(axis = 'y', length=5, width = 1.6)
    ax4.tick_params(axis = 'x', length=5, width = 1.6)
    ax4.set_ylabel(peak_list[-4], FontSize = 5)

    ax5 = plt.subplot2grid((12, 7), (10, 0), rowspan=1,colspan=6,sharex=ax4)
    ax5.plot(list(bd_cell_score_original['bd_score'][st:ed]), color='black')
    ax5.bar(x_axis_range, list(bd_cell_score_original['bd_score'][st:ed]), color = '#D65F4D')
    ax5.spines['bottom'].set_linewidth(1.6)
    ax5.spines['left'].set_linewidth(1.6)
    ax5.spines['right'].set_linewidth(1.6)
    ax5.spines['top'].set_linewidth(1.6)
    ax5.tick_params(axis = 'y', length=5, width = 1.6)
    ax5.tick_params(axis = 'x', length=5, width = 1.6)
    ax5.set_ylabel('Bd score', FontSize = 5)
    
    ax6 = plt.subplot2grid((12, 7), (11, 0), rowspan=1,colspan=6,sharex=ax5)
    ax6.plot(list(bd_cell_score_refined['bd_score'][st:ed]), color='black')
    ax6.bar(x_axis_range, list(bd_cell_score_refined['bd_score'][st:ed]), color = '#4392c3')
    ax6.spines['bottom'].set_linewidth(1.6)
    ax6.spines['left'].set_linewidth(1.6)
    ax6.spines['right'].set_linewidth(1.6)
    ax6.spines['top'].set_linewidth(1.6)
    ax6.tick_params(axis = 'y', length=5, width = 1.6)
    ax6.tick_params(axis = 'x', length=5, width = 1.6)
    ax6.set_ylabel('Bd score', FontSize = 5)

    if len(target_site) != 0:
        site_use = []
        for i in range(len(target_site)):
            if target_site[i] < st:
                pass
            elif target_site[i] <= end:
                site_use.append(target_site[i])
            else:
                break
        plt.vlines(np.array(site_use) - st, 0, 14)

    plt.savefig(save_name, format = 'svg') 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)

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
    
def build_mat_target(bin1, bin2, mat_use, resolution, cut_dist, expand_fold = 1):
    '''
    Build local normalization matrix, the normalized matrix will fill the original position of zero-matrix.
    Thus, the splice of matrix won't change.
    '''
    domain_l = bin2 - bin1 + 1
    mat_target = np.zeros([len(mat_use), len(mat_use)])
    
    bin_st = int(np.max([0, bin1 - expand_fold * domain_l]))
    bin_ed = int(np.min([len(mat_use), bin2 + expand_fold * domain_l + 1]))
    mat_use_target = mat_use[bin_st : bin_ed, bin_st:bin_ed] 

    norm_type = 'z-score'
    mat_target_norm_z = dist_normalize_matrix(mat_use_target, resolution, norm_type, cut_dist)
    mat_target[bin_st : bin_ed, bin_st:bin_ed] =  mat_target_norm_z
    return mat_target, mat_target_norm_z

def get_mat_shuffle(mat_use, resolution, cut_dist = 5000000):
    mat_shuff = copy.deepcopy(mat_use)
    mat_shuff = np.zeros([len(mat_shuff), len(mat_shuff)])
    cut = int(cut_dist / resolution)
    for i in range(cut):
        diag_value = np.diag(mat_use, i)
        diag_shuff = list(diag_value)
        random.shuffle(diag_shuff)
        mat_shuff += np.diag(diag_shuff, k = i)
    mat_shuff = mat_shuff + mat_shuff.T - np.diag(np.diag(mat_shuff, 0), k = 0)   
    return mat_shuff

def get_multi_scale_bd_insulate_pvalue(tad_seperation_score, mat_use, filter_size, window_list, resolution):    
    '''
    Input: window list used to calculate pavlue of interaction of cross and intra region for each bin.
           filter_size: fixed size to filter bins in the head or tail of chromosome.
    Output: multi-window-size pvalue for each bin. -1 for head or tail; 1 for sparse region.
    '''
    filter_size = int(filter_size / resolution)
    df_pvalue_result = pd.DataFrame()
    for square_size in window_list:
        scale_label = str(int(square_size / 1000)) + 'kb-window'
        print('Dealing with ' + scale_label)
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
    df_pvalue_result['bd_score'] = tad_seperation_score['bd_score']
    return df_pvalue_result

def get_multi_scale_bd_insulate(tad_seperation_score, mat_use, filter_size, window_list, resolution):    
    '''
    Input: window list used to calculate pavlue of interaction of cross and intra region for each bin.
           filter_size: fixed size to filter bins in the head or tail of chromosome.
    Output: multi-window-size CI-value for each bin. 
    '''
    filter_size = int(filter_size / resolution)
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
    df_Insvalue_result['bd_score'] = tad_seperation_score['bd_score']
    return df_Insvalue_result

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


### Three kinds of operation   

### add   
def add_bd_according_to_pvalue(bd_score_cell, df_bd_insul_pvalue, Chr, window_list, p_cut):
    num = 0
    w_min = str(int(window_list[0] / 1000)) + 'kb-window'
    bd_score_cell_add = copy.deepcopy(bd_score_cell)
    for i in range(len(bd_score_cell)):
        bd_score = bd_score_cell['bd_score'][i]
        bd_pvalue = df_bd_insul_pvalue[w_min][i]
        if bd_pvalue == -1:
            continue
        if bd_score == 0 and bd_pvalue <= p_cut:
            bd_score_cell_add['bd_score'][i] = 1
            num += 1
    print('Add score for ' + str(num) + ' bin')
    df_boundary_region_add = deal_with_tad_seperation_score(bd_score_cell_add, Chr)
    return df_boundary_region_add, bd_score_cell_add
    
### split and shrink
def get_local_min_in_pvalue(df_bd_insul_pvalue, w_min, p_cut):   
    local_min_judge = [0]
    for i in range(1, len(df_bd_insul_pvalue)-1):
        if df_bd_insul_pvalue[w_min][i] == -1 or df_bd_insul_pvalue[w_min][i] <= p_cut:
            local_min_judge.append(0)
        else:
            p_h = df_bd_insul_pvalue[w_min][i]
            p_up = df_bd_insul_pvalue[w_min][i-1]
            p_down = df_bd_insul_pvalue[w_min][i+1]
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
            up2 = df_bd_insul_pvalue[w_min][i-2]
            up1 = df_bd_insul_pvalue[w_min][i-1]
            down1 = df_bd_insul_pvalue[w_min][i+1]
            down2 = df_bd_insul_pvalue[w_min][i+2]
            if up1 < up2 and up1 < down2:
                expand_l.append(i-1)
            if down1 < up2 and down1 < down2:
                expand_l.append(i+1)
    for i in range(len(expand_l)):
        local_min_judge[expand_l[i]] = 1
    '''
    st = 50
    ed = 120
    plt.figure(figsize=(5, 4.2))
    x_axis_range = range(len(df_bd_insul_pvalue[w_min][st:ed]))    
    plt.subplot(2, 1, 1)   
    plt.plot(np.array(df_bd_insul_pvalue[w_min][st:ed]), color='black')
    plt.bar(x_axis_range, np.array(df_bd_insul_pvalue[w_min][st:ed]))
    plt.subplot(2, 1, 2)   
    plt.bar(x_axis_range, np.array(local_min_judge[st:ed]))
    '''
    return local_min_judge
                

def adjust_bd_according_to_pvalue(bd_score_cell, df_bd_insul_pvalue, Chr, window_list, p_cut, high_score_cut = 5):
    w_min = str(int(window_list[0] / 1000)) + 'kb-window'    
    local_min_judge = get_local_min_in_pvalue(df_bd_insul_pvalue, w_min, p_cut)      
    bd_score_cell_adjust = copy.deepcopy(bd_score_cell)
    bd_score_cell_adjust[(bd_score_cell_adjust['bd_score'] < high_score_cut) & (df_bd_insul_pvalue[w_min] > p_cut)] = 0
    for i in range(len(local_min_judge)):
        if local_min_judge[i] == 0:
            continue
        else:
            score_hold = copy.deepcopy(bd_score_cell['bd_score'][i])
            if bd_score_cell_adjust['bd_score'][i] == 0:
                bd_score_cell_adjust['bd_score'][i] = score_hold
    df_boundary_region_adjust = deal_with_tad_seperation_score(bd_score_cell_adjust, Chr)
    return df_boundary_region_adjust, bd_score_cell_adjust
    
##### combine
def combine_boundary_region(df_boundary_region, bd_score_cell, Chr, combine_dist = 2):
    '''
    这里的combine用了非常严格的条件，只有相邻的两个region，中间隔了一个0值的bin才会合并。
    '''
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
    print('There are ' + str(num) + ' times combination.')
    df_boundary_region_combine = deal_with_tad_seperation_score(bd_score_cell_combine, Chr)
    return df_boundary_region_combine, bd_score_cell_combine 


def tad_seperation_landscape_build(bd_score_cell, mat_dense, cell_color_use, p_cut, window_list, resolution, filter_length, mat_norm_check = True, cut_dist = 12000000, norm_type = 'z-score'):
    result_record = {}
    if mat_norm_check == False:
        print('Normlaize the Hi-C matrix...')
        mat_hic = dist_normalize_matrix(mat_dense, resolution, norm_type, cut_dist)
    else:
        print('Normlaization done')
        mat_hic = mat_dense
    
    print('Calculate multi-scale seperation pvalue...')
    df_bd_insul_pvalue= get_multi_scale_bd_insulate_pvalue(bd_score_cell, mat_hic, filter_length, window_list, resolution)
    
    print('Dealing with TAD seperation landscape...')
    df_boundary_region = deal_with_tad_seperation_score(bd_score_cell, Chr)    
    sns.jointplot(x = 'length',y = 'ave_score', data = df_boundary_region, kind = 'kde', color = cell_color_use, space=0, cbar = False)
    plt.title('Original')
    result_record['Original'] = {}
    result_record['Original']['bd_region'] = df_boundary_region
    result_record['Original']['TAD_score'] = bd_score_cell
    
    print('Step1: Add')
    df_boundary_region_add, bd_score_cell_add = add_bd_according_to_pvalue(bd_score_cell, df_bd_insul_pvalue, Chr, window_list, p_cut = 0.05)
    sns.jointplot(x = 'length',y = 'ave_score', data = df_boundary_region_add, kind = 'kde', color = cell_color_use, space=0, cbar = False)
    plt.title('Add')
    result_record['Add'] = {}
    result_record['Add']['bd_region'] = df_boundary_region_add
    result_record['Add']['TAD_score'] = bd_score_cell_add
    
    print('Step2: Adjust')
    df_boundary_region_adjust, bd_score_cell_adjust = adjust_bd_according_to_pvalue(bd_score_cell_add, df_bd_insul_pvalue, Chr, window_list, p_cut = 0.05, high_score_cut = 5)
    sns.jointplot(x = 'length',y = 'ave_score', data = df_boundary_region_adjust, kind = 'kde', color = cell_color_use, space=0, cbar = False)
    plt.title('Adjust')
    result_record['Adjust'] = {}
    result_record['Adjust']['bd_region'] = df_boundary_region_adjust
    result_record['Adjust']['TAD_score'] = bd_score_cell_adjust
       
    print('Step3: Combine')
    df_boundary_region_combine, bd_score_cell_combine = combine_boundary_region(df_boundary_region_adjust, bd_score_cell_adjust, Chr, combine_dist = 2)
    sns.jointplot(x = 'length',y = 'ave_score', data = df_boundary_region_combine, kind = 'kde', color = cell_color_use, space=0, levels = 10, cbar = False)
    plt.title('Combine')
    result_record['Combine'] = {}
    result_record['Combine']['bd_region'] = df_boundary_region_combine
    result_record['Combine']['TAD_score'] = bd_score_cell_combine
    return result_record, df_bd_insul_pvalue

def get_best_window_size_for_pvalue_calculate(cell_type_list, boundary_score_cell_all, hic_mat_all_cell_replicate_znorm, window_list_multi, Chr = 'chr2', resolution = 50000, filter_length = 1000000):    
    df_pvalue_score_cor = pd.DataFrame()
    w_list_multi = []
    for w_use in window_list_multi:
       w_list_multi.append(str(int(w_use / 1000)) + 'kb-window')
    cor_result_all = []
    cell_enzyme_l = []
    cell_best_window = {}
    for cell_type in cell_type_list:
        print('This is ' + cell_type)
        if cell_type == 'GM12878':
            enzyme_list = ['DpnII', 'MboI']
        else:
            enzyme_list = ['MboI']
        for enzyme in enzyme_list:
            cell_enzyme_l.append(cell_type + '_' + enzyme)
            cor_result_cell = []
            bd_score_cell = boundary_score_cell_all[cell_type][enzyme]
            mat_dense_norm = hic_mat_all_cell_replicate_znorm[cell_type][enzyme]
            df_bd_insul_pvalue_multi = get_multi_scale_bd_insulate_pvalue(bd_score_cell, mat_dense_norm, filter_length, window_list_multi, resolution)

            for i in range(len(w_list_multi)):                
                w_use = w_list_multi[i]
                print('For ' + str(w_use))
                window_size = window_list_multi[i]
                w_cut = int(window_size / resolution)                
                cor_pvalue_score = scipy.stats.pearsonr(np.array(bd_score_cell['bd_score'].iloc[w_cut:len(bd_score_cell)-w_cut+1]), -np.array(df_bd_insul_pvalue_multi[w_use].iloc[w_cut:len(bd_score_cell)-w_cut+1]))[0]
                #cor_pvalue_score = scipy.stats.pearsonr(np.array(bd_score_cell['bd_score']), -np.array(df_bd_insul_pvalue_multi[w_use]))[0]
                print(cor_pvalue_score)
                cor_result_cell.append(cor_pvalue_score)            
            cell_best_window[cell_type + '_' + enzyme] = w_list_multi[np.argmax(cor_result_cell)]
            cor_result_all.append(cor_result_cell)
            print('\n')
    df_pvalue_score_cor = pd.DataFrame(cor_result_all)
    df_pvalue_score_cor.columns = w_list_multi
    df_pvalue_score_cor.index = cell_enzyme_l
    return df_pvalue_score_cor, cell_best_window


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

'''
norm_type = 'obs_exp'
hic_mat_all_cell_replicate_obex = {}
for cell_type in cell_type_list:
    print('This is ' + cell_type)
    hic_mat_all_cell_replicate_obex[cell_type] = {}
    if cell_type == 'GM12878':
        enzyme_list = ['DpnII', 'MboI']
    else:
        enzyme_list = ['MboI']
    for enzyme in enzyme_list:
        print('For ' + enzyme)
        mat_dense = hic_mat_all_cell_replicate[cell_type][enzyme]['iced']
        mat_norm_os = dist_normalize_matrix(mat_dense, resolution, norm_type, cut_dist)
        hic_mat_all_cell_replicate_obex[cell_type][enzyme] = mat_norm_os   
'''

save_add_build = 'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape/Build'
window_list_multi = [100000, 150000, 200000, 300000, 400000, 500000, 600000, 800000, 1000000, 2000000] 
df_pvalue_score_cor, cell_best_window = get_best_window_size_for_pvalue_calculate(cell_type_list, boundary_score_cell_all, hic_mat_all_cell_replicate_znorm, window_list_multi, Chr = 'chr2', resolution = 50000, filter_length = 1000000)

result_record_all = {}
for cell_type in cell_type_list:
    print('This is ' + cell_type)
    cell_color_use = cell_color[cell_type]
    if cell_type == 'GM12878':
        enzyme_list = ['DpnII', 'MboI']
    else:
        enzyme_list = ['MboI']
    for enzyme in enzyme_list:
        print('For ' + enzyme)
        
        window_best = cell_best_window[cell_type + '_' + enzyme]
        window_best_number = int(window_best.split('-')[0].split('kb')[0]) * 1000
        window_list = [window_best_number]
        window_list += [400000, 800000, 1000000]
        
        result_record_all[cell_type + '_' + enzyme] = {}
        bd_score_cell = boundary_score_cell_all[cell_type][enzyme]
        mat_dense = hic_mat_all_cell_replicate[cell_type][enzyme]['iced']
        mat_dense_norm = hic_mat_all_cell_replicate_znorm[cell_type][enzyme]
        
        result_record, df_bd_insul_pvalue = tad_seperation_landscape_build(bd_score_cell, mat_dense_norm, cell_color_use, p_cut, window_list, resolution, filter_length, mat_norm_check = True, cut_dist = 12000000, norm_type = 'z-score')
        result_record_all[cell_type + '_' + enzyme]['BD_region'] = result_record
        result_record_all[cell_type + '_' + enzyme]['pvalue'] = df_bd_insul_pvalue
        print('\n')

save_data(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape' + '/' + 'TAD_separation_landscape_for_all_cell_type.pkl', result_record_all)    
 
 
### draw case for refined boundary score profile and original boundary score profile 

cell_type = 'GM12878'
enzyme = 'MboI'
mat_dense = hic_mat_all_cell_replicate[cell_type][enzyme]['iced']
df_bd_insul_pvalue = result_record_all[cell_type + '_' + enzyme]['pvalue']
result_record = result_record_all[cell_type + '_' + enzyme]['BD_region']

bd_score_cell_original = result_record['Original']['TAD_score']
bd_score_cell_combine = result_record['Combine']['TAD_score']

## draw heatmap and multi-scale p-value and bd score
save_name = save_add_build + '/' + 'TAD_seperation_multi_pvalue_show_case_ori_ref_GM12878_MboI.svg'
#w_list = ['200kb-window', '400kb-window', '800kb-window', '1000kb-window']
w_list = list(df_bd_insul_pvalue.columns[:-1])
st = 750
ed = 820
target_site = []
draw_bd_score_combine_ms_pvalue_2_bd_score(mat_dense, Chr, st, ed, bd_score_cell_original, bd_score_cell_combine, df_bd_insul_pvalue, w_list, resolution, save_name, p_cut, target_site, bin_size = 10)
  
   
### Comparison of refined boundary score profile and original boundary score profile 

def compare_density_of_bd_region_avescore(df_boundary_region_original, df_boundary_region_combine, save_name = ''):
    bin_num = 11
    plt.figure(figsize= (6, 4))
    original_region_score = np.array(df_boundary_region_original['ave_score'])
    combine_region_score = np.array(df_boundary_region_combine['ave_score'])
    sns.distplot(np.array(df_boundary_region_original['ave_score']), bins=bin_num, hist = False, color = '#D65F4D', label = 'Original', kde_kws={"lw": 3, 'linestyle':'-', 'shade':True})
    sns.distplot(np.array(df_boundary_region_combine['ave_score']), bins=bin_num, hist = False, color = '#4392c3', label = 'Refined', kde_kws={"lw": 3, 'linestyle':'-', 'shade':True})
    p_value = scipy.stats.ks_2samp(original_region_score, combine_region_score)[1]
    print(p_value)
    plt.xlabel('Average region score', FontSize = 12)
    plt.ylabel('Density', FontSize = 12)
    #plt.xticks(range(0,15))
    plt.xticks(FontSize = 12)
    plt.yticks(FontSize = 12)
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

def compare_density_of_dist_between_bd_region(df_boundary_region_original, df_boundary_region_combine, save_name = ''):
    bin_num = 100
    original_bd_dist = np.array(df_boundary_region_original['up_dist'][1:]) - 1
    combine_bd_dist = np.array(df_boundary_region_combine['up_dist'][1:]) - 1
    plt.figure(figsize= (6, 4))
    #sns.distplot(original_bd_dist, bins=bin_num, hist = False, color = '#4392c3', label = 'original', kde_kws={"lw": 3, 'linestyle':'-', 'shade':False, 'cumulative':True})
    #sns.distplot(combine_bd_dist, bins=bin_num, hist = False, color = '#D65F4D', label = 'Combine', kde_kws={"lw": 3, 'linestyle':'-', 'shade':False, 'cumulative':True})
    plt.hist(original_bd_dist, bins = bin_num, density=True, histtype='step', cumulative=True, label = 'Original', color = '#D65F4D', linewidth=3)
    plt.hist(combine_bd_dist, bins = bin_num, density=True, histtype='step', cumulative=True, label = 'Refined', color = '#4392c3', linewidth=3)
    plt.xlabel('Distance between boundary region')
    plt.ylabel('Cumulative probability')
    #plt.xticks(range(0,10))
    plt.xlim([0,18])
    plt.legend(loc = 'lower right')
    p_value = scipy.stats.ks_2samp(original_bd_dist, combine_bd_dist)[1]
    print(p_value)
    plt.xticks(FontSize = 12)
    plt.yticks(FontSize = 12)
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

def compare_bd_region_score_boxplot(result_record_all, cell_type_list, save_name = ''):
    df_region_score = pd.DataFrame(columns = ['region_score', 'cell_type', 'type'])
    r_score_l = []
    cell_type_l = []
    type_l = []
    for cell_type in cell_type_list:
        if cell_type == 'GM12878':
            enzyme_list = ['DpnII', 'MboI']
        else:
            enzyme_list = ['MboI']
        for enzyme in enzyme_list:
            c_type = cell_type + '_' + enzyme
            print('For ' + c_type)
            result_record = result_record_all[cell_type + '_' + enzyme]['BD_region']
            
            df_boundary_region_original = result_record['Original']['bd_region']
            original_score = list(df_boundary_region_original['ave_score'])
            r_score_l += original_score
            cell_type_l += [c_type for i in range(len(df_boundary_region_original))]
            type_l += ['original' for i in range(len(df_boundary_region_original))]
            
            df_boundary_region_combine = result_record['Combine']['bd_region']
            combine_score = list(df_boundary_region_combine['ave_score'])
            r_score_l += combine_score
            cell_type_l += [c_type for i in range(len(df_boundary_region_combine))]
            type_l += ['processed' for i in range(len(df_boundary_region_combine))]

            p_value = scipy.stats.mannwhitneyu(original_score, combine_score)[1] 
            print(p_value)
            
    df_region_score['region_score'] = r_score_l
    df_region_score['cell_type'] = cell_type_l
    df_region_score['type'] = type_l
    plt.figure(figsize= (7, 5))
    sns.boxplot(x = 'cell_type', y = 'region_score', data = df_region_score, hue = 'type', fliersize=0, palette=['#D65F4D', '#4392c3'], saturation=1)
    plt.legend(loc = 'upper left', fontsize = 12)
    plt.xlabel('cell type',  FontSize = 0)
    plt.ylabel('Average Score',  FontSize = 12)
    plt.xticks(FontSize = 12, rotation = -30)
    plt.yticks(FontSize = 12)
    plt.ylim([0,15])
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

def get_domain_length_between_bd_region(df_boundary_region):
    domain_length_l = []
    for i in range(len(df_boundary_region) - 1):
        st = df_boundary_region['start'][i]
        ed = df_boundary_region['end'][i]
        st_n = df_boundary_region['start'][i+1]
        ed_n = df_boundary_region['end'][i+1]
        if (ed - st + 1) % 2 == 0:
            bd1 = int((ed+st)/2)+1
        else:
            bd1 = int((ed+st)/2)
        bd2 = int((ed_n+st_n)/2)
        d_l = bd2 - bd1 +1
        domain_length_l.append(d_l)
    return domain_length_l

def compare_domain_length_boxplot(result_record_all, cell_type_list, save_name = ''):
    df_region_score = pd.DataFrame(columns = ['domain_length', 'cell_type', 'type'])
    d_length_l = []
    cell_type_l = []
    type_l = []
    for cell_type in cell_type_list:
        if cell_type == 'GM12878':
            enzyme_list = ['DpnII', 'MboI']
        else:
            enzyme_list = ['MboI']
        for enzyme in enzyme_list:
            c_type = cell_type + '_' + enzyme
            print('For ' + c_type)
            result_record = result_record_all[cell_type + '_' + enzyme]['BD_region']
            
            df_boundary_region_original = result_record['Original']['bd_region']           
            domain_length_original = get_domain_length_between_bd_region(df_boundary_region_original)                        
            d_length_l += domain_length_original
            cell_type_l += [c_type for i in range(len(domain_length_original))]
            type_l += ['Original' for i in range(len(domain_length_original))]

            df_boundary_region_combine = result_record['Combine']['bd_region']           
            domain_length_combine = get_domain_length_between_bd_region(df_boundary_region_combine)                        
            d_length_l += domain_length_combine
            cell_type_l += [c_type for i in range(len(domain_length_combine))]
            type_l += ['Refined' for i in range(len(domain_length_combine))]

            p_value = scipy.stats.mannwhitneyu(domain_length_original, domain_length_combine)[1] 
            print(p_value) 
            #print(np.mean(domain_length_original))
            #print(np.mean(domain_length_combine))
    df_region_score['domain_length'] = d_length_l
    df_region_score['cell_type'] = cell_type_l
    df_region_score['type'] = type_l
    plt.figure(figsize= (7, 5))
    sns.boxplot(x = 'cell_type', y = 'domain_length', data = df_region_score, hue = 'type', fliersize=0, palette=['#D65F4D', '#4392c3'], saturation=1)
    plt.legend(loc = 'upper left', fontsize = 12)
    plt.xlabel('cell type',  FontSize = 0)
    plt.ylabel('Distance between center of adjacent boundary region ',  FontSize = 12)
    plt.xticks(FontSize = 12, rotation = -30)
    plt.yticks([3,6,9,12,15,18,21], [3,6,9,12,15,18,21], FontSize = 12)
    plt.ylim([2,20])
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
    return df_region_score

def compare_bd_region_distance_boxplot(result_record_all, cell_type_list, save_name = ''):
    df_region_score = pd.DataFrame(columns = ['domain_length', 'cell_type', 'type'])
    d_length_l = []
    cell_type_l = []
    type_l = []
    for cell_type in cell_type_list:
        if cell_type == 'GM12878':
            enzyme_list = ['DpnII', 'MboI']
        else:
            enzyme_list = ['MboI']
        for enzyme in enzyme_list:
            c_type = cell_type + '_' + enzyme
            print('For ' + c_type)
            result_record = result_record_all[cell_type + '_' + enzyme]['BD_region']
            
            df_boundary_region_original = result_record['Original']['bd_region']           
            domain_length_original = np.array((df_boundary_region_original['up_dist'][1:])) - 1                        
            d_length_l += list(domain_length_original)
            cell_type_l += [c_type for i in range(len(domain_length_original))]
            type_l += ['Original' for i in range(len(domain_length_original))]

            df_boundary_region_combine = result_record['Combine']['bd_region']           
            domain_length_combine = np.array((df_boundary_region_combine['up_dist'][1:])) - 1                       
            d_length_l += list(domain_length_combine)
            cell_type_l += [c_type for i in range(len(domain_length_combine))]
            type_l += ['Refined' for i in range(len(domain_length_combine))]

            p_value = scipy.stats.mannwhitneyu(domain_length_original, domain_length_combine)[1] 
            print(p_value)       
    df_region_score['domain_length'] = d_length_l
    df_region_score['cell_type'] = cell_type_l
    df_region_score['type'] = type_l
    plt.figure(figsize= (7, 5))
    sns.boxplot(x = 'cell_type', y = 'domain_length', data = df_region_score, hue = 'type', fliersize=0, palette=['#D65F4D', '#4392c3'], saturation=1)
    plt.legend(loc = 'upper left', fontsize = 12)
    plt.xlabel('cell type',  FontSize = 0)
    plt.ylabel('Distance between adjacent boundary region',  FontSize = 12)
    plt.xticks(FontSize = 12, rotation = -30)
    plt.yticks(FontSize = 12)
    plt.ylim([0,13])
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

####  GM12878 DpnII as example to show original and refined difference about score and BD region distance
cell_type = 'GM12878'
enzyme = 'MboI'
result_record = result_record_all[cell_type + '_' + enzyme]['BD_region']
df_boundary_region_original = result_record['Original']['bd_region']
df_boundary_region_combine = result_record['Combine']['bd_region']

save_name = save_add_build + '/' + cell_type + '_' + enzyme + '_ave_score_ori_ref.svg'
compare_density_of_bd_region_avescore(df_boundary_region_original, df_boundary_region_combine, save_name)

save_name = save_add_build + '/' + cell_type + '_' + enzyme + '_dist_bd_region_ref.svg'
compare_density_of_dist_between_bd_region(df_boundary_region_original, df_boundary_region_combine, save_name)


## all cell type compare
save_name = save_add_build + '/' + 'All_celltype_ave_score_ori_ref.svg'
compare_bd_region_score_boxplot(result_record_all, cell_type_list, save_name)

save_name = save_add_build + '/' + 'All_celltype_dist_bd_center_ori_ref.svg'
df_domain_length_all_cell_ori_ref = compare_domain_length_boxplot(result_record_all, cell_type_list, save_name)

save_name = save_add_build + '/' + 'All_celltype_dist_bd_region_ori_ref.svg'
compare_bd_region_distance_boxplot(result_record_all, cell_type_list, save_name)

  
###### show the domain length of single method
def get_domain_region(df_method_bd_score_st_ed, method, Chr):
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
    bd_score_new = copy.deepcopy(df_method_bd_score_st_ed[method])
    #tad_seperation_score_new[tad_seperation_score_new['bd_score'] <= 1]=0
    region_on = False
    for i in range(len(bd_score_new)):
        bd_score = bd_score_new[i]
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

def draw_median_domain_length_compare_ori_refine_method(cell_type_list, df_domain_length_all_cell_ori_ref, method_score_cell_all, method_list, save_name = ''):
    df_median_domain_l = pd.DataFrame(columns = ['median_domain_length', 'cell_type', 'method'])
    median_domain_length_cell_all = []
    cell_type_l = []
    method_l = []
    c_type_l = []
    print('Dealing with all cell type for median domain length for every method...')
    for cell_type in cell_type_list:
        if cell_type == 'GM12878':
            enzyme_list = ['DpnII', 'MboI']
        else:
            enzyme_list = ['MboI']
        for enzyme in enzyme_list:
            c_type = cell_type + '_' + enzyme
            print('For ' + c_type)
            c_type_l.append(c_type)
            df_method_bd_score_st_ed = method_score_cell_all[cell_type][enzyme]
            median_domain_length_m = []
            for method in method_list:
                df_bd_region_m = get_domain_region(df_method_bd_score_st_ed, method, Chr = 'chr2')
                #df_bd_region_m = df_bd_region_m[df_bd_region_m['up_dist'] <= 199]
                #df_bd_region_m = df_bd_region_m.reset_index(drop = True)
                median_domain_length_m.append(np.median(np.array(df_bd_region_m['up_dist'][1:]) + 1))
            median_domain_length_cell_all += median_domain_length_m    
            cell_type_l += [c_type for i in range(len(median_domain_length_m))]
            method_l += method_list
    df_median_domain_l['median_domain_length'] = median_domain_length_cell_all
    df_median_domain_l['cell_type'] = cell_type_l
    df_median_domain_l['method'] = method_l
             
    plt.figure(figsize= (6, 4))
    sns.barplot(x="cell_type", y="median_domain_length", data = df_median_domain_l, 
                capsize = 0.2, saturation = 8,             
            errcolor = 'black', errwidth = 1.5, 
            ci = 'sd', edgecolor='black', palette = ['#B4B4B4'])
    
    sns.swarmplot(x="cell_type", y="median_domain_length", data = df_median_domain_l, palette = ['black'], dodge=False, size = 3.5)

    for j in range(len(c_type_l)):
        c_type = c_type_l[j]
        print('For ' + c_type)
        df_domain_length_cell_ori = df_domain_length_all_cell_ori_ref[(df_domain_length_all_cell_ori_ref['cell_type'] == c_type) & (df_domain_length_all_cell_ori_ref['type'] == 'Original')]
        df_domain_length_cell_ref = df_domain_length_all_cell_ori_ref[(df_domain_length_all_cell_ori_ref['cell_type'] == c_type) & (df_domain_length_all_cell_ori_ref['type'] == 'Refined')]
        width = 0.8
        #plt.text(x + width/2, 100, str(j), ha='center', va='bottom', fontsize = 9)
        if j == 0:
            plt.hlines(np.median(df_domain_length_cell_ori['domain_length'])-0.1, j-width/2, j+width/2, colors='#D65F4D', linestyles='solid', linewidth=3, label='Original', alpha = 0.8)   
        else:
            plt.hlines(np.median(df_domain_length_cell_ori['domain_length']), j-width/2, j+width/2, colors='#D65F4D', linestyles='solid', linewidth=3, label='Original', alpha = 0.8)   
        plt.hlines(np.median(df_domain_length_cell_ref['domain_length']), j-width/2, j+width/2, colors='#4392c3', linestyles='solid', linewidth=3, label='Refined', alpha = 0.8)   
        print(np.median(df_domain_length_cell_ori['domain_length']))
        print(np.median(df_domain_length_cell_ref['domain_length']))

    #plt.legend(loc = 'upper left', fontsize = 12)
    plt.xlabel('cell type',  FontSize = 0)
    plt.ylabel('Median domain length (#bins)',  FontSize = 12)
    plt.xticks(FontSize = 12, rotation = -30)
    plt.yticks(FontSize = 12)
    plt.ylim([0,40])
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

save_name = save_add_build + '/' + 'Median_domain_length_compare_method_ori_ref.svg'
draw_median_domain_length_compare_ori_refine_method(cell_type_list, df_domain_length_all_cell_ori_ref, method_score_cell_all, method_list, save_name)
















