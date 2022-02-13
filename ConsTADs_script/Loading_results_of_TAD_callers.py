# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 17:04:15 2022

@author: dcdang
"""
import os
import pandas as pd
import numpy as np
import copy
import pickle


def save_data(file, objects):
    save_file = open(file, 'wb')
    pickle.dump(objects, save_file, 2)


def read_save_data(file):
    read_file = open(file, 'rb')
    objects = pickle.load(read_file)
    read_file.close()
    return objects


def get_TAD_boundary_result_for_basic(method, df_result, target_chr, resolution, chr_size):
    df_result.columns = ['st', 'ed']
    df_TAD = pd.DataFrame(columns = ['chr', 'start', 'end', 'name', 'boundary_st', 'boundary_ed'])
    df_boundary = pd.DataFrame(columns = ['chr', 'start', 'end', 'TAD_name'])
    TAD_st_list = (df_result['st'] - 1) * resolution 
    TAD_ed_list = df_result['ed'] * resolution
    last = False
    if TAD_ed_list[len(TAD_ed_list)-1] > chr_size:
        hold = copy.deepcopy(TAD_ed_list[len(TAD_ed_list)-1])
        last = True
        TAD_ed_list[len(TAD_ed_list)-1] = chr_size
    df_TAD['start'] = TAD_st_list    
    df_TAD['end'] = TAD_ed_list 
    df_TAD['chr'] = [target_chr for i in range(len(df_TAD))]
    df_TAD = df_TAD.sort_values(by = ['chr', 'start', 'end'])
    df_TAD = df_TAD.reset_index(drop = True)
    ### filter wrong TAD
    df_TAD = df_TAD[(df_TAD['end'] - df_TAD['start']) > 2 * resolution]
    df_TAD = df_TAD.reset_index(drop = True)
    
    df_TAD['name'] = [method + '_' + target_chr + '_TAD_' + str(i) for i in range(len(df_TAD))]
    df_TAD['boundary_st'] = [target_chr + ':' + str(df_TAD['start'][i]) + '-' +  
          str(df_TAD['start'][i] + resolution) for i in range(len(df_TAD))]
    df_TAD['boundary_ed'] = [target_chr + ':' + str(df_TAD['end'][i] - resolution) + '-' +  
          str(df_TAD['end'][i]) for i in range(len(df_TAD))]
    if last == True:
        bd_l = list(df_TAD['boundary_ed'])
        bd_l[-1] = target_chr + ':' + str(hold - resolution) + '-' + str(chr_size)
        df_TAD['boundary_ed'] = bd_l
        #df_TAD['boundary_ed'].iloc[-1] = target_chr + ':' + str(hold - resolution) + '-' + str(chr_size)
    df_boundary['start'] = list(df_TAD['start']) +  list(df_TAD['end'] - resolution)
    if last == True:
        st_l = list(df_boundary['start'])
        st_l[-1] = hold - resolution 
        df_boundary['start'] = st_l
        #df_boundary['start'].iloc[-1] = hold - resolution    
    df_boundary['end'] = list(df_TAD['start'] + resolution) +  list(df_TAD['end'])
    df_boundary['chr'] = [target_chr for i in range(len(df_boundary))]
    df_boundary['TAD_name'] = list(df_TAD['name']) + list(df_TAD['name'])
    df_boundary = df_boundary.sort_values(by = ['chr','start', 'end'])
    df_boundary = df_boundary.reset_index(drop = True)
    return df_TAD, df_boundary


def get_low_level_domain_3DNet(df_result):
    judge_l = []
    for i in range(len(df_result)):
        st = df_result['st'][i]
        ed = df_result['ed'][i]
        if i == 0:
            judge_l.append('Yes')
            hold_ind = i
            hold_st = st
            hold_ed = ed
            continue
        if hold_st <= st < hold_ed and hold_ed < ed:
            judge_l.append('No')
        elif hold_st <= st < hold_ed and hold_st < ed <= hold_ed:
            judge_l[hold_ind] = 'No'
            judge_l.append('Yes')
            hold_ind = i
            hold_st = st
            hold_ed = ed
        elif st >= hold_ed:
            judge_l.append('Yes')
            hold_ind = i
            hold_st = st
            hold_ed = ed            
    df_result_part = copy.deepcopy(df_result)
    df_result_part['judge'] = judge_l
    df_result_part = df_result_part[df_result_part['judge'] == 'Yes']
    df_result_part = df_result_part.reset_index(drop = True)
    return df_result_part
        

def get_TAD_boundary_result_for_3DNetMod(method, df_result, target_chr, resolution, chr_size, part = True):
    df_result = df_result.sort_values(by = [1,2])
    df_result = df_result.reset_index(drop = True)
    df_result[1] = np.array(df_result[1] / resolution).astype('int32')
    df_result[2] = np.array(df_result[2] / resolution).astype('int32')    
    df_result.columns = ['chr', 'st', 'ed']
    if part == True:
        df_result = get_low_level_domain_3DNet(df_result)    
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    df_result_new['st'] = df_result['st'] + 1
    df_result_new['ed'] = df_result['ed']
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary
  
      
def get_TAD_boundary_result_for_CaTCH(method, df_result, target_chr, resolution, chr_size):
    df_result.columns = ['index', 'chr', 'RI', 'start', 'end', 'IS']
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    df_result_new['st'] = copy.deepcopy(df_result['start'])
    df_result_new['ed'] = copy.deepcopy(df_result['end'])
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary


def get_TAD_boundary_result_for_CHDF(method, df_result, target_chr, resolution, chr_size):
    df_result.columns = ['start', 'end']
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    df_result_new['st'] = copy.deepcopy(df_result['start'])
    df_result_new['ed'] = copy.deepcopy(df_result['end'])
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary


def get_TAD_boundary_result_for_ClusterTAD(method, df_result, target_chr, resolution, chr_size):
    df_result.columns = ['start', 'st_cor', 'end', 'ed_cor']
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    df_result_new['st'] = df_result['start'] + 1
    df_result_new['ed'] = df_result['end']
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary


def get_TAD_boundary_result_for_deDoc(method, df_result, target_chr, resolution, chr_size):
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    st_list = []
    ed_list = []
    for i in range(len(df_result)):
       st = int(df_result.iloc[i][0].split(' ')[0])
       ed = int(df_result.iloc[i][0].split(' ')[-1])
       st_list.append(st)
       ed_list.append(ed)
    df_result_new['st'] = st_list   
    df_result_new['ed'] = ed_list     
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary


def get_TAD_boundary_result_for_DI(method, df_result, target_chr, resolution, chr_size):
    df_result.columns = ['chr', 'start', 'end']
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    st_ser= df_result['start'] / resolution
    ed_ser= df_result['end'] / resolution
    st_ser = st_ser.astype(np.int32)
    ed_ser = ed_ser.astype(np.int32)
    # DI result is 0-based, but end need not add 1, becaues the input is loc,not index.
    df_result_new['st'] = st_ser + 1
    df_result_new['ed'] = ed_ser
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary


def get_TAD_boundary_result_for_GMAP(method, df_result, target_chr, resolution, chr_size):
    df_result.columns = ['start', 'end']
    df_result = df_result.sort_values(by = ['start', 'end'])
    df_result = df_result.reset_index(drop = True)
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    df_result_new['st'] = df_result['start'] - int(resolution / 2)
    df_result_new['ed'] = df_result['end'] + int(resolution / 2)    
    st_ser= df_result_new['st'] / resolution
    ed_ser= df_result_new['ed'] / resolution
    st_ser = st_ser.astype(np.int32)
    ed_ser = ed_ser.astype(np.int32)    
    df_result_new['st'] = st_ser + 1
    df_result_new['ed'] = ed_ser - 1
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary


def get_TAD_boundary_result_for_HiCDB(method, df_result, target_chr, resolution, chr_size):
    df_result.columns = ['chr', 'st_bd', 'ed_bd', 'pvalue', 'CI']
    df_result = df_result.sort_values(by = ['st_bd', 'ed_bd'])
    df_result = df_result.reset_index(drop = True)    
    st_l = []
    ed_l = []
    for i in range(len(df_result)-1):
        st = int(df_result['ed_bd'][i] / resolution)
        ed = int(df_result['st_bd'][i+1] / resolution)
        st_l.append(st)
        ed_l.append(ed)
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    df_result_new['st'] = st_l
    df_result_new['ed'] = ed_l
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary


def get_TAD_boundary_result_for_Hiseg(method, df_result, target_chr, resolution, chr_size):
    df_result.columns = ['segment']
    df_result = df_result.reset_index(drop = True)
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    st_list = []
    ed_list = []
    # Hiseg is R method, 1-based
    for i in range(len(df_result) - 1):
        if i != len(df_result) - 2:
            st_list.append(df_result['segment'][i])
            ed_list.append(df_result['segment'][i+1]-1)
        else:
            st_list.append(df_result['segment'][i])
            ed_list.append(df_result['segment'][i+1])           
    df_result_new['st'] = st_list
    df_result_new['ed'] = ed_list
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary
 
    
def get_TAD_boundary_result_for_HiTAD(method, df_result, target_chr, resolution, chr_size, part = True):   
    df_result.columns = ['chr', 'st', 'ed', 'TAD_level']
    df_result = df_result.sort_values(by = [ 'st', 'ed'])
    df_result = df_result.reset_index(drop = True)
    if part == True:
        df_result_part = get_low_level_domain_3DNet(df_result)    
        df_result = df_result_part
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    st_ser= df_result['st'] / resolution
    ed_ser= df_result['ed'] / resolution
    st_ser = st_ser.astype(np.int32)
    ed_ser = ed_ser.astype(np.int32)
    # DI result is 0-based
    df_result_new['st'] = st_ser + 1
    df_result_new['ed'] = ed_ser
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    df_TAD['TAD_level'] = df_result['TAD_level']
    return df_TAD, df_boundary


def get_TAD_boundary_result_for_ICFinder(method, df_result, target_chr, resolution, chr_size):
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result, target_chr, resolution, chr_size)
    return df_TAD, df_boundary

    
def get_TAD_boundary_result_for_IS_old(method, df_result, target_chr, resolution, chr_size):
    boundary_list = df_result['header.1']
    segment_list = []
    for i in range(len(boundary_list)):
        end = int(boundary_list[i].split('-')[-1]) - 1
        segment_list.append(int(end / resolution))
        df_result_new = pd.DataFrame(segment_list)
    df_TAD, df_boundary = get_TAD_boundary_result_for_Hiseg(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary


def get_TAD_boundary_result_for_IS(method, df_result, target_chr, resolution, chr_size):
    df_result.columns = ['chr', 'range_st', 'range_ed', 'header', 'IS_strength']
    boundary_list = df_result['header']
    segment_list = []
    for i in range(len(boundary_list)):
        end = int(boundary_list[i].split('-')[-1]) - 1
        segment_list.append(int(end / resolution))
        df_result_new = pd.DataFrame(segment_list)
    df_TAD, df_boundary = get_TAD_boundary_result_for_Hiseg(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary


def get_TAD_boundary_result_for_MSTD(method, df_result, target_chr, resolution, chr_size):
    df_result.columns = ['start', 'end', 'center']
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    df_result_new['st'] = df_result['start'] + 1
    df_result_new['ed'] = df_result['end'] + 1 - 1
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary

    
def get_TAD_boundary_result_for_OnTAD(method, df_result, target_chr, resolution, chr_size, part = True):   
    df_result.columns = ['st', 'ed', 'TAD_level', 'score1', 'score2']
    df_result = df_result.sort_values(by = [ 'st', 'ed'])
    df_result = df_result.reset_index(drop = True)
    if df_result['st'][0] == 1 and df_result['ed'][0] == np.ceil(chr_size / resolution):
       df_result = df_result.iloc[1:]
       df_result = df_result.reset_index(drop = True)
    if part == True:
        df_result_part = get_low_level_domain_3DNet(df_result)    
        df_result = df_result_part
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    df_result_new['st'] = df_result['st']
    df_result_new['ed'] = df_result['ed']-1
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    df_TAD['TAD_level'] = df_result['TAD_level']
    return df_TAD, df_boundary


def get_TAD_boundary_result_for_Spectral(method, df_result, target_chr, resolution, chr_size):
    df_result.columns = ['info']
    df_result_new = pd.DataFrame(columns = ['chr', 'start', 'end'])
    chr_list = [target_chr for i in range(len(df_result))]
    st_list = []
    ed_list = []
    for i in range(len(df_result)):
        target = df_result['info'][i]
        st = int(target.split('-')[0])
        ed = int(target.split('-')[-1])
        st_list.append(st)
        ed_list.append(ed)
    df_result_new['chr'] = chr_list
    df_result_new['start'] = st_list
    df_result_new['end'] = ed_list
    df_TAD, df_boundary = get_TAD_boundary_result_for_DI(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary


def get_TAD_boundary_result_for_TopDom(method, df_result, target_chr, resolution, chr_size):
    df_result_part = df_result[df_result['tag'] == 'domain']
    df_result_part = df_result_part.reset_index(drop = True)
    df_result_new = pd.DataFrame(columns = ['st', 'ed'])
    df_result_new['st'] = df_result_part['from.id']
    df_result_new['ed'] = df_result_part['to.id']    
    df_TAD, df_boundary = get_TAD_boundary_result_for_basic(method, df_result_new, target_chr, resolution, chr_size)
    return df_TAD, df_boundary


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

TAD_result_all_cell_type = {}

for cell_type in cell_type_list:
    print('This is ' + cell_type)
    TAD_result_all_cell_type[cell_type] = {}
    if cell_type == 'GM12878':
        enzyme_list = ['DpnII', 'MboI']
    else:
        enzyme_list = ['MboI']
    for enzyme in enzyme_list:
        print('Enzyme is ' + enzyme)
        tad_method_add = tad_result_add + '/' + 'Rao2014-' + cell_type + '-' + enzyme + '-allreps-filtered-50kb' + '/' + 'method_result'
        Files = os.listdir(tad_method_add)
        TAD_result_all_cell_type[cell_type][enzyme] = {}
        for method in method_list:
            print('Dealing with ' + method)
            file_name = method + '_Result.txt'
            file = tad_method_add + '/' + file_name           
            if method in ['GMAP', 'HiCseg']:
                df_result = pd.read_csv(file, sep = ' ', header = 0)
            #elif method in ['TADtree', 'TopDom', 'MSTD', 'InsulationScore']:
            elif method in ['TADtree', 'TopDom', 'MSTD', 'ClusterTAD']:
                df_result = pd.read_csv(file, sep = '\t', header = 0)
            elif method in ['IS']:
                df_result = pd.read_csv(file, sep = '\t', header = None, skiprows=1)
            elif method in ['CaTCH']:
                df_result = pd.read_csv(file, sep = ' ', header = None)
            else:         
                df_result = pd.read_csv(file, sep = '\t', header = None)
            TAD_result_all_cell_type[cell_type][enzyme][method] = {}
            if method == '3DNetMod':
                df_TAD, df_boundary = get_TAD_boundary_result_for_3DNetMod(method, df_result, target_chr, resolution, chr_size, part = True)
            if method == 'CaTCH':
                df_TAD, df_boundary = get_TAD_boundary_result_for_CaTCH(method, df_result, target_chr, resolution, chr_size)
            if method == 'CHDF':
                df_TAD, df_boundary = get_TAD_boundary_result_for_CHDF(method, df_result, target_chr, resolution, chr_size)
            if method == 'ClusterTAD':
                df_TAD, df_boundary = get_TAD_boundary_result_for_ClusterTAD(method, df_result, target_chr, resolution, chr_size)
            if method == 'deDoc':
                df_TAD, df_boundary = get_TAD_boundary_result_for_deDoc(method, df_result, target_chr, resolution, chr_size)
            if method == 'DI':
                df_TAD, df_boundary = get_TAD_boundary_result_for_DI(method, df_result, target_chr, resolution, chr_size)
            if method =='GMAP':
                df_TAD, df_boundary = get_TAD_boundary_result_for_GMAP(method, df_result, target_chr, resolution, chr_size)
            if method =='HiCDB':
                df_TAD, df_boundary = get_TAD_boundary_result_for_HiCDB(method, df_result, target_chr, resolution, chr_size)
            if method =='HiCseg':
                df_TAD, df_boundary = get_TAD_boundary_result_for_Hiseg(method, df_result, target_chr, resolution, chr_size)
            if method == 'HiTAD':
                df_TAD, df_boundary = get_TAD_boundary_result_for_HiTAD(method, df_result, target_chr, resolution, chr_size, part = True)
            if method =='ICFinder':
                df_TAD, df_boundary = get_TAD_boundary_result_for_ICFinder(method, df_result, target_chr, resolution, chr_size)
            if method =='IS':
                df_TAD, df_boundary = get_TAD_boundary_result_for_IS(method, df_result, target_chr, resolution, chr_size)
            if method =='MSTD':
                df_TAD, df_boundary = get_TAD_boundary_result_for_MSTD(method, df_result, target_chr, resolution, chr_size)
            if method == 'Spectral':
                 df_TAD, df_boundary = get_TAD_boundary_result_for_Spectral(method, df_result, target_chr, resolution, chr_size)
            if method =='OnTAD':
                df_TAD, df_boundary = get_TAD_boundary_result_for_OnTAD(method, df_result, target_chr, resolution, chr_size, part = True)
            if method =='TopDom':
                df_TAD, df_boundary = get_TAD_boundary_result_for_TopDom(method, df_result, target_chr, resolution, chr_size)
            
            TAD_result_all_cell_type[cell_type][enzyme][method]['TAD_domain'] = df_TAD
            TAD_result_all_cell_type[cell_type][enzyme][method]['TAD_boundary'] = df_boundary
        print('\n')
 
    
save_data(r'E:/Users/dcdang/TAD_intergate/final_run/TAD_seperation_landscape' + '/' + 'TAD_result_all_cell_type.pkl', TAD_result_all_cell_type)    
 






