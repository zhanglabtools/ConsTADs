# -*- coding: utf-8 -*-
"""
Created on Sat May 21 12:45:18 2022

@author: dcdang
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
import matplotlib
import copy
import matplotlib.colors as mcolors


def DrawBdScoreMultiScalePvalue(mat_dense, Chr, st, ed, bd_cell_score_original, bd_cell_score_refined, df_pvalue_multi, window_l, resolution, save_name, p_cut, target_site = [], fgsize = (12, 8), bin_size = 10):
    plt.figure(figsize=(fgsize[0], fgsize[1]))
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
    img = ax1.imshow(dense_matrix_part, cmap='coolwarm', vmin = np.percentile(dense_matrix_part, 10), 
                     vmax = np.percentile(dense_matrix_part, 90))
    
    ax1.set_xticks([])
    ax1.spines['bottom'].set_linewidth(0)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(0)
    ax1.spines['top'].set_linewidth(0)
    ax1.tick_params(axis = 'y', length=5, width = 1.6)
    ax1.tick_params(axis = 'x', length=5, width = 1.6)
    #plt.xticks(cord_list, x_ticks_l, fontsize = 0,  rotation = 90)
    plt.xticks(cord_list, ['' for k in range(len(cord_list))], fontsize = 0, rotation = 90)
    plt.yticks(cord_list, y_ticks_l, fontsize = 10)
    ax1.set_title('TAD landscape of region:' + region_name, fontsize=12, pad = 15.0)

    cax = plt.subplot2grid((12, 7), (0, 6), rowspan=6,colspan=1)
    cbaxes = inset_axes(cax, width="30%", height="94%", loc=3) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])
    
    for i in range(1, len(window_l)+1):
        wd = window_l[-i]
        ax1_5 = plt.subplot2grid((12, 7), (6 + i - 1, 0), rowspan=1,colspan=6, sharex=ax1)
        ax1_5.plot(list(df_pvalue_multi[wd][st:ed]), color='black')
        ax1_5.bar(x_axis_range, list(df_pvalue_multi[wd][st:ed]), label=wd, color='#B4E0F9')
        if p_cut != 0:
            ax1_5.hlines(p_cut, x_axis_range[0], x_axis_range[-1], color = 'black', linestyles = '--')
        ax1_5.spines['bottom'].set_linewidth(1.6)
        ax1_5.spines['left'].set_linewidth(1.6)
        ax1_5.spines['right'].set_linewidth(1.6)
        ax1_5.spines['top'].set_linewidth(1.6)
        ax1_5.tick_params(axis = 'y', length=5, width = 1.6)
        ax1_5.tick_params(axis = 'x', length=5, width = 1.6)
        if i == len(window_l):
            ax1_5.set_ylabel(wd.split('-')[0], fontsize = 10, color = 'red')
        else:
            ax1_5.set_ylabel(wd.split('-')[0], fontsize = 10)
        #ax1_5.set_xticks(cord_list, x_ticks_l, fontsize = 0, rotation = 90)
        #ax1_5.set_xticks(cord_list, ['' for k in range(len(cord_list))], fontsize = 0, rotation = 90)
        ax1_5.set_xticks(cord_list)
        ax1_5.set_xticklabels(labels=['' for k in range(len(cord_list))], fontsize=0, rotation=90)

        cord_list_y = [0, 1]
        ax1_5.set_yticks(cord_list_y)
        ax1_5.set_yticklabels(labels=['', 1], fontsize=8)

        plt.ylim([0, 1])
        if len(target_site) != 0:
            site_use = []
            for i in range(len(target_site)):
                if target_site[i] < st:
                    pass
                elif target_site[i] < end:
                    site_use.append(target_site[i])
                else:
                    break
            plt.vlines(np.array(site_use) - st, 0, 1, color = 'black')
    
    ax5 = plt.subplot2grid((12, 7), (6 + i + 1 - 1, 0), rowspan=1,colspan=6,sharex=ax1)
    ax5.plot(list(bd_cell_score_original['bd_score'][st:ed]), color='black')
    ax5.bar(x_axis_range, list(bd_cell_score_original['bd_score'][st:ed]), color = '#D65F4D')
    ax5.spines['bottom'].set_linewidth(1.6)
    ax5.spines['left'].set_linewidth(1.6)
    ax5.spines['right'].set_linewidth(1.6)
    ax5.spines['top'].set_linewidth(1.6)
    ax5.tick_params(axis = 'y', length=5, width = 1.6)
    ax5.tick_params(axis = 'x', length=5, width = 1.6)
    ax5.set_ylabel('Bd score', fontsize = 10)
    #ax5.set_xticks(cord_list, ['' for k in range(len(cord_list))], fontsize = 0, rotation = 90)
    ax5.set_xticks(cord_list)
    ax5.set_xticklabels(labels=['' for k in range(len(cord_list))], fontsize=0, rotation=90)

    ax6 = plt.subplot2grid((12, 7), (6 + i + 2 - 1, 0), rowspan=1,colspan=6,sharex=ax1)
    ax6.plot(list(bd_cell_score_refined['bd_score'][st:ed]), color='black')
    ax6.bar(x_axis_range, list(bd_cell_score_refined['bd_score'][st:ed]), color = '#4392c3')
    ax6.spines['bottom'].set_linewidth(1.6)
    ax6.spines['left'].set_linewidth(1.6)
    ax6.spines['right'].set_linewidth(1.6)
    ax6.spines['top'].set_linewidth(1.6)
    ax6.tick_params(axis = 'y', length=5, width = 1.6)
    ax6.tick_params(axis = 'x', length=5, width = 1.6)
    ax6.set_ylabel('TSL \n Bd score', fontsize = 10, color = 'red')
    
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['right'].set_linewidth(1.8)
    ax.spines['top'].set_linewidth(1.8)
    ax.tick_params(axis = 'y', length=4, width = 1.8)
    ax.tick_params(axis = 'x', length=4, width = 1.8)
    #ax.set_xticks(cord_list, x_ticks_l, fontsize = 10, rotation = -30)
    ax.set_xticks(cord_list)
    ax.set_xticklabels(labels = x_ticks_l, fontsize = 10, rotation = -30)

def DrawBdScoreMultiScalePvalue_old(mat_dense, Chr, st, ed, bd_cell_score_original, bd_cell_score_refined, df_bd_pvalue_result, peak_list, resolution, save_name, p_cut, target_site = [], fgsize = (12, 8), bin_size = 10):
    plt.figure(figsize=(fgsize[0], fgsize[1]))
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
    plt.xticks(cord_list, x_ticks_l, fontsize = 10)
    plt.yticks(cord_list, y_ticks_l, fontsize = 10)
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
    ax1_5.set_ylabel(peak_list[-1].split('-')[0], fontsize = 10)

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
    ax2.set_ylabel(peak_list[-2].split('-')[0], fontsize = 10)

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
    ax3.set_ylabel(peak_list[-3].split('-')[0], fontsize =10)

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
    ax4.set_ylabel(peak_list[-4].split('-')[0], fontsize = 10)

    ax5 = plt.subplot2grid((12, 7), (10, 0), rowspan=1,colspan=6,sharex=ax4)
    ax5.plot(list(bd_cell_score_original['bd_score'][st:ed]), color='black')
    ax5.bar(x_axis_range, list(bd_cell_score_original['bd_score'][st:ed]), color = '#D65F4D')
    ax5.spines['bottom'].set_linewidth(1.6)
    ax5.spines['left'].set_linewidth(1.6)
    ax5.spines['right'].set_linewidth(1.6)
    ax5.spines['top'].set_linewidth(1.6)
    ax5.tick_params(axis = 'y', length=5, width = 1.6)
    ax5.tick_params(axis = 'x', length=5, width = 1.6)
    ax5.set_ylabel('Bd score', fontsize = 10)
    
    ax6 = plt.subplot2grid((12, 7), (11, 0), rowspan=1,colspan=6,sharex=ax5)
    ax6.plot(list(bd_cell_score_refined['bd_score'][st:ed]), color='black')
    ax6.bar(x_axis_range, list(bd_cell_score_refined['bd_score'][st:ed]), color = '#4392c3')
    ax6.spines['bottom'].set_linewidth(1.6)
    ax6.spines['left'].set_linewidth(1.6)
    ax6.spines['right'].set_linewidth(1.6)
    ax6.spines['top'].set_linewidth(1.6)
    ax6.tick_params(axis = 'y', length=5, width = 1.6)
    ax6.tick_params(axis = 'x', length=5, width = 1.6)
    ax6.set_ylabel('Bd score', fontsize = 10)

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
    if save_name != '':
        plt.savefig(save_name, format = 'svg') 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)



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

 
def draw_tad_region_upper_half(st, ed, range_t, color, size_v, size_h):  
    if st < 0:
        plt.vlines(ed, 0, ed, colors=color, linestyles='solid', linewidths=size_v)
    elif ed > range_t:
        plt.hlines(st, st, range_t, colors=color, linestyles='solid', linewidths=size_v)
    else:## 画竖线
        #plt.vlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_v)
        plt.vlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_v)
        ## 画横线
        plt.hlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_h)
        #plt.hlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_h)

def draw_tad_region_lower_half(st, ed, range_t, color, size_v, size_h):  
    if st < 0:
        plt.hlines(ed, 0, ed, colors=color, linestyles='solid', linewidths=size_h)
    elif ed > range_t: 
        plt.vlines(st, st, range_t, colors=color, linestyles='solid', linewidths=size_v)
    else:
        ## 画竖线
        plt.vlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_v)
        #plt.vlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_v)
        ## 画横线
        #plt.hlines(st, st, ed, colors=color, linestyles='solid', linewidths=size_h)
        plt.hlines(ed, st, ed, colors=color, linestyles='solid', linewidths=size_h)

def get_bd_type_symbol(df_bd_region_type, bd_score_cell_combine, symbol_dic):
    bd_symbol = np.zeros(len(bd_score_cell_combine))
    for i in range(len(df_bd_region_type)):
        region = df_bd_region_type['region'][i]
        bd_type = df_bd_region_type['region_type_adjust'][i]
        symbol = symbol_dic[bd_type]
        for x in region:
            bd_symbol[x] = symbol
    return bd_symbol


def DrawConsTADsAndBdRegion(contact_map, Chr, st, ed, TAD_list, bd_score_cell_combine, bd_symbol, resolution, fgsize = (10, 10), save_name = '', bin_size = 10):
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
    
    plt.figure(figsize=(fgsize[0], fgsize[1]))     
    ax0 = plt.subplot2grid((8, 8), (0, 0), rowspan=6,colspan=6)
    dense_matrix_part = contact_map[st:ed+1, st:ed+1]
    #img = ax0.imshow(dense_matrix_part, cmap='seismic', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 91))
    img = ax0.imshow(dense_matrix_part, cmap='coolwarm', vmin = np.percentile(dense_matrix_part, 10), vmax = np.percentile(dense_matrix_part, 90))
    ax0.set_xticks([])
    #ax0.set_yticks([])
    ax0.spines['bottom'].set_linewidth(0)
    ax0.spines['left'].set_linewidth(1.6)
    ax0.spines['right'].set_linewidth(0)
    ax0.spines['top'].set_linewidth(0)
    ax0.tick_params(axis = 'y', length=5, width = 1.6)
    ax0.tick_params(axis = 'x', length=5, width = 1.6)
    plt.xticks(cord_list, x_ticks_l, fontsize = 10)
    plt.yticks(cord_list, y_ticks_l, fontsize = 10)
    ax0.set_title(region_name, fontsize=12, pad = 15.0)
    
    TAD_color = 'black'
    if len(TAD_list) != 0:
        for TAD in TAD_list:
            st_tad = TAD[0] - st
            ed_tad = TAD[1] - st
            #print(st_tad, ed_tad)
            draw_tad_region(st_tad, ed_tad, TAD_color, size_v=3, size_h=3)
            #draw_tad_region_upper_half(st_tad, ed_tad, TAD_color, size_v=3, size_h=3)

    cax = plt.subplot2grid((8, 8), (0, 6), rowspan=6,colspan=1)
    #divider = make_axes_locatable(cax)
    #cax = divider.append_axes("right", size="1.5%", pad= 0.2)
    #cbar = plt.colorbar(img, cax=cax, ticks=MultipleLocator(2.0), format="%.1f",orientation='vertical',extendfrac='auto',spacing='uniform')
    cbaxes = inset_axes(cax, width="30%", height="94%", loc=3) 
    plt.colorbar(img, cax = cbaxes, orientation='vertical')
    cax.spines['bottom'].set_linewidth(0)
    cax.spines['left'].set_linewidth(0)
    cax.spines['right'].set_linewidth(0)
    cax.spines['top'].set_linewidth(0)
    cax.tick_params(axis = 'y', length=0, width = 0)
    cax.tick_params(axis = 'x', length=0, width = 0)
    cax.set_xticks([])
    cax.set_yticks([])

    ax1 = plt.subplot2grid((8, 8), (6, 0), rowspan=1, colspan=6, sharex=ax0)    
    ax1.plot(x_axis_range, bd_score_cell_combine['bd_score'][st:ed], marker = '.', linewidth = 2, c = 'black')
    ax1.bar(x_axis_range, list(bd_score_cell_combine['bd_score'][st:ed]))
    plt.ylabel('Bd score')
    ax1.set_xticks([])
    #ax1.set_yticks([])
    ax1.spines['bottom'].set_linewidth(1.6)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(1.6)
    ax1.spines['top'].set_linewidth(1.6)
    ax1.tick_params(axis = 'y', length=5, width = 1.6)
    ax1.tick_params(axis = 'x', length=5, width = 1.6)

    ax2 = plt.subplot2grid((8, 8), (7, 0), rowspan=1,colspan=6, sharex=ax0)    
    bd_data = []
    cmap=['#B0B0B0','#459457','#D65F4D','#4392C3']
    my_cmap = ListedColormap(cmap)
    bounds=[0,0.9,1.9,2.9,3.9]
    norm = matplotlib.colors.BoundaryNorm(bounds, my_cmap.N)    
    for i in range(10):
        bd_data.append(bd_symbol[st:ed])
    #bd_data_expand = np.reshape(np.array(bd_data), (10, len(bd_data[0])))
    ax2.imshow(bd_data, cmap = my_cmap, norm = norm)
    plt.ylabel('Bd type')
    plt.yticks([1, 10], ['', ''])
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


def get_tad_list_in_target_ranges(st, ed, df_tad_use_1, pos_type = 'bin', resolution = 50000):
    if pos_type == 'cord':
        df_tad_use = copy.deepcopy(df_tad_use_1)
        df_tad_use['start'] = np.array(df_tad_use['start'] / resolution).astype(np.int32)
        df_tad_use['end'] = np.array(df_tad_use['end'] / resolution).astype(np.int32) - 1        
    else:
        df_tad_use = copy.deepcopy(df_tad_use_1)
    TAD_list = []
    for i in range(len(df_tad_use)):
        start = df_tad_use['start'][i]
        end = df_tad_use['end'][i]
        if start > st and end < ed:
            st_bin = start
            ed_bin = end
            TAD_list.append((st_bin, ed_bin))
        elif start < st and ( st < end <= ed):
            st_bin = start
            ed_bin = end
            TAD_list.append((st_bin, ed_bin))
        elif (ed > start >= st) and end > ed:
            st_bin = start
            ed_bin = end
            TAD_list.append((st_bin, ed_bin))
    return TAD_list


def matrix_part_max_norm(mat_region):
    vec_diag = np.diag(mat_region) 
    mat_diag = np.diag(vec_diag)
    mat_region -= mat_diag
    mat_region = mat_region / np.max(mat_region)
    return mat_region

def DrawPairWiseMapCompare(cell_1, cell_2, mat_dense1, mat_dense2, Chr, st, ed, bd_cell_score1, bd_cell_score2, TAD_list_1, TAD_list_2, resolution, figsize = (6,8), TAD_color_1 = 'black', TAD_color_2 = 'white', target_site = [], save_name = '', bin_size = 10): 
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
    #norm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    #plt.figure(figsize=(6,6))
    #plt.imshow(dense_matrix_combine, cmap = 'seismic', vmin = vmin, vmax =vmax, norm=norm) 
    #plt.imshow(dense_matrix_combine, cmap = 'coolwarm', vmin = vmin, vmax =vmax, norm=norm) 
    #plt.colorbar()

    plt.figure(figsize=(figsize[0], figsize[1]))
    ax1 = plt.subplot2grid((9, 7), (0, 0), rowspan=6,colspan=6)
    #img = ax1.imshow(dense_matrix_combine, cmap = 'coolwarm', vmin = vmin, vmax =vmax, norm=norm) 
    img = ax1.imshow(dense_matrix_combine, cmap = 'coolwarm', norm=norm) 
    ax1.set_xticks([])
    #ax1.set_yticks([])
    ax1.spines['bottom'].set_linewidth(0)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['right'].set_linewidth(0)
    ax1.spines['top'].set_linewidth(0)
    ax1.tick_params(axis = 'y', length=5, width = 1.6)
    ax1.tick_params(axis = 'x', length=5, width = 1.6)
    #ax1.set_xticks(cord_list, ['' for k in range(len(cord_list))], fontsize = 0, rotation = 90)
    ax1.set_xticks(cord_list)
    ax1.set_xticklabels(labels=['' for k in range(len(cord_list))], fontsize=0, rotation=90)

    plt.yticks(cord_list, y_ticks_l, fontsize = 10)
    ax1.set_title(cell_1 + ' vs ' + cell_2 + ' in ' + region_name, fontsize=12, pad = 15.0)

    range_t = ed - st - 1
    if len(TAD_list_1) != 0:
        for i in range(len(TAD_list_1)):
            TAD = TAD_list_1[i]
            st_tad = TAD[0] - st
            ed_tad = TAD[1] - st 
            #print(st_tad, ed_tad)
            draw_tad_region_upper_half(st_tad, ed_tad, range_t, TAD_color_1, size_v=5, size_h=5)
    if len(TAD_list_2) != 0:
        for i in range(len(TAD_list_2)):
            TAD = TAD_list_2[i]
            st_tad = TAD[0] - st
            ed_tad = TAD[1] - st 
            #print(st_tad, ed_tad)
            draw_tad_region_lower_half(st_tad, ed_tad, range_t, TAD_color_2, size_v=5, size_h=5)
            
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
    #ax2.set_xticks(cord_list, ['' for k in range(len(cord_list))], fontsize = 0, rotation = 90)
    ax2.set_xticks(cord_list)
    ax2.set_xticklabels(labels=['' for k in range(len(cord_list))], fontsize=0, rotation=90)

    ax2.set_ylabel(cell_1 + ' \n TSL', fontsize = 10)

    ax3 = plt.subplot2grid((9, 7), (7, 0), rowspan=1,colspan=6,sharex=ax1)
    ax3.plot(list(bd_cell_score2['bd_score'][st:ed]), color='black')
    ax3.bar(x_axis_range, list(bd_cell_score2['bd_score'][st:ed]), label='score2', color='#3D50C3')
    ax3.spines['bottom'].set_linewidth(1.6)
    ax3.spines['left'].set_linewidth(1.6)
    ax3.spines['right'].set_linewidth(1.6)
    ax3.spines['top'].set_linewidth(1.6)
    ax3.tick_params(axis = 'y', length=5, width = 1.6)
    ax3.tick_params(axis = 'x', length=5, width = 1.6)
    ax3.set_ylabel(cell_2 + ' \n TSL', fontsize = 10)
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
    
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['right'].set_linewidth(1.8)
    ax.spines['top'].set_linewidth(1.8)
    ax.tick_params(axis = 'y', length=4, width = 1.8)
    ax.tick_params(axis = 'x', length=4, width = 1.8)
    #ax.set_xticks(cord_list, x_ticks_l, fontsize = 10, rotation = -30)
    ax.set_xticks(cord_list)
    ax.set_xticklabels(labels=x_ticks_l, fontsize=10, rotation=-0)

    if save_name != '':
        plt.savefig(save_name, format = 'svg') 
    plt.show()
    #fig = plt.gcf() #获取当前figure
    #plt.close(fig)



























