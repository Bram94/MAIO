# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:33:36 2018

@author: bramv
"""
import numpy as np
import matplotlib.pyplot as plt
import calendar

import calculate_geostrophic_wind as gw
import read_cabauw_data as r
import settings as s

for i in range(8,9):
    s.month = format(i, '02d')
    s.n_days = calendar.monthrange(int(s.year), int(s.month))[1]
    gw_data = gw.calculate_geostrophic_wind()
        
    data = r.read_and_process_cabauw_data()    
    
        
    #%%
    figure_numbers_pos = [-0.125, 1.04]
    
    fig, ax = plt.subplots(int(np.ceil(s.n_days/5)),5, figsize = (20,20))
    plot_hours = np.array([0, 6, 12, 18])
    colors = ['blue', 'red', 'green', 'yellow']
    
    handles_windprofile = []
    def plot_windprofiles(ax, j):
        ax.set_aspect('equal')
        for i in range(len(plot_hours)):
            hour = plot_hours[i]
            time_index = np.argmin(np.abs(data.hours - hour))
            
            u_j, v_j = data.u[j, time_index, :-1], data.v[j, time_index, :-1] #Exclude the last element as it is np.nan
            u_g = gw_data.V_g_interpolated[j, time_index, 0]; v_g = gw_data.V_g_interpolated[j, time_index, 1]
            ax.plot(u_j, v_j, color = colors[i], marker = 'o', markersize = 3)
            ax.plot(u_g, v_g, color = colors[i], marker = 'o', markersize = 5)
            
            handles_windprofile.append(ax.plot(u_j, v_j, color = colors[i], linestyle = '-')[0])
            if i == 0:
                u_min = u_j.min(); u_max = u_j.max()
                v_min = v_j.min(); v_max = v_j.max()
            else:
                u_min = np.min([u_min, u_g, u_j.min()]); u_max = np.max([u_max, u_g, u_j.max()])
                v_min = np.min([v_min, v_g, v_j.min()]); v_max = np.max([v_max, v_g, v_j.max()])            
                
            for k in range(len(u_j)):
                if k in (0, len(u_j) - 1):
                    ax.text(u_j[k], v_j[k], str(int(data.z[k])))
                    
        ax.text(figure_numbers_pos[0], figure_numbers_pos[1], str(j+1)+')', transform=ax.transAxes, fontsize = 15)
    
        max_radius = int(np.ceil(np.max(np.abs([u_min, u_max, v_min, v_max]))))
        dr = int(max_radius/3)
        for i in np.arange(0.0, max_radius+dr, dr):
            ax.plot(i * np.sin(np.linspace(0, 2*np.pi, 50)), i * np.cos(np.linspace(0, 2*np.pi, 50)), color = 'black', linewidth = 0.5)
        
        do = 1
        x_min = np.min([int(np.floor(u_min/do)*do)-do, -do]); y_min = np.min([int(np.floor(v_min/do)*do)-do, -do])
        x_max = np.max([int(np.ceil(u_max/do)*do)+do, do]); y_max = np.max([int(np.ceil(v_max/do)*do)+do, do])
        x_range = x_max-x_min; y_range = y_max-y_min
        if x_range>y_range: 
            y_min -= (x_range-y_range)/2; y_max += (x_range-y_range)/2
        else:
            x_min -= (y_range-x_range)/2; x_max += (y_range-x_range)/2
    
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    
    for j in range(len(ax.flat)):
        try:
            plot_windprofiles(ax.flat[j],j)
            if j == 0:
                ax.flat[j].set_xlabel('u (m/s)'); ax.flat[j].set_ylabel('v (m/s)')
        except Exception: continue #Will occur when j >= n_days
    plt.figlegend(handles_windprofile, [format(j, '02d')+'z' for j in plot_hours], loc = [0.37,0.05], ncol = 4, labelspacing=0., fontsize = 12 )
    plt.figlegend(handles_windprofile, [format(j, '02d')+'z' for j in plot_hours], loc = [0.915,0.5], ncol = 1, labelspacing=0., fontsize = 12 )
    plt.savefig(s.imgs_path+'_wind_'+s.year+s.month+'.jpg', dpi = 120, bbox_inches = 'tight')
    plt.show()
    
    #%%
    fig, ax = plt.subplots(int(np.ceil(s.n_days/5)),5, figsize = (20,20))
    plot_heights = [10, 80, 200]
    colors = ['blue', 'red', 'green', 'yellow']
    
    handles_windcycle = []
    def plot_windcycles(ax, j):
        ax.set_aspect('equal')
        for i in range(len(plot_heights)):
            height = plot_heights[i]
            z_index = np.argmin(np.abs(data.z - height))
            
            u_j, v_j = data.u[j, :, z_index], data.v[j, :, z_index]
            ax.plot(u_j, v_j, color = colors[i], marker = 'o', markersize = 1.5)
            handles_windcycle.append(ax.plot(u_j, v_j, color = colors[i], linestyle = '-', linewidth = 0.75)[0])
            
            if i == 0:
                u_min = u_j.min(); u_max = u_j.max()
                v_min = v_j.min(); v_max = v_j.max()
            else:
                u_min = np.min([u_min, u_j.min()]); u_max = np.max([u_max, u_j.max()])
                v_min = np.min([v_min, v_j.min()]); v_max = np.max([v_max, v_j.max()])  
                
            for k in (0, -1):
                ax.text(u_j[k], v_j[k], 's' if k == 0 else 'e', fontsize = 12)
    
        u_g = gw_data.V_g_interpolated[j, :, 0]; v_g = gw_data.V_g_interpolated[j, :, 1]    
        ax.plot(u_g, v_g, color = 'black', marker = 'o', markersize = 1.5)
        handles_windcycle.append(ax.plot(u_g, v_g, color = 'black', linestyle = '-', linewidth = 0.75)[0])
        for k in (0, -1):
            ax.text(u_g[k], v_g[k], 's' if k == 0 else 'e', fontsize = 12)
        
        u_min = np.min([u_min, u_g.min()]); u_max = np.max([u_max, u_g.max()])
        v_min = np.min([v_min, v_g.min()]); v_max = np.max([v_max, v_g.max()])  
        
        ax.text(figure_numbers_pos[0], figure_numbers_pos[1], str(j+1)+')', transform=ax.transAxes, fontsize = 15)
    
        max_radius = int(np.ceil(np.max(np.abs([u_min, u_max, v_min, v_max]))))
        dr = int(max_radius/3)
        for i in np.arange(0.0, max_radius+dr, dr):
            ax.plot(i * np.sin(np.linspace(0, 2*np.pi, 50)), i * np.cos(np.linspace(0, 2*np.pi, 50)), color = 'black', linewidth = 0.5)
        
        do = 1
        x_min = np.min([int(np.floor(u_min/do)*do)-do, -do]); y_min = np.min([int(np.floor(v_min/do)*do)-do, -do])
        x_max = np.max([int(np.ceil(u_max/do)*do)+do, do]); y_max = np.max([int(np.ceil(v_max/do)*do)+do, do])
        x_range = x_max-x_min; y_range = y_max-y_min
        if x_range>y_range: 
            y_min -= (x_range-y_range)/2; y_max += (x_range-y_range)/2
        else:
            x_min -= (y_range-x_range)/2; x_max += (y_range-x_range)/2
    
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    for j in range(len(ax.flat)):
        try:
            plot_windcycles(ax.flat[j], j)
            if j == 0:
                ax.flat[j].set_xlabel('u (m/s)'); ax.flat[j].set_ylabel('v (m/s)')
        except Exception: continue #Will occur when j >= n_days
    plt.figlegend(handles_windcycle, [str(j)+' m' for j in plot_heights]+['V_g'], loc = [0.37,0.05], ncol = 4, labelspacing=0., fontsize = 12 )
    plt.figlegend(handles_windcycle, [str(j)+' m' for j in plot_heights]+['V_g'], loc = [0.915,0.5], ncol = 1, labelspacing=0., fontsize = 12 )
    plt.savefig(s.imgs_path+'_wind_cycle_'+s.year+s.month+'.jpg', dpi = 120, bbox_inches = 'tight')
    plt.show()
    
    #%%
    fig, ax = plt.subplots(int(np.ceil(s.n_days/5)),5, figsize = (20,20))
    
    handles_theta = []
    def plot_thetaprofiles(ax, j):
        for i in range(len(plot_hours)):
            hour = plot_hours[i]
            time_index = np.argmin(np.abs(data.hours - hour))
            
            theta_j = data.theta[j, time_index]
            theta_min = theta_j.min() if i == 0 else np.min([theta_min, theta_j.min()])
            theta_max = theta_j.max() if i == 0 else np.max([theta_max, theta_j.max()])
            
            handles_theta.append(ax.plot(theta_j, data.z, color = colors[i])[0])
        ax.set_xlim([theta_min - 2, theta_max + 2])
        ax.text(figure_numbers_pos[0], figure_numbers_pos[1], str(j+1)+')', transform=ax.transAxes, fontsize = 15)
        ax.grid()
    
    for j in range(len(ax.flat)):
        try:
            plot_thetaprofiles(ax.flat[j], j)
            if j == 0:
                ax.flat[j].set_xlabel('$\\theta$ (K)'); ax.flat[j].set_ylabel('h (m)')
        except Exception: continue #Will occur when j >= n_days
    plt.figlegend(handles_theta, [format(j, '02d')+'z' for j in plot_hours], loc = [0.37,0.05], ncol = 4, labelspacing=0., fontsize = 12)
    plt.figlegend(handles_theta, [format(j, '02d')+'z' for j in plot_hours], loc = [0.925,0.5], ncol = 1, labelspacing=0., fontsize = 12)
    plt.savefig(s.imgs_path+'_theta_'+s.year+s.month+'.jpg', dpi = 120, bbox_inches = 'tight')
    plt.show()
    
    #%%
    fig, ax = plt.subplots(s.n_days, 3, figsize = (12, 100))
    for i in range(len(ax)):
        plot_thetaprofiles(ax[i][0], i)
        plot_windprofiles(ax[i][1], i)
        plot_windcycles(ax[i][2], i)
        if i == 0:
            ax[i][0].set_xlabel('$\\theta$ (K)'); ax[i][0].set_ylabel('h (m)')
            ax[i][1].set_xlabel('u (m/s)'); ax[i][1].set_ylabel('v (m/s)')
            ax[i][2].set_xlabel('u (m/s)'); ax[i][2].set_ylabel('v (m/s)')
    plt.figlegend(handles_theta, [format(j, '02d')+'z' for j in plot_hours], loc = [0.12,0.0625], ncol = 1, labelspacing=0., fontsize = 12)
    plt.figlegend(handles_windprofile, [format(j, '02d')+'z' for j in plot_hours], loc = [0.44,0.0625], ncol = 1, labelspacing=0., fontsize = 12)
    plt.figlegend(handles_windcycle, [str(j)+' m' for j in plot_heights]+['V_g'], loc = [0.78,0.0625], ncol = 1, labelspacing=0., fontsize = 12)
    
    plt.savefig(s.imgs_path+'_combi_'+s.year+s.month+'.jpg', dpi = 120, bbox_inches = 'tight')
    plt.show()