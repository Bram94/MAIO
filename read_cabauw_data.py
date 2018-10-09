# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:33:36 2018

@author: bramv
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt



g = 9.81
Cp = 1005


f = xr.open_dataset('cesar_tower_meteo_lc1_t10_v1.0_201808.nc', decode_times = False)
print(f)
print(f.variables)
time = np.array(f.variables['time']) #Is given in hours, with data every 10 minutes
z = np.array(f.variables['z'])
speed = np.array(f.variables['F'])
direction = np.array(f.variables['D'])
T = np.array(f.variables['TA'])
Td = np.array(f.variables['TD'])

n_time = len(time); n_z = len(z)

#Reshape the data by adding an extra axis that represents different days
n_times_day = 6*24 #Data every 10 minutes
n_days = int(n_time / n_times_day)
hours = np.reshape(np.mod(time, 24), (n_days, n_times_day))
speed = np.reshape(speed, (n_days, n_times_day, n_z))
direction = np.reshape(direction, speed.shape)
u = speed * np.sin(direction * np.pi/180.)
v = speed * np.cos(direction * np.pi/180.)
T = np.reshape(T, speed.shape)
theta = T + g/Cp*z[np.newaxis, np.newaxis, :]
Td = np.reshape(Td, speed.shape)

#%%
figure_numbers_pos = [-0.125, 1.04]

fig, ax = plt.subplots(6,5, figsize = (20,20))
plot_hours = np.array([0, 6, 12, 18])
colors = ['blue', 'red', 'green', 'yellow']

handles_windprofile = []
def plot_windprofiles(ax, j):
    ax.set_aspect('equal')
    for i in range(len(plot_hours)):
        hour = plot_hours[i]
        time_index = np.argmin(np.abs(hours - hour))
        
        u_j, v_j = u[j, time_index, :-1], v[j, time_index, :-1] #Exclude the last element as it is np.nan
        ax.plot(u_j, v_j, color = colors[i], marker = 'o', markersize = 3)
        
        handles_windprofile.append(ax.plot(u_j, v_j, color = colors[i], linestyle = '-')[0])
        if i == 0:
            u_min = u_j.min(); u_max = u_j.max()
            v_min = v_j.min(); v_max = v_j.max()
        else:
            u_min = np.min([u_min, u_j.min()]); u_max = np.max([u_max, u_j.max()])
            v_min = np.min([v_min, v_j.min()]); v_max = np.max([v_max, v_j.max()])            
            
        for k in range(len(u_j)):
            if k in (0, len(u_j) - 1):
                ax.text(u_j[k], v_j[k], str(int(z[k])))
                
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
    plot_windprofiles(ax.flat[j],j)
plt.figlegend(handles_windprofile, [format(j, '02d')+'Z' for j in plot_hours], loc = [0.37,0.05], ncol = 4, labelspacing=0., fontsize = 12 )
plt.figlegend(handles_windprofile, [format(j, '02d')+'Z' for j in plot_hours], loc = [0.915,0.5], ncol = 1, labelspacing=0., fontsize = 12 )
plt.savefig('wind.jpg', dpi = 120, bbox_inches = 'tight')
plt.show()

#%%
fig, ax = plt.subplots(6,5, figsize = (20,20))
plot_heights = [10, 80, 200]
colors = ['blue', 'red', 'green', 'yellow']

handles_windcycle = []
def plot_windcycles(ax, j):
    ax.set_aspect('equal')
    for i in range(len(plot_heights)):
        height = plot_heights[i]
        z_index = np.argmin(np.abs(z - height))
        
        u_j, v_j = u[j, :, z_index], v[j, :, z_index]
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
    plot_windcycles(ax.flat[j], j)
plt.figlegend(handles_windcycle, [str(j)+' m' for j in plot_heights], loc = [0.37,0.05], ncol = 3, labelspacing=0., fontsize = 12 )
plt.figlegend(handles_windcycle, [str(j)+' m' for j in plot_heights], loc = [0.915,0.5], ncol = 1, labelspacing=0., fontsize = 12 )
plt.savefig('wind_cycle.jpg', dpi = 120, bbox_inches = 'tight')
plt.show()

#%%
fig, ax = plt.subplots(6,5, figsize = (20,20))

handles_theta = []
def plot_thetaprofiles(ax, j):
    for i in range(len(plot_hours)):
        hour = plot_hours[i]
        time_index = np.argmin(np.abs(hours - hour))
        
        theta_j = theta[j, time_index]
        theta_min = theta_j.min() if i == 0 else np.min([theta_min, theta_j.min()])
        theta_max = theta_j.max() if i == 0 else np.max([theta_max, theta_j.max()])
        
        handles_theta.append(ax.plot(theta_j, z, color = colors[i])[0])
    ax.set_xlim([theta_min - 2, theta_max + 2])
    ax.text(figure_numbers_pos[0], figure_numbers_pos[1], str(j+1)+')', transform=ax.transAxes, fontsize = 15)
    ax.grid()

for j in range(len(ax.flat)):
    plot_thetaprofiles(ax.flat[j], j)
plt.figlegend(handles_theta, [format(j, '02d')+'Z' for j in plot_hours], loc = [0.37,0.05], ncol = 4, labelspacing=0., fontsize = 12)
plt.figlegend(handles_theta, [format(j, '02d')+'Z' for j in plot_hours], loc = [0.925,0.5], ncol = 1, labelspacing=0., fontsize = 12)
plt.savefig('theta.jpg', dpi = 120, bbox_inches = 'tight')
plt.show()

#%%
fig, ax = plt.subplots(30, 3, figsize = (12, 100))
for i in range(len(ax)):
    plot_thetaprofiles(ax[i][0], i)
    plot_windprofiles(ax[i][1], i)
    plot_windcycles(ax[i][2], i)
plt.figlegend(handles_theta, [format(j, '02d')+'Z' for j in plot_hours], loc = [0.12,0.0625], ncol = 1, labelspacing=0., fontsize = 12)
plt.figlegend(handles_windprofile, [format(j, '02d')+'Z' for j in plot_hours], loc = [0.44,0.0625], ncol = 1, labelspacing=0., fontsize = 12)
plt.figlegend(handles_windcycle, [str(j)+' m' for j in plot_heights], loc = [0.78,0.0625], ncol = 1, labelspacing=0., fontsize = 12)

plt.savefig('combi.jpg', dpi = 120, bbox_inches = 'tight')
plt.show()