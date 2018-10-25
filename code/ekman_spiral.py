# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:37:18 2018

@author: bramv
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import settings as s
import functions as ft
import read_cabauw_data as r
import calculate_geostrophic_wind as gw
import calculate_temperature_gradient as tg


mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['axes.labelweight'] = 'bold'
for j in ('xtick','ytick'):
    mpl.rcParams[j+'.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 18



#%%
months = list(range(1,13)) * 9 + list(range(1, 9))
years = [2009] * 12 + [2010] * 12 + [2011] * 12 + [2012] * 12 + [2013] * 12 + [2014] * 12 + [2015] * 12 + [2016] * 12 + [2017] * 12 + [2018] * 8
#months = list(range(1,9))
#years = [2018] * 8
data = r.read_and_process_cabauw_data(years, months)
gw_data = gw.calculate_geostrophic_wind(years, months)
tg_data = tg.calculate_temperature_gradient(years, months)



#%%
"""First take into account vertical variations of the geostrophic wind, by adding the thermal wind to the geostrophic wind.
"""
#Calculate the pressure at 200 m by integrating the hydrostatic balance
g = 9.81
R = 287.
f = 2 * 7.29e-5 * np.sin(gw_data.coords_cabauw[1] * np.pi/180.)

def f_dpdz_dTdz(y): #y = np.array([p, T])
    p, T = y
    dpdz = -g/R * p/T
    return np.array([dpdz, dTdz]) #dTdz is defined in the loop below
    
p_0 = data.p0 * 100. #Pressure at 1.5 m
p_200m = p_0.copy()
for i in range(1, len(data.z)):
    z_old = data.z[-i]; z_new = data.z[-(i+1)]
    dz = z_new - z_old
    dTdz = (data.T[:,:,-(i+1)] - data.T[:,:,-i]) / dz
    p_200m += ft.RK4(f_dpdz_dTdz, np.array([p_200m, data.T[:,:,-i]]), dz)[0]
V_thermal = R / f * np.array([[[-1, 1]]]) * tg_data.T_gradient_xy[:,:,::-1] * np.log(p_200m / p_0)[:,:,np.newaxis]


Vg = gw_data.V_g[:,:,np.newaxis,:] + (data.z[np.newaxis, np.newaxis, :-1, np.newaxis] - 1.5) / 198.5 * V_thermal[:,:,np.newaxis,:]
#Vg = data.V[:,:,0][:,:,np.newaxis,:]
Vg_speed = np.linalg.norm(Vg, axis = 3)
Vg_speed_0 = Vg_speed[:,:,-1]
V = data.V[:,:,:-1]



"""Normalize V by rotating the wind profile in such a way that the geostrophic wind vector points to the east, and
scaling the profile by the geostrophic wind speed.
"""
angle_rotate = np.arctan2(Vg[:,:,:,1], Vg[:,:,:,0])
rot_matrix = np.zeros(V.shape[:3] + (2, 2))
rot_matrix[:,:,:,0,0] = np.cos(angle_rotate); rot_matrix[:,:,:,0,1] = np.sin(angle_rotate)
rot_matrix[:,:,:,1,0] = - np.sin(angle_rotate); rot_matrix[:,:,:,1,1] = np.cos(angle_rotate)
normalized_V = np.matmul(V[:,:,:,np.newaxis,:], np.transpose(rot_matrix, axes = [0,1,2,4,3]))[:,:,:,0] / Vg_speed[:,:,:,np.newaxis]

Vg_speed_0 = Vg_speed_0.flatten()
normalized_V = np.reshape(normalized_V, (V.shape[0] * V.shape[1], V.shape[2], V.shape[3]))
dtheta = (data.theta[:,:,0] - data.theta[:,:,-2]).flatten() #Difference in theta between 10 and 200 m



Vg_speed_0_lowerlimit = 5
Vg_speed_0_upperlimit = 15
stability_classes = [[-2, 0], [0, 3], [3, 6], [6, 10]]

data_time_ranges = [[i, i+3] for i in range(0, 24, 3)]
for data_time_range in data_time_ranges:
    data_classes = {}
    data_means_classes = {}
    data_stdevs_classes = {}
    data_classes_nsamples = {}

    for c in stability_classes:
        data_classes[str(c)] = {}; data_means_classes[str(c)] = {}; data_stdevs_classes[str(c)] = {}; data_classes_nsamples[str(c)] = {}
        in_class = (dtheta >= c[0]) & (dtheta < c[1]) & (Vg_speed_0 >= Vg_speed_0_lowerlimit) & (Vg_speed_0 <= Vg_speed_0_upperlimit)
        if data_time_range[0] < data_time_range[1]:
            time_criterion = (data.hours.flatten() >= data_time_range[0]) & (data.hours.flatten() <= data_time_range[1])
        else:
            time_criterion = (data.hours.flatten() >= data_time_range[0]) | (data.hours.flatten() <= data_time_range[1])
        in_class = in_class & (np.linalg.norm(V_thermal, axis = 2).flatten() < 2) & time_criterion
        
        data_classes[str(c)]['normalized_V'] = normalized_V[in_class]    
        data_means_classes[str(c)]['normalized_V'] = np.mean(data_classes[str(c)]['normalized_V'], axis = 0)
        diffs = np.linalg.norm(data_classes[str(c)]['normalized_V'] - data_means_classes[str(c)]['normalized_V'][np.newaxis,:,:], axis = 2)
        data_stdevs_classes[str(c)]['normalized_V'] = np.std(diffs, axis = 0)
        data_classes_nsamples[str(c)]['normalized_V'] = np.count_nonzero(in_class)
    
    
    
    plt.figure(figsize = (20, 10))
    colors = ['blue', 'red', 'green', 'yellow', 'black']
    legend_handles = []
    for i in range(len(stability_classes)-1, -1, -1):
        c = str(stability_classes[i])
        legend_handles.append(plt.plot(data_means_classes[c]['normalized_V'][:,0], data_means_classes[c]['normalized_V'][:,1], colors[i], linewidth = 6)[0])
        index_diff = 6
        normal_vector = np.cross([0, 0, 1], np.concatenate([[data_means_classes[c]['normalized_V'][index_diff-1] - data_means_classes[c]['normalized_V'][0] for j in range(1, index_diff//2 + 1)], data_means_classes[c]['normalized_V'][index_diff:] - data_means_classes[c]['normalized_V'][:-index_diff], [data_means_classes[c]['normalized_V'][-1] - data_means_classes[c]['normalized_V'][-index_diff] for j in range(2, index_diff//2 + 2)]], axis = 0))[:,:2]
        #Normalize the normal vector
        normal_vector /= np.linalg.norm(normal_vector, axis = 1)[:, np.newaxis]
        stdev_points1 = data_means_classes[c]['normalized_V'] - data_stdevs_classes[str(c)]['normalized_V'][:, np.newaxis] * normal_vector
        stdev_points2 = data_means_classes[c]['normalized_V'] + data_stdevs_classes[str(c)]['normalized_V'][:, np.newaxis] * normal_vector
        plt.plot(stdev_points1[:, 0], stdev_points1[:, 1], colors[i], linestyle = '--', linewidth = 3)
        plt.plot(stdev_points2[:, 0], stdev_points2[:, 1], colors[i], linestyle = '--', linewidth = 3)
        
        C = np.array([[1] for j in range(stdev_points1.shape[0])]).T
        X = np.array([stdev_points1[:,0], stdev_points2[:,0]])
        Y = np.array([stdev_points1[:,1], stdev_points2[:,1]])
        cmap = mpl.colors.ListedColormap([colors[i]])
        plt.pcolormesh(X, Y, C, cmap = cmap, alpha = 0.15, edgecolor = 'none')
    
    #Plot the analytical Ekman profile
    z = np.arange(0,10,0.1)
    u_a = 1. - np.exp(-z) * np.cos(z)
    v_a = np.exp(-z) * np.sin(z)
    legend_handles.append(plt.plot(u_a, v_a, colors[-1], linewidth = 3)[0])
    plt.xlim([-0.1, 1.6]); plt.ylim([-0.1, 0.75])
    plt.xlabel('u / G'); plt.ylabel('v / G')
    plt.title('Normalized 10-200 meter wind profile compared to Ekman profile for '+format(data_time_range[0], '02d')+'-'+format(data_time_range[1], '02d')+'Z')
    plt.axes().set_aspect('equal', 'box')
    plt.legend(legend_handles, [str(c) + ' K:  '+str(data_classes_nsamples[str(c)]['normalized_V']) for c in stability_classes[::-1]] + ['Ekman'])
    plt.grid()
    plt.savefig(s.imgs_path+'normalized_BL_windprofile_'+str(data_time_range[0])+'-'+str(data_time_range[1])+'Z.jpg', dpi = 120, bbox_inches = 'tight')
    plt.show()
    
    
    
    plt.figure()
    c = str(stability_classes[-1])
    for j in range(data_classes[c]['normalized_V'].shape[0]):
        plt.plot(data_classes[c]['normalized_V'][j,:,0], data_classes[c]['normalized_V'][j,:,1])
    plt.show()