# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:37:18 2018

@author: bramv
"""
import numpy as np
import matplotlib.pyplot as plt

import functions as ft
import read_cabauw_data as r
import calculate_geostrophic_wind as gw
import calculate_temperature_gradient as tg



months = list(range(1, 9))
#months = [1]
data = r.read_and_process_cabauw_data(months = months)
gw_data = gw.calculate_geostrophic_wind(months = months)
tg_data = tg.calculate_temperature_gradient(months = months)



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

#%%
Vg = gw_data.V_g[:,:,np.newaxis,:] + (data.z[np.newaxis, np.newaxis, :-1, np.newaxis] - 1.5) / 198.5 * V_thermal[:,:,np.newaxis,:]
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
stability_classes = [[-2, 0], [0, 2], [2, 4], [4, 6], [6, 10]]
data_classes = {}
data_means_classes = {}
for c in stability_classes:
    data_classes[str(c)] = {}; data_means_classes[str(c)] = {}
    in_class = (dtheta >= c[0]) & (dtheta < c[1]) & (Vg_speed_0 >= Vg_speed_0_lowerlimit) & (Vg_speed_0 <= Vg_speed_0_upperlimit)
    print(c, np.count_nonzero(in_class))
    data_classes[str(c)]['Vg_speed_0'] = Vg_speed_0[in_class]
    data_classes[str(c)]['normalized_V'] = normalized_V[in_class]
    data_classes[str(c)]['dtheta'] = dtheta[in_class]
    
    data_means_classes[str(c)]['Vg_speed_0'] = np.mean(data_classes[str(c)]['Vg_speed_0'])
    data_means_classes[str(c)]['normalized_V'] = np.mean(data_classes[str(c)]['normalized_V'], axis = 0)
    data_means_classes[str(c)]['dtheta'] = np.mean(data_classes[str(c)]['dtheta'])   
    
    
    
plt.figure()
for c in stability_classes:
    c = str(c)
    plt.plot(data_means_classes[c]['normalized_V'][:,0], data_means_classes[c]['normalized_V'][:,1]) 
#Plot the analytical Ekman profile
z = np.arange(0,10,0.1)
u_a = 1. - np.exp(-z) * np.cos(z)
v_a = np.exp(-z) * np.sin(z)
plt.plot(u_a, v_a)
plt.xlim([-0.2, 1.5]); plt.ylim([-0.2, 0.75])
plt.axes().set_aspect('equal', 'box')
    
plt.legend(labels = [str(c) + ' K' for c in stability_classes] + ['Ekman'])
plt.grid()
plt.show()



plt.figure()
c = str(stability_classes[-1])
for j in range(data_classes[c]['normalized_V'].shape[0]):
    plt.plot(data_classes[c]['normalized_V'][j,:,0], data_classes[c]['normalized_V'][j,:,1])
plt.show()