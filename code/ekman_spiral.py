# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:37:18 2018

@author: bramv
"""
import numpy as np
import matplotlib.pyplot as plt

import read_cabauw_data as r
import calculate_geostrophic_wind as gw



months = list(range(1, 9))
#months = [8]
data = r.read_and_process_cabauw_data(months = months)
gw_data = gw.calculate_geostrophic_wind(months = months)


Vg = gw_data.V_g
Vg_speed = gw_data.V_g_speed
"""Normalize V by rotating the wind profile in such a way that the geostrophic wind vector points to the east, and
scaling the profile by the geostrophic wind speed.
"""
angle_rotate = np.arctan2(Vg[:,:,1], Vg[:,:,0])
rot_matrix = np.zeros(data.V.shape[:2] + (2, 2))
rot_matrix[:,:,0,0] = np.cos(angle_rotate); rot_matrix[:,:,0,1] = np.sin(angle_rotate)
rot_matrix[:,:,1,0] = - np.sin(angle_rotate); rot_matrix[:,:,1,1] = np.cos(angle_rotate)
normalized_V = np.matmul(data.V[:,:,:-1], np.transpose(rot_matrix, axes = [0,1,3,2])) / Vg_speed[:,:,np.newaxis,np.newaxis]

Vg_speed = Vg_speed.flatten()
normalized_V = np.reshape(normalized_V, (data.V.shape[0] * data.V.shape[1], data.V.shape[2] - 1, data.V.shape[3]))
dtheta = (data.theta[:,:,0] - data.theta[:,:,-2]).flatten() #Difference in theta between 10 and 200 m



Vg_speed_lowerlimit = 5
Vg_speed_upperlimit = 15
stability_classes = [[-2, 0], [0, 2], [2, 4], [4, 6], [6, 10]]
data_classes = {}
data_means_classes = {}
for c in stability_classes:
    data_classes[str(c)] = {}; data_means_classes[str(c)] = {}
    in_class = (dtheta >= c[0]) & (dtheta < c[1]) & (Vg_speed >= Vg_speed_lowerlimit) & (Vg_speed <= Vg_speed_upperlimit)
    print(c, np.count_nonzero(in_class))
    data_classes[str(c)]['Vg_speed'] = Vg_speed[in_class]
    data_classes[str(c)]['normalized_V'] = normalized_V[in_class]
    data_classes[str(c)]['dtheta'] = dtheta[in_class]
    
    data_means_classes[str(c)]['Vg_speed'] = np.mean(data_classes[str(c)]['Vg_speed'])
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
plt.xlim([-0.1, 1.5]); plt.ylim([-0.1, 0.75])
plt.axes().set_aspect('equal', 'box')
    
plt.legend(labels = [str(c) + ' K' for c in stability_classes] + ['Ekman'])
plt.grid()
plt.show()



plt.figure()
c = str(stability_classes[-1])
for j in range(data_classes[c]['normalized_V'].shape[0]):
    plt.plot(data_classes[c]['normalized_V'][j,:,0], data_classes[c]['normalized_V'][j,:,1])
plt.show()