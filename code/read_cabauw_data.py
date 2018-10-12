# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:57:40 2018

@author: bramv
"""
import numpy as np
import xarray as xr

import settings as s

g = 9.81
Cp = 1005


class Cabauw_Data():
    """Class to which data can be assigned
    """
    def __init__(self):
        pass
    
    
def read_and_process_cabauw_data():
    data = Cabauw_Data()
    
    f = xr.open_dataset(s.data_path+'cesar_tower_meteo_lc1_t10_v1.0_'+s.year+s.month+'.nc', decode_times = False)
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
    
    data.z = z
    data.hours = np.reshape(np.mod(time, 24), (n_days, n_times_day))
    data.speed = np.reshape(speed, (n_days, n_times_day, n_z))
    data.direction = np.reshape(direction, data.speed.shape)
    #- signs, because direction gives the direction from which the wind is blowing, and not to which the wind is blowing.
    data.u = - data.speed * np.sin(data.direction * np.pi/180.)
    data.v = - data.speed * np.cos(data.direction * np.pi/180.)
    data.V = np.zeros(data.u.shape + (2,))
    data.V[:,:,:,0] = data.u; data.V[:,:,:,1] = data.v
    data.T = np.reshape(T, data.speed.shape)
    data.theta = data.T + g/Cp*z[np.newaxis, np.newaxis, :]
    data.Td = np.reshape(Td, data.speed.shape)
    
    return data #Return data object with the data as attributes