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


class Cabauw_Data(object):
    """Class to which data can be assigned
    """
    def __init__(self):
        pass
    
    
def read_and_process_cabauw_data(years = [], months = []): 
    """This function reads and processes Cabauw data, and returns a data object that contains the relevant datasets as attributes.
    If year and months are unspecified, then the year and month specified in settings.py are used.
    Otherwise both should be lists with the same length, that contain the years and months that need to be considered, 
    in which the ith year corresponds to the ith month.
    """
    if len(months) == 0: months = [s.month]
    else: months = [j if isinstance(j, str) else format(j, '02d') for j in months]
    if len(years) == 0: years = [s.year for j in months]
    else: years = [str(j) for j in years]

    data = Cabauw_Data()
    n_days = 0
    for i in range(len(months)):
        print('read_cabauw_data', years[i], months[i])
        f = xr.open_dataset(s.data_path+'cesar_tower_meteo_lc1_t10_v1.0_'+years[i]+months[i]+'.nc', decode_times = False)
        time = np.array(f.variables['time']) #Is given in hours, with data every 10 minutes
        z = np.array(f.variables['z'])
        speed = np.array(f.variables['F'])
        direction = np.array(f.variables['D'])
        T = np.array(f.variables['TA'])
        Td = np.array(f.variables['TD'])
        
        n_time = len(time); n_z = len(z)
        
        #Reshape the data by adding an extra axis that represents different days
        n_times_day = 6*24 #Data every 10 minutes
        n_days_month = int(n_time / n_times_day)
        n_days += n_days_month
        
        hours = np.reshape(np.mod(time, 24), (n_days_month, n_times_day))
        speed = np.reshape(speed, (n_days_month, n_times_day, n_z))
        direction = np.reshape(direction, speed.shape)
        #- signs, because direction gives the direction from which the wind is blowing, and not to which the wind is blowing.
        u = - speed * np.sin(direction * np.pi/180.)
        v = - speed * np.cos(direction * np.pi/180.)
        V = np.zeros(u.shape + (2,))
        V[:,:,:,0] = u; V[:,:,:,1] = v
        T = np.reshape(T, speed.shape)
        theta = T + g/Cp*z[np.newaxis, np.newaxis, :]
        Td = np.reshape(Td, speed.shape)
        
        
        #Import the second file that contains a.o. surface pressures
        f2 = xr.open_dataset(s.data_path+'cesar_surface_meteo_lc1_t10_v1.0_'+years[i]+months[i]+'.nc', decode_times = False)
        p0 = np.reshape(np.array(f2.variables['P0']), speed.shape[:2])
        
        variables = ['hours','speed','direction','u','v','V','T','theta','Td','z','p0']
        for j in variables:
            if i == 0:
                exec('data.'+j+' = '+j)
            elif j != 'z':
                exec('data.'+j+' = np.concatenate([data.'+j+','+j+'], axis = 0)')
                         
    return data #Return data object with the data as attributes