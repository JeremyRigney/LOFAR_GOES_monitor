#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:00:14 2023

@author: jeremy
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
#%config InlineBackend.figure_format = 'retina'

import numpy as np
import os
from pathlib import Path
import json
import requests

from astropy import units as u
from astropy.time import Time, TimeDelta

from sunpy.net import Fido, attrs as a 
from sunpy import timeseries as ts

from matplotlib.ticker import ScalarFormatter

from radiospectra.spectrogram2 import Spectrogram


def sb_to_freq(sb, obs_mode):
    """
    Converts LOFAR single station subbands to frequency
    Returns frequency as astropy.units.Quantity (MHz)
    Inputs: subband number, observation mode (3, 5 or 7)
    """
    nyq_zone_dict = {3:1, 5:2, 7:3}
    nyq_zone = nyq_zone_dict[obs_mode]
    clock_dict = {3:200, 4:160, 5:200, 6:160, 7:200} #MHz
    clock = clock_dict[obs_mode]
    freq = (nyq_zone-1+sb/512)*(clock/2)
    return freq*u.MHz #MHz

#LOFAR date time range  (can be made dynamic by pulling from scheduling file at a later date)
datetime_start = "2023/07/18 06:30:00"
datetime_end = "2023/07/18 11:00:00"



# All the GOES data aquisition and parsing

# Get the last 7 days of GOES data
goes_xray = json.loads(requests.get('https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json').text)

goes_timestamps = []
goes_xrayflux = []
goes_xrayfluxb = []

# Split into GOES short and long
for i in goes_xray:
    if i['energy'] == '0.05-0.4nm':
        goes_timestamps.append(i['time_tag'])
        goes_xrayflux.append(i['observed_flux'])
    if i['energy'] == '0.1-0.8nm':
        goes_xrayfluxb.append(i['observed_flux'])

# Creating datetime objects for GOES timeseries subplot
goes_times = []
for i in goes_timestamps:
    goes_times.append(datetime.strptime(i, '%Y-%m-%dT%H:%M:%SZ'))

    

# If you want to plot GOES data from earlier than 7 days ago, use the following code to get the data (sunpy FIDO) 
# (needs some modifying, sometimes there's no data and it crashes returning an empty list)

# GOES tstart and tend need to encompass the time the LOFAR data covers (datetime_start, datetime_end)

tstart = "2023-07-18 00:00:00.000"
tend = "2023-07-18 12:00:00.000"
try:    
    result = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("flx1s"), a.goes.SatelliteNumber(16))
    file_goes = Fido.fetch(result)
    
    goes_all = ts.TimeSeries(file_goes, concatenate=True)
    goes_all = goes_all.to_dataframe()
    
    all_goes_short_data = goes_all['xrsb_flux']
    all_goes_long_data = goes_all['xrsa_flux']
    all_goes_times = goes_all['time']

except:
    pass


#LOFAR data file
cur_folder = Path(__file__).parent.resolve()
bstfile357 = str(cur_folder / "sampledata/20230718_065932_bst_00X.dat")
data357 = np.fromfile(bstfile357)

print("Number of data points:",data357.shape[0])
print("File size:",os.path.getsize(bstfile357))
print("Bitmode:",os.path.getsize(bstfile357)/data357.shape[0])
t_len357 = data357.shape[0]/488
print("Time samples:",t_len357 )
data357 = data357.reshape(-1,488)
data357 = data357.T

sbs = np.array((np.arange(54,454,2),np.arange(54,454,2),np.arange(54,230,2)), dtype=object)
blank_sbs = np.array((np.arange(454,512,2),np.arange(0,54,2),np.arange(454,512,2),np.arange(0,54,2)), dtype=object)
obs_mode = np.array((3,5,7))
blank_obs_mode = np.array((3,5,5,7))
freqs = np.array([sb_to_freq(sb,mode) for sb,mode in zip(sbs, obs_mode)], dtype=object)
blank_freqs = np.array([sb_to_freq(sb,mode) for sb,mode in zip(blank_sbs, blank_obs_mode)], dtype=object)

sbs=np.concatenate((sbs[0], blank_sbs[0], blank_sbs[1], sbs[1],
                            blank_sbs[2], blank_sbs[3], sbs[2]))

freqs=np.concatenate((freqs[0], blank_freqs[0], blank_freqs[1], freqs[1],
                            blank_freqs[2], blank_freqs[3], freqs[2]))


blank_data = np.zeros((freqs.shape[0],data357.shape[1]))
#1st 200 sbs mode 3, blank, next 200 sbs mode 5, blank, last 88 sbs mode 7
blank_data[:200,:] = data357[:200,:]
blank_len_0 = len(blank_freqs[0]) + len(blank_freqs[1])
blank_data[200 + blank_len_0:400 + blank_len_0,:] = data357[200:400,:]
blank_len_1 = len(blank_freqs[2]) + len(blank_freqs[3])
blank_data[400 + blank_len_0 + blank_len_1 :,:] = data357[400:,:]
data357 = blank_data

mask = np.zeros(data357.shape)
mask[200:256,:] = 1	#1st 200 sbs are mode 3, 56 blank sbs
mask[456:512,:] = 1 #Next 200 sbs are mode 5, 56 blank sbs. Last 88 sbs are mode 7
data357 = np.ma.array(data357, mask=mask)


obs_start357 = bstfile357[len(bstfile357)-27:len(bstfile357)-12]
obs_start357 = Time.strptime(obs_start357, "%Y%m%d_%H%M%S")
t_arr357 = np.arange(0,t_len357)
t_arr357 = t_arr357*TimeDelta(1*u.s, format='sec')
t_arr357 = obs_start357+t_arr357

data357 = (data357.T/np.mean(data357[:,10:20], axis=1)).T

print('Data ready to plot')



fig = plt.figure(figsize=(14,8))
gs = fig.add_gridspec(2, hspace=0, height_ratios=[1,3]) #hspace=0.05
axs = gs.subplots(sharex=True)

# for plotting data older than 7 days
#axs[0].plot(all_goes_times, all_goes_short_data, '-',c='orange', label = 'GOES Short 0.05 - 0.4 nm')
#axs[0].plot(all_goes_times, all_goes_long_data, '-b', label = 'GOES Short 0.05 - 0.4 nm')

axs[0].plot(goes_times,goes_xrayflux, '-b', label='GOES 0.05 - 0.4 nm')
axs[0].plot(goes_times,goes_xrayfluxb, '-',c='orange', label='GOES 0.1 - 0.8 nm')



axs[0].xaxis_date()
date_format2 = mdates.DateFormatter('%H:%M')
axs[0].xaxis.set_major_formatter(date_format2)
axs[1].xaxis.set_major_formatter(date_format2)
axs[0].set_yscale('log')
axs[0].set_ylim(1.01e-8, 0.5e-3)

# All ax2 for Flare Class axis
ax2 = axs[0].twinx()
mn, mx = axs[0].get_ylim()
ax2.set_ylim(mn, mx)
ax2.set_yscale('log')
ax2.set_ylabel('Flare Class', fontsize=16)
ax2.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
ax2.set_yticklabels(['A','B','C','M','X'])
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.grid('True', which='both',axis='both', ls='dashed', alpha=0.3)

axs[0].grid(which='major', color='b', linestyle='-')

axs[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=30)) # to get a tick every 2 hours
axs[0].grid('True', which='major',axis='both', ls='dashed', alpha=0.3)

#plt.minorticks_on()
axs[0].tick_params(bottom=True, top=True, left=True, right=True,direction="in")
axs[1].tick_params(bottom=True, top=True, left=True, right=True,direction="in")

axs[0].grid(which='major', linestyle = ':', c='gray', alpha=0.3)

axs[0].xaxis_date()
axs[0].xaxis.set_major_formatter(date_format2)


axs[1].imshow(data357, aspect="auto", extent=[t_arr357[0].plot_date, t_arr357[-1].plot_date, freqs.value[-1], freqs.value[0]],
         vmin = np.percentile(data357, 5), vmax = np.percentile(data357, 99), cmap='RdBu_r')

axs[1].set_xlim(datetime.strptime(datetime_start, '%Y/%m/%d %H:%M:%S'), datetime.strptime(datetime_end, '%Y/%m/%d %H:%M:%S'))


#axs[1].set_ylim(50, 185)
#axs[1].set_yticks([50, 60, 70, 80, 90, 100, 120, 140, 160, 180], minor=True)

#axs[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=30)) # to get a tick every n minutes
axs[1].xaxis.set_major_locator(mdates.HourLocator(interval=1)) # to get a tick every n hours, must be changed on both subplots
axs[0].xaxis.set_major_locator(mdates.HourLocator(interval=1)) # to get a tick every n hours, must be changed on both subplots


axs[1].set_xlabel("Time (HH:MM) "+datetime_start[0:10], fontsize=16)

fig.text(0.065, 0.35, "Frequency (MHz)", rotation="vertical", va="center",fontsize=16)
fig.text(0.065, 0.8, "Watts $m^{-2}$", rotation="vertical", va="center",fontsize=16)


axs[0].tick_params(axis='y', which='major', labelsize=14)
axs[0].tick_params(axis='y', which='minor', labelsize=14)

axs[0].set_yticks([ 1e-7, 1e-6, 1e-5, 1e-4], minor=True)

axs[1].set_yscale('log')

axs[1].tick_params(axis='both', which='major', labelsize=14)

axs[1].tick_params(axis='y', which='minor', labelsize=14)
axs[1].xaxis_date()
axs[1].yaxis.set_major_formatter(ScalarFormatter())
axs[1].yaxis.set_minor_formatter(ScalarFormatter())

#plt.savefig('test.fig'...)
plt.show()


