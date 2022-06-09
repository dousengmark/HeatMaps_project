# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:03:34 2021

@author: Olga
"""
import pyaudio # Soundcard audio I/O access library
import wave # Python 3 module for reading / writing simple .wav files

# Import bunch of stuff
#from scipy.signal import butter, lfilter
#import librosa, librosa.display
import scipy.io.wavfile as towav
from scipy.io import wavfile
import numpy as np
import time
# import Phase_Calc
import sys
import os

#os.remove("FFT_filter2.wav")
# Empty lists
frms = []
frms_no_filter = []




import matplotlib.pyplot as plt

# Read data that will be FFTed
sr, data = wavfile.read('FFT_filter2 02.wav', 'wb')
# data = np.reshape(data, (2, len(data)))

# Split channels
# Channel 1
data0 = data[:,2]
# Channel 2
data1 = data[:,3]
data2 = data[:,4]
data3 = data[:,5]
data4 = data[:,6]
data5 = data[:,7]






''' NEW CODE GUYS! This time we do beamforming! '''
# Import cheat codes for beam forming
from arlpy import bf, utils
import pyroomacoustics as pr
# Three mics
# pos = [[0,0],[-0.046,0],[-0.0693,0.0400]]
print(data[:,0].dtype)

pos = [[0,0],[-0.046,0],[-0.0693,0.0400],[-0.046, 0.0800],[0, 0.08000],[0.023,0.0400]]

d7 = (data[:,7])
d6 = (data[:,6])
d5 = (data[:,5])
d4 = (data[:,4])
d3 = (data[:,3])
d2 = (data[:,2])
d1 = (data[:,1])
d0 = (data[:,0])
stft = pr.transform.STFT(24000)

s2 = stft.analysis(d2).T

s3 = stft.analysis(d3).T
s4 = stft.analysis(d4).T
s5 = stft.analysis(d5).T
s6 = stft.analysis(d6).T
s7 = stft.analysis(d7).T
print(s2)
sigs = np.array([s2,s3,s4,s5,s6,s7])
print(sigs)
doa = pr.doa.MUSIC(np.asarray(pos).T, 48000, 24000)
doa.locate_sources(sigs)
doa.polar_plt_dirac()



''' Calculate directional mic ANGLE '''
