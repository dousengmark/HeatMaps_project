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
import Phase_Calc
import sys
import os

#os.remove("FFT_filter2.wav")
# Empty lists
frms = []
frms_no_filter = []

# Setup channel info
FORMAT = pyaudio.paInt16 # data type formate
CHANNELS = 6 # Adjust to your number of channels
RATE = 48000 # Sample Rate
CHUNK = 24000 # Block Size
RECORD_SECONDS = 2 # Record time
WAVE_OUTPUT_FILENAME = "FFT_filter2.wav" 

# Startup pyaudio instance
audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True, input_device_index=2,
                frames_per_buffer=CHUNK)
print( "recording...")

# Record for RECORD_SECONDS amount of time
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    # Read data with length of CHUNK from buffer
    data = stream.read(CHUNK)
    frames = np.frombuffer(data, dtype= np.int16)
    # Split stereo channels
    frames = np.reshape(frames, (6, CHUNK))
    
    print(np.shape(frames))
    print(frames)
    nsamples = len(frames)
    # Store data
    frms_no_filter.append(frames)    
    frms.append(frames)

print("_________________")
print(frms[1])
print("_________________")


# Write result in .wav file
wf = wave.open('FFT_filter2.wav', 'wb') # Open file
wf.setnchannels(CHANNELS) # Set Number of Channels
wf.setsampwidth(2) # Set number of bytes per sample
wf.setframerate(RATE) # Set framerate
wf.writeframes(b''.join(frms)) # Store data
wf.close() # Close
time.sleep(1) # Delay 1 s


import matplotlib.pyplot as plt

# Read data that will be FFTed
sr, data = wavfile.read('FFT_filter2.wav', 'wb')
# data = np.reshape(data, (2, len(data)))

# Split channels
# Channel 1
data0 = data[:,0]
# Channel 2
data1 = data[:,1]
data2 = data[:,2]
data3 = data[:,3]
print(data)
print("++++++++++++++++++++++===")

#Hamming window which will be multiplied by data from each mic
window = np.hamming(48000)
# FFT the second second of the 1st channel's data
y_f = np.fft.fft(data0[48000:96001]*window)
y_f3 = np.fft.fft(data2[48000:96001]*window)
y_f4 = np.fft.fft(data3[48000:96001]*window)
# Generate frequency range (d = inverse sampling rate)
freqz = np.fft.fftfreq(len(y_f), d = 1/48000)

# Statements for debug 
print(freqz)
N = len(y_f)
ref =32768
print(len(data), N, len(freqz))
#y_ff = np.fft.fft(data2)
#print(len(y_ff), len(y_f))
print(len(y_f))

# Print magnitude of 500 Hz freq
print(20*np.log10(np.abs(y_f[500])/ref))
print("-------------------")

# Find freq with highest Magnitude
rslt = np.where(y_f[20:20000] == max(y_f[20:20000]))
print("Elements with value 15 exists at following indices", rslt[0], sep='\n')
print("-------------------")

# Print angle of 500 Hz freq
print(np.angle(y_f[500]))
#plt.plot(range(len(y_ff[:N//2])), 2.0/N * np.abs(y_ff[:N//2]))

# # Plot 1st channel's FFT
# plt.plot(freqz, 20 * np.log10((np.abs(y_f)/ref)))
# plt.xlim(20, 20000)
# plt.xscale('log')
# plt.title("FFT")
# plt.show()
# 
# # Plot 1st channel's FFT's angle
# plt.plot(range(len(y_f)), np.angle(y_f))
# plt.xlim(100, 800)
# plt.xscale('log')
# plt.title("Phases")
# plt.show()


# FFT the second second of the 2nd channel's data  
y_f2 = np.fft.fft(data1[48000:96001]*window)
# Create freq range
freqz2 = np.fft.fftfreq(len(y_f2), d = 1/48000) 

""" 
Frequency of interest
"""
freqq = 400
freqq = int(freqq)

# # Plot FFT of channel 2
# 
# 
# # Plot Phase plot of channel 2
# plt.plot(range(len(y_f)), np.angle(y_f2))
# plt.xlim(100, 800)
# plt.xscale('log')
# plt.title("Phases")
# plt.show()

# Print magnitudes of freq of interest
print("magnitude, freq: ", freqq)
print("channel 1: ",20*np.log10(np.abs(y_f[freqq])/ref))
print("channel 2: ",20*np.log10(np.abs(y_f2[freqq])/ref))
print("phase, freq: ", freqq)

# Get phase angle
an1 = np.angle(y_f[int(freqq)])
an2 = np.angle(y_f2[int(freqq)])
an3 = np.angle(y_f3[int(freqq)])
an4 = np.angle(y_f4[int(freqq)])

# Print phase of freq of interest
print("channel 1: ",an1)
print("channel 2: ",an2)
print("difference, phases: ", np.angle(y_f[freqq])-np.angle(y_f2[freqq]))


''' Calculate directional mic ANGLE '''
Phase_Calc.Angle(an1, an2, freqq, 0.04134)
Phase_Calc.Angle(an1, an3, freqq, 0.04134*np.sqrt(2))
Phase_Calc.Angle(an1, an4, freqq, 0.112942980385)
Phase_Calc.Angle(an2, an3, freqq, 0.04134)
Phase_Calc.Angle(an2, an4, freqq, 0.04134*np.sqrt(2))
Phase_Calc.Angle(an3, an4, freqq, 0.04134)


# Stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

# sys.quit()