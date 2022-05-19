# -*- coding: utf-8 -*-
"""
Created on Thu May  5 17:13:58 2022

@author: Olga

Call this file as a library in a separate script

Example:
    import Phase_Calc
    
    Phase_Calc.Angle(an1, an2, freqq)
    
    where an1 is first angle in radians, an2 second angle in radians 
    and freqq is frequency of interest
    
"""

import numpy as np
# Far field model

# Main function
def Angle(phase1, phase2, freq):
    # Function to convert phase angle (rad) to time delay (seconds)
    def phase_to_td(phase, freq):
        # Error when phase is weird
        if phase > 2*np.pi or phase < -2*np.pi:
            raise("Phase out of range: ", phase)
        else:    
            # Return the time delay
            # (phase/(2pi*f))
            print(phase/(2*np.pi)/freq)
            return phase/(2*np.pi)/freq
    
    # Function to determine direction of the source in Deg    
    # Input: time delay 1, time delay 2, distance between mics, frequency
    def direction(td1, td2, dist, freq):
        # Measure time difference
        td = td1-td2
        print(td)
        
        # Unitless value (far field model)
        thing = td*343/dist
        
        # Clipping
        # if thing > 1:
        #     thing = 1
        # elif thing < -1:
        #     thing = -1
        print(thing)
        
        # Convert to Rad
        angle = np.arccos(thing)
        # Convert to Deg
        angle = angle*180/np.pi
        return angle

    
    
    
    # Run functions and return the result
    td1 = phase_to_td(phase1, freq)
    td2 = phase_to_td(phase2, freq)
    out = direction(td1, td2, 0.06478, freq)
    print(out)
    return out

# Angle(0, -0.251728845, 300)