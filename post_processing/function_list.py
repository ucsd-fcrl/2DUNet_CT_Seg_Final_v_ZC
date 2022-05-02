#!/usr/bin/env python

# this script defined functions used in other scripts

import numpy as np
import math
import glob as gb
import glob
import os
from scipy.interpolate import RegularGridInterpolator
import nibabel as nib
from nibabel.affines import apply_affine
import math
import string
import matplotlib.pyplot as plt
import cv2
import pandas as pd



# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)

# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(gb.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F


# function: find time frame of a file
def find_timeframe(file,num_of_end_signal,start_signal = '/',end_signal = '.'):
    k = list(file)
    num_of_dots = num_of_end_signal

    if num_of_dots == 1: #.png
        num1 = [i for i, e in enumerate(k) if e == end_signal][-1]
    else:
        num1 = [i for i, e in enumerate(k) if e == end_signal][-2]
    num2 = [i for i,e in enumerate(k) if e== start_signal][-1]
    kk=k[num2+1:num1]
    total = 0
    for i in range(0,len(kk)):
        total += int(kk[i]) * (10 ** (len(kk) - 1 -i))
    return total

# function: sort files based on their time frames
def sort_timeframe(files,num_of_end_signal,start_signal = '/',end_signal = '.'):
    time=[]
    time_s=[]
    num_of_dots = num_of_end_signal

    for i in files:
        a = find_timeframe(i,num_of_dots,start_signal,end_signal)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files 