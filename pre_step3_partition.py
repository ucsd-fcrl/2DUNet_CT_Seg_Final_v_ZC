#!/usr/bin/env python

# this script will partition the patient cohort to do the cross-validation

# System
import os
import glob as gb
import pathlib as plib
import numpy as np
import dvpy as dv
import segcnn
import function_list as ff
cg = segcnn.Experiment()

np.random.seed(cg.seed)

# make the directories
os.makedirs(cg.partition_dir, exist_ok = True)

# define trial name
trial_name = 'final'

# Create a list of all patients. (write your own way to define the patient list)
patient_list_raw = ff.find_all_target_files(['*/*'],cg.local_dir)  
patient_list = []
for p in patient_list_raw:
    if len(ff.find_all_target_files(['*.npy'],os.path.join(p,'img-nii-0.625-adapted'))) > 0:
        patient_list.append(p)
patient_list = np.asarray(patient_list)
print(patient_list.shape,patient_list[0:20])

# Randomly Shuffle the patients.
np.random.shuffle(patient_list)
print(patient_list[0:20])

# # Split the list into `cg.num_partitions` (approximately) equal subgroups
partitions = np.array_split(patient_list, cg.num_partitions)

# # Save the partitions.
np.save(os.path.join(cg.partition_dir,'partitions_'+trial_name+'.npy'), partitions)

# create numpy arrays that save the list of paths for image files and segmentation files in each subgroup
# DL model will read these numpy arrays to know who are in training and who are in validation.
def create_img_lists(save_folder):
    partitions = np.load(os.path.join(cg.partition_dir,'partitions_'+trial_name+'.npy'),allow_pickle = True)

    for i, partition in enumerate(partitions):
        imgs_list = []
        segs_list = []
        for p in partition:
            # load the pre-defined time frames used for model training (ignore if you are going to use all the frames)
            EDES_info = open(os.path.join(p,'ED_ES_frame_based_on_segmentation.txt'),"r")
            Lines = EDES_info.readlines()
            num1 = [i for i, e in enumerate(Lines[0]) if e == '='][-1]; ED = int(Lines[0][num1+2:len(Lines[0])-1])
            num2 = [i for i, e in enumerate(Lines[1]) if e == '='][-1]; ES = int(Lines[1][num2+2:len(Lines[1])])
            t = [ED,ES]

            # find the file path of segmentation and image file of this/these time frame(s)
            # ground truth manual segmentation (Adapted)
            segs = ff.sort_timeframe(ff.find_all_target_files(['pred_s_'+str(ED)+'.npy', 'pred_s_'+str(ES)+'.npy'],os.path.join(p,'seg-pred-0.625-4classes-connected-retouch-adapted')),1,'_')
            # images (Adapated)
            imgs = ff.sort_timeframe(ff.find_all_target_files([str(ED)+'.npy', str(ES)+'.npy'],os.path.join(p,'img-nii-0.625-adapted')),1)
            
            assert(len(imgs) == len(segs))
            imgs_list.append(imgs[0]) ; imgs_list.append(imgs[1])
            segs_list.append(segs[0]) ; segs_list.append(segs[1])
        

        os.makedirs(os.path.join(cg.partition_dir, save_folder), exist_ok = True)
        np.save(os.path.join(cg.partition_dir,save_folder,'img_list_'+str(i)+'.npy'), np.asarray(imgs_list).reshape(-1))
        np.save(os.path.join(cg.partition_dir,save_folder,'seg_list_'+str(i)+'.npy'), np.asarray(segs_list).reshape(-1))
    

# main
create_img_lists(trial_name)

a = np.load(os.path.join(cg.partition_dir,trial_name,'img_list_0.npy'),allow_pickle = True)
print(a.shape)
b = np.load(os.path.join(cg.partition_dir,trial_name,'seg_list_0.npy'),allow_pickle = True)
print(b.shape)
