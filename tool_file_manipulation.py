#!/usr/bin/env python

## 
# this script can do file copy and removal
##

import os
import numpy as np
import function_list as ff
import shutil
import pandas as pd
import segcnn

cg = segcnn.Experiment()

# #delete
# folders = ff.find_all_target_files(['*/*/img-nii-1.5'],cg.local_dir)
# print(folders.shape)
# for f in folders:
#     shutil.rmtree(f)
# folders = ff.find_all_target_files(['*/*/img-nii-1.5'],cg.local_dir)
# print(folders.shape)

# file transfer intra-NAS
# save_folder = os.path.join('/Data/McVeighLabSuper/wip/zhennong/2020_after_Junes','downsample-nii-images-1.5mm')
# patient = ff.find_all_target_files(['Abnormal/*','Normal/*'],cg.image_data_dir)
# for p in patient:
#     patient_class = os.path.basename(os.path.dirname(p))
#     patient_id = os.path.basename(p)
    

#     image_files = ff.find_all_target_files(['0.nii.gz'],os.path.join(p,'img-nii-1.5'))
#     for img in image_files:
#         destination = os.path.join(save_folder,patient_class,patient_id,'img-nii-1.5',os.path.basename(img))
        
#         ff.make_folder([os.path.dirname(os.path.dirname(destination)),os.path.dirname(destination)])
#         if os.path.isfile(destination) == 0:
#             print(patient_class,patient_id)
#             shutil.copy(img,destination)

# file transfer into octomore
# save_folder = cg.local_dir
# patient = ff.find_all_target_files(['Abnormal/*','Normal/*'],cg.image_data_dir)
# for p in patient:
#     patient_class = os.path.basename(os.path.dirname(p))
#     patient_id = os.path.basename(p)
    

#     image_files = ff.find_all_target_files(['*.nii.gz'],os.path.join(p,'img-nii-0.625'))
#     for img in image_files:
#         destination = os.path.join(save_folder,patient_class,patient_id,'img-nii-0.625',os.path.basename(img))
        
#         ff.make_folder([os.path.dirname(os.path.dirname(destination)),os.path.dirname(destination)])
#         if os.path.isfile(destination) == 0:
#             print(patient_class,patient_id)
#             shutil.copy(img,destination)


# file transfer to octomore
print(cg.local_dir)
patient_list = ff.find_all_target_files(['*/*'],cg.image_data_dir)
for p in patient_list:
    patient_id = os.path.basename(p)
    patient_class = os.path.basename(os.path.dirname(p))
    print(patient_class,patient_id)
     
    ff.make_folder([os.path.join(cg.local_dir,patient_class),os.path.join(cg.local_dir,patient_class,patient_id)])
    
    img_list = ff.find_all_target_files(['*.npy'],os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-0.625-adapted'))
    for i in img_list:
        img_save_folder = os.path.join(cg.local_dir,patient_class,patient_id,'img-nii-0.625-adapted')
        ff.make_folder([img_save_folder])
        if os.path.isfile(os.path.join(img_save_folder,os.path.basename(i))) == 0:
            shutil.copyfile(i,os.path.join(img_save_folder,os.path.basename(i)))
   
    seg_list = ff.find_all_target_files(['*.npy'],os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-adapted'))
    for s in seg_list:
        seg_save_folder = os.path.join(cg.local_dir,patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-adapted')
        ff.make_folder([seg_save_folder])
        if os.path.isfile(os.path.join(seg_save_folder,os.path.basename(s))) == 0:
            shutil.copyfile(s,os.path.join(seg_save_folder, os.path.basename(s)))

    txt_file = ff.find_all_target_files(['ED_ES_frame_based_on_segmentation.txt'],os.path.join(cg.seg_data_dir,patient_class,patient_id))
    if len(txt_file) > 0:
        shutil.copyfile(txt_file[0],os.path.join(cg.local_dir,patient_class,patient_id,os.path.basename(txt_file[0])))
    
    

# compress
# patient_list = ff.get_patient_list_from_csv(os.path.join(cg.spreadsheet_dir,'Final_patient_list_include.csv'))
# print(len(patient_list))
# for p in patient_list:
#     print(p[0],p[1])
#     f1 = os.path.join(cg.seg_data_dir,p[0],p[1],'seg-nii-1.5-upsample-retouch-adapted-LV')
#     f2 = os.path.join(cg.seg_data_dir,p[0],p[1],'seg-nii-1.5-upsample-retouch-adapted')
#     shutil.make_archive(os.path.join(cg.seg_data_dir,p[0],p[1],'seg-nii-1.5-upsample-retouch-adapted-LV'),'zip',f1)
#     shutil.make_archive(os.path.join(cg.seg_data_dir,p[0],p[1],'seg-nii-1.5-upsample-retouch-adapted'),'zip',f2)
#     shutil.rmtree(f1)
#     shutil.rmtree(f2)

# change file name
# patient_list = ff.find_all_target_files(['*/*'],cg.image_data_dir)

# for p in patient_list:
#     files = ff.find_all_target_files(['img-nii/img_*.nii.gz'],p)
#     print(p,len(files))

#     if len(files) == 0:
#         print('no data. skip')
#     else:
#         for f in files:
#             tf = ff.find_timeframe(f,2,'_')
#             if os.path.isfile(os.path.join(p,'img-nii',str(tf)+'.nii.gz')) == 0:
#                 shutil.copy(f,os.path.join(p,'img-nii',str(tf)+'.nii.gz'))
        
#         for f in files:
#             os.remove(f)
    

  