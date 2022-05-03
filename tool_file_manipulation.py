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

patient_list_raw = ff.find_all_target_files(['*/*'],cg.local_dir)  
patient_list = []
for p in patient_list_raw:
    if len(ff.find_all_target_files(['*.npy'],os.path.join(p,'img-nii-0.625-adapted'))) > 0:
        patient_list.append(p)
patient_list = np.asarray(patient_list)

for p in patient_list:
    patient_class = os.path.basename(os.path.dirname(p))
    patient_id = os.path.basename(p)
    # img_files = ff.find_all_target_files(['*.npy'],os.path.join(p,'img-nii-0.625-adapted'))
    # for i in img_files:
    #     a = np.load(i,allow_pickle = True)
    #     if (a.shape[0] != 352) or (a.shape[1] != 352) or (a.shape[2] != 256):
    #         print(patient_class,patient_id,a.shape)
    
    seg_files = ff.find_all_target_files(['*.npy'],os.path.join(p,'seg-pred-0.625-4classes-connected-retouch-adapted'))
    for s in seg_files:
        a = np.load(s,allow_pickle = True)
        if (a.shape[0] != 352) or (a.shape[1] != 352) or (a.shape[2] != 256) or (a.shape[3] !=4):
            print(patient_class,patient_id,s, a.shape)
        

    

# #delete
# folders = ff.find_all_target_files(['*/*/img-nii-1.5'],cg.local_dir)
# print(folders.shape)
# for f in folders:
#     shutil.rmtree(f)
# folders = ff.find_all_target_files(['*/*/img-nii-1.5'],cg.local_dir)
# print(folders.shape)


#  file transfer to octomore
# print(cg.local_dir)
# patient_list = ff.find_all_target_files(['Abnormal/CVC1907301359'],cg.image_data_dir)
# for p in patient_list:
#     patient_id = os.path.basename(p)
#     patient_class = os.path.basename(os.path.dirname(p))
#     print(patient_class,patient_id)
     
#     ff.make_folder([os.path.join(cg.local_dir,patient_class),os.path.join(cg.local_dir,patient_class,patient_id)])
    
#     img_list = ff.find_all_target_files(['*.npy'],os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-0.625-adapted'))
#     for i in img_list:
#         img_save_folder = os.path.join(cg.local_dir,patient_class,patient_id,'img-nii-0.625-adapted')
#         ff.make_folder([img_save_folder])
#         if os.path.isfile(os.path.join(img_save_folder,os.path.basename(i))) == 0:
#             shutil.copyfile(i,os.path.join(img_save_folder,os.path.basename(i)))
   
#     seg_list = ff.find_all_target_files(['*.npy'],os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-adapted'))
#     for s in seg_list:
#         seg_save_folder = os.path.join(cg.local_dir,patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-adapted')
#         ff.make_folder([seg_save_folder])
#         if os.path.isfile(os.path.join(seg_save_folder,os.path.basename(s))) == 0:
#             shutil.copyfile(s,os.path.join(seg_save_folder, os.path.basename(s)))

#     txt_file = ff.find_all_target_files(['ED_ES_frame_based_on_segmentation.txt'],os.path.join(cg.seg_data_dir,patient_class,patient_id))
#     if len(txt_file) > 0:
#         shutil.copyfile(txt_file[0],os.path.join(cg.local_dir,patient_class,patient_id,os.path.basename(txt_file[0])))
    
    

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
    

  