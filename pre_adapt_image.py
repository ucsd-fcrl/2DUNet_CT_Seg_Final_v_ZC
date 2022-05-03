#!/usr/bin/env python

# this script will adapt the image/segmentation (crop/pad + normalize + relabel + resize...) for the U-Net *Training*

import os
import numpy as np
import nibabel as nb
import dvpy as dv
import segcnn
import segcnn.utils as ut
import function_list as ff
cg = segcnn.Experiment()

# define patient list (write your own way to define patient list )
# patient_list = ff.get_patient_list_from_csv(os.path.join(cg.spreadsheet_dir,'Final_patient_list_include.csv'))
patient_list = ff.find_all_target_files(['*/*'],cg.seg_data_dir)
print(len(patient_list))

for p in patient_list:
    patient_id = os.path.basename(p)
    patient_class = os.path.basename(os.path.dirname(p))
    print(patient_class,patient_id)

    # find ED and ES frame (ignore if you are going to use all time frames)
    # this is the path for ground truth manual/refined segmentation in each CT study
    seg_file_list = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],
                    os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch')),2,'_') 
    
    if len(seg_file_list) < 5:
        print('no refined segmentation, skip this patient')
        continue

    if os.path.isfile(os.path.join(cg.seg_data_dir,patient_class,patient_id,'ED_ES_frame_based_on_segmentation.txt')) == 0:
        ED, ES = ff.find_ED_ES(seg_file_list)
        print(ED, ES)
        t_file = open(os.path.join(cg.seg_data_dir,patient_class,patient_id,'ED_ES_frame_based_on_segmentation.txt'),"w+")
        t = t_file.write("ED = %d\nES = %d" % (ED,ES))
        t_file.close()
    else:
        EDES_info = open(os.path.join(cg.seg_data_dir,patient_class,patient_id,'ED_ES_frame_based_on_segmentation.txt'),"r")
        Lines = EDES_info.readlines()
        num1 = [i for i, e in enumerate(Lines[0]) if e == '='][-1]; ED = int(Lines[0][num1+2:len(Lines[0])-1])
        num2 = [i for i, e in enumerate(Lines[1]) if e == '='][-1]; ES = int(Lines[1][num2+2:len(Lines[1])])
        print(ED,ES)



    # adapt input image - CT volume
    save_folder = os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-0.625-adapted')
    ff.make_folder([save_folder])

    img_list = ff.find_all_target_files(['*.nii.gz'],os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-0.625'))
    for i in img_list:
        time = ff.find_timeframe(i,2)
        if time == ED or time == ES:
            if os.path.isfile(os.path.join(save_folder,str(time)+'.npy')) == 1:
                print('already done, skip')
            else:
                x = ut.in_adapt(i)
                if cg.normalize == 1:
                    print('normalize is done')
                    x = ut.normalize_image(x)
                np.save(os.path.join(save_folder,str(time)+'.npy'),x)
    
    
    # adapt manual segmentation
    save_folder = os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-adapted')
    ff.make_folder([save_folder])

    seg_list = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch')),2,'_')
    
    for s in seg_list:
        time = ff.find_timeframe(s,2,'_')
        if time == ED or time == ES:
            if os.path.isfile(os.path.join(save_folder,'pred_s_'+str(time)+'.npy')) == 1:
                print('already done, skip')
            else:
                y = ut.out_adapt(s,cg.relabel_LVOT)
                # print(y.shape,np.unique(y))
                np.save(os.path.join(save_folder,'pred_s_'+str(time)+'.npy'),y)
    





        
        







