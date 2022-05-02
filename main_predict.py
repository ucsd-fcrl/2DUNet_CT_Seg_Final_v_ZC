#!/usr/bin/env python

# this script uses pre-trained model to predict segmentation on new cases
# To run the script, in the terminal type python main_predict.py

# System
import argparse
import os

# Third Party
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.models import Model
from keras.layers import Input, \
                         Conv1D, Conv2D, Conv3D, \
                         MaxPooling1D, MaxPooling2D, MaxPooling3D, \
                         UpSampling1D, UpSampling2D, UpSampling3D, \
                         Reshape, Flatten, Dense
from keras.initializers import Orthogonal
from keras.regularizers import l2
import nibabel as nb
from sklearn.metrics import mean_squared_error  
import math
# Internal
from segcnn.generator import ImageDataGenerator
import segcnn.utils as ut
import dvpy as dv
import dvpy.tf_2d
import segcnn
import glob
import function_list as ff

cg = segcnn.Experiment()

########### Define the pre-trained model file ########
batch = 0
trial_name = 'final' # trial name
epoch = '020' # pick your epoch with highest validation accuracy
model_folder = os.path.join(cg.model_save_dir,'models_'+trial_name,'model_batch'+str(batch))
model_filename = 'model-'+trial_name + '-batch'+str(batch) +'-seg-' + epoch + '*'

model_files = ff.find_all_target_files([model_filename],model_folder)
assert len(model_files) == 1
print(model_files)
##################################################### 

########### Define the patient list #################
dv.section_print('Get patient list...')
### Define the patient list you will predict on (write your own version)
patient_list = ff.find_all_target_files(['*/*'],cg.image_data_dir)
print(patient_list)
#####################################################

#===========================================
dv.section_print('Loading Saved Weights...')
# BUILT U-NET
shape = cg.dim + (1,)
model_inputs = [Input(shape)]
model_outputs=[]
_, _, unet_output = dvpy.tf_2d.get_unet(cg.dim,
                                cg.num_classes,
                                cg.conv_depth,
                                layer_name='unet',
                                dimension =cg.unetdim,
                                unet_depth = cg.unet_depth,)(model_inputs[0])

model_outputs += [unet_output]
model = Model(inputs = model_inputs,outputs = model_outputs)
    
# Load weights
print(model_files[0])
model.load_weights(model_files[0],by_name = True)
# build generator
valgen = dv.tf_2d.ImageDataGenerator(
                  cg.unetdim,
                  input_layer_names=['input_1'],
                  output_layer_names=['unet'],
                  )


#===========================================
dv.section_print('Prediction...')

for p in patient_list:
  patient_class = os.path.basename(os.path.dirname(p))
  patient_id = os.path.basename(p)
  print(patient_class,patient_id)

  # define save folder
  save_folder = os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-try')
  ff.make_folder([os.path.dirname(os.path.dirname(save_folder)), os.path.dirname(save_folder), save_folder])

  # define all the CT images
  img_list = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],os.path.join(p,'img-nii-0.625')),2)

  # Prediction
  for img in img_list:
    
    time = ff.find_timeframe(img,2)

    if os.path.isfile(os.path.join(save_folder,'pred_s_' + str(time) + '.nii.gz')) == 1:
      print('already done')
      continue

    # define predict generator
    u_pred = model.predict_generator(valgen.predict_flow(np.asarray([img]),
      slice_num = cg.slice_num,
      batch_size = cg.slice_num,
      relabel_LVOT = cg.relabel_LVOT,
      shuffle = False,
      input_adapter = ut.in_adapt,
      output_adapter = ut.out_adapt,
      shape = cg.dim,
      input_channels = 1,
      output_channels = cg.num_classes,
      adapted_already = 0, 
      ),
      verbose = 1,
      steps = 1,)

    # save u_net segmentation
    u_gt_nii = nb.load(img) # load image for affine matrix
    u_pred = np.rollaxis(u_pred, 0, 3)
    u_pred = np.argmax(u_pred , axis = -1).astype(np.uint8)
    u_pred = dv.crop_or_pad(u_pred, u_gt_nii.get_fdata().shape)
    u_pred[u_pred == 3] = 4  # particular for LVOT
    u_pred = nb.Nifti1Image(u_pred, u_gt_nii.affine)
    save_file = os.path.join(save_folder,'pred_s_' + str(time) + '.nii.gz') #predicted segmentation
    nb.save(u_pred, save_file)