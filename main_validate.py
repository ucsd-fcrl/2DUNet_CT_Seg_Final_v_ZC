#!/usr/bin/env python

# if first time run this, in the terminal type chmod +x main_validate.py

# To run the script, in the terminal type ./main_validate.py --batch Number  
# this script will generate predicted segmentations on the validation batch

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


def validate(batch):
  
  ########### Define the pre-trained model weights you are going to use
  trial_name = 'final' # trial name
  epoch = '020' # pick your epoch with highest validation accuracy
  model_folder = os.path.join(cg.model_save_dir,'models_'+trial_name,'model_batch'+str(batch))
  model_filename = 'model-'+trial_name + '-batch'+str(batch) +'-seg-' + epoch + '*'

  model_files = ff.find_all_target_files([model_filename],model_folder)
  assert len(model_files) == 1
  print(model_files)
  ###########

  # deine the name of predicted segmentation file 
  seg_filename = 'pred_s_' 

  #===========================================
  dv.section_print('Calculating Image Lists...')

  partition_file_folder = 'final'

  imgs_list_tst=[np.load(os.path.join(cg.partition_dir,partition_file_folder,'img_list_'+str(p)+'.npy'),allow_pickle = True) for p in range(cg.num_partitions)]
  segs_list_tst=[np.load(os.path.join(cg.partition_dir,partition_file_folder,'seg_list_'+str(p)+'.npy'),allow_pickle = True) for p in range(cg.num_partitions)]
    
  if batch == None:
    raise ValueError('No batch was provided: wrong!')
    # print('pick all batches')
    # batch = 'all'
    # imgs_list_tst = np.concatenate(imgs_list_tst)
    # segs_list_tst = np.concatenate(segs_list_tst)
  else:
    imgs_list_tst = imgs_list_tst[batch]
    segs_list_tst = segs_list_tst[batch]
      
  print(imgs_list_tst.shape)
    
  #===========================================
  dv.section_print('Loading Saved Weights...')

  # Build the U-NET
  shape = cg.dim + (1,)
  model_inputs = [Input(shape)]
  model_outputs=[]
  _, _, unet_output = dvpy.tf_2d.get_unet(cg.dim,
                                    cg.num_classes,
                                    cg.conv_depth,
                                    layer_name='unet',
                                    dimension =cg.unetdim,
                                    unet_depth = cg.unet_depth,
                                   )(model_inputs[0])
  model_outputs += [unet_output]
  model = Model(inputs = model_inputs,outputs = model_outputs)
    
  # Load weights
  model.load_weights(model_files[0],by_name = True)
  
  #===========================================
  dv.section_print('Making Predictions...')
  # build data generator
  valgen = dv.tf_2d.ImageDataGenerator(
                cg.unetdim,
                input_layer_names=['input_1'],
                output_layer_names=['unet'],
                )


  # predict
  for img, seg in zip(imgs_list_tst, segs_list_tst):
    patient_id = os.path.basename(os.path.dirname(os.path.dirname(img)))
    patient_class = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img))))
    print(patient_class,patient_id,img,seg)

    # define save folder
    save_folder = os.path.join(cg.seg_data_dir,patient_class,patient_id,'seg-try')
    ff.make_folder([os.path.dirname(os.path.dirname(save_folder)), os.path.dirname(save_folder), save_folder])

    u_pred = model.predict_generator(valgen.flow(np.asarray([img]),np.asarray([seg]),
      slice_num = cg.slice_num,
      batch_size = cg.slice_num,
      relabel_LVOT = cg.relabel_LVOT,
      shuffle = False,
      input_adapter = ut.in_adapt,
      output_adapter = ut.out_adapt,
      shape = cg.dim,
      input_channels = 1,
      output_channels = cg.num_classes,
      adapted_already = cg.adapted_already,
      ),
      verbose = 1,
      steps = 1,)
                               
    # save u_net segmentation
    time = ff.find_timeframe(seg,1,'_')
    u_gt_nii = nb.load(os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-0.625',str(time)+'.nii.gz')) # load the image for affine matrix
    u_pred = np.rollaxis(u_pred, 0, 3)
    u_pred = np.argmax(u_pred , axis = -1).astype(np.uint8)
    u_pred = dv.crop_or_pad(u_pred, u_gt_nii.get_fdata().shape)
    u_pred[u_pred == 3] = 4  # use for LVOT only
    u_pred = nb.Nifti1Image(u_pred, u_gt_nii.affine)
    save_file = os.path.join(save_folder,seg_filename + str(time) + '.nii.gz') # predicted segmentation file
    nb.save(u_pred, save_file)
    break
     

      
# run prediction
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int)
  args = parser.parse_args()
  assert(0 <= args.batch < cg.num_partitions)

  validate(args.batch)