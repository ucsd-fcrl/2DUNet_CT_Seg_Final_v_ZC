#!/usr/bin/env python

# this script trains the deep learning (DL) model to do segmentation.

# if first time run this, in the terminal type chmod +x main_train.py

# To run the script, in terminal, type ./main_train.py --batch Number
# ./main_train.py --batch 0 means you pick the first (0th) group as the validation 
# ./main_train.py --batch means you don't define a batch (or you define it as None), and then the model will train and validate on all cases.

# System
import argparse
import os

# Third Party
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import CSVLogger
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Input, \
                         Conv1D, Conv2D, Conv3D, \
                         MaxPooling1D, MaxPooling2D, MaxPooling3D, \
                         UpSampling1D, UpSampling2D, UpSampling3D, \
                         Reshape, Flatten, Dense
from keras.layers.merge import concatenate, multiply
from keras.initializers import Orthogonal
from keras.regularizers import l2
from keras.layers.merge import concatenate, multiply

import tensorflow as tf

# Internal
import function_list as ff
import segcnn.utils as ut
import dvpy as dv
import dvpy.tf_2d
import segcnn

cg = segcnn.Experiment()

K.set_image_dim_ordering('tf')  # Tensorflow dimension ordering in this code

# Allow Dynamic memory allocation.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def train(batch):
    print(cg.dim)
    print('BATCH_SIZE = ',cg.batch_size)
    
    # define a name of your trial
    trial_name = 'final' 

    # define partition file
    partition_file_folder = 'final'

    # define hdf5 file save folder (hdf5 file is the model weights file)
    print(cg.model_save_dir)
    weight_file_save_folder = os.path.join(cg.model_save_dir,'models_'+trial_name)
    ff.make_folder([weight_file_save_folder,os.path.join(cg.model_save_dir,'logs')])

    #===========================================
    dv.section_print('Calculating Image Lists...')
    
    # obtain image list and segmentation list in training and validation
    imgs_list_trn=[np.load(os.path.join(cg.partition_dir,partition_file_folder,'img_list_'+str(p)+'.npy'),allow_pickle = True) for p in range(cg.num_partitions)]
    segs_list_trn=[np.load(os.path.join(cg.partition_dir,partition_file_folder,'seg_list_'+str(p)+'.npy'),allow_pickle = True) for p in range(cg.num_partitions)]
   

    imgs_list_tst = imgs_list_trn.pop(batch)
    segs_list_tst = segs_list_trn.pop(batch)
      

    imgs_list_trn = np.concatenate(imgs_list_trn)
    segs_list_trn = np.concatenate(segs_list_trn)

    len_list=[len(imgs_list_trn),len(segs_list_trn),len(imgs_list_tst),len(segs_list_tst)]
    print(len_list,segs_list_trn[0])

    #===========================================
    dv.section_print('Creating and compiling model...')
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
    opt = Adam(lr = 1e-4) 
    losses={'unet':'categorical_crossentropy'} 
    model.compile(optimizer= opt, 
                 loss= losses,
                 metrics= {'unet':'acc',})
    
    #======================
    dv.section_print('Fitting model...')
   
    # define the name of each model weight file
    if batch is None:
      model_name = 'model-'+trial_name+'-batch_all-seg'
      model_fld = 'model_batch_all'
    else:
      model_name = 'model-'+trial_name+'-batch'+str(batch)+'-seg'
      model_fld = 'model_batch'+str(batch)
    filename = model_name +'-{epoch:03d}.hdf5'
    filepath=os.path.join(weight_file_save_folder,model_fld,filename)   
    ff.make_folder([os.path.dirname(filepath)])
 
    # set callbacks
    csv_logger = CSVLogger(os.path.join(cg.model_save_dir, 'logs',  model_name + '_training-log' + '.csv')) # log will automatically record the train_accuracy/loss and validation_accuracy/loss in each epoch
    callbacks = [csv_logger,
                 ModelCheckpoint(filepath,          
                                 monitor='val_loss',
                                 save_best_only=False, # set True if only save model weight file when "monitor" gets improved, set False if save every epoch 
                                 ),
                 LearningRateScheduler(dv.learning_rate_step_decay2),   # learning decay
                ]
   
    # training data generator (with data augmentation)
    datagen = dv.tf_2d.ImageDataGenerator(
        cg.unetdim,  # Dimension of input image
        input_layer_names = ['input_1'],
        output_layer_names = ['unet'],
        translation_range=cg.xy_range,  # randomly shift images vertically (fraction of total height)
        rotation_range=cg.rt_range,  # randomly rotate images in the range (degrees, 0 to 180)
        scale_range=cg.zm_range,
        flip=cg.flip,)
    
    datagen_flow = datagen.flow(imgs_list_trn,
      segs_list_trn,
      slice_num = cg.slice_num,
      batch_size = cg.batch_size,
      patients_in_one_batch = cg.patients_in_one_batch,
      relabel_LVOT = cg.relabel_LVOT, 
      shuffle = True,
      input_adapter = ut.in_adapt,
      output_adapter = ut.out_adapt,
      shape = cg.dim,
      input_channels = 1,
      output_channels = cg.num_classes,
      augment = True, # only True in the training process to randomly translate, rotate and scale the image.
      normalize = cg.normalize,
      adapted_already = cg.adapted_already, # True when you already did the image adaption in the pre-processing step.
      )

    # validation data generator (no data augmentation)
    valgen = dv.tf_2d.ImageDataGenerator(
        cg.unetdim, 
        input_layer_names=['input_1'],
        output_layer_names=['unet'],
        )

    valgen_flow = valgen.flow(imgs_list_tst,
      segs_list_tst,
      slice_num = cg.slice_num,
      batch_size = cg.batch_size,
      patients_in_one_batch = 1, # set as 1 in validation
      relabel_LVOT = cg.relabel_LVOT,
      shuffle = True,
      input_adapter = ut.in_adapt,
      output_adapter = ut.out_adapt,
      shape = cg.dim,
      input_channels = 1,
      output_channels = cg.num_classes,
      normalize = cg.normalize,
      adapted_already = cg.adapted_already,
      )

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen_flow,
                        steps_per_epoch = imgs_list_trn.shape[0] * cg.slice_num // cg.batch_size,
                        epochs = cg.epochs,
                        workers = 1,
                        validation_data = valgen_flow,
                        validation_steps = imgs_list_tst.shape[0] * cg.slice_num // cg.batch_size,
                        callbacks = callbacks,
                        verbose = 1,
                       )

    
if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int)
  args = parser.parse_args()

  if args.batch is not None:
    assert(0 <= args.batch < cg.num_partitions)

  train(args.batch)
  