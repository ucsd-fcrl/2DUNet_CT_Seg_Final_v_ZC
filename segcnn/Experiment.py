# System
import os

class Experiment():

  def __init__(self):
  
    # Number of partitions in the crossvalidation.
    self.num_partitions = int(os.environ['CG_NUM_PARTITIONS'])

    # define whether we are going to use pre-adapted image
    self.adapted_already = int(os.environ['CG_ADAPTED_ALREADY'])
    
    # Dimension of padded input, for training.
    self.dim = (int(os.environ['CG_CROP_X']), int(os.environ['CG_CROP_Y']))
    #self.dim = (int(os.environ['CG_CROP_X']), int(os.environ['CG_CROP_Y']),int(os.environ['CG_CROP_Z']))
    self.slice_num = int(os.environ['CG_CROP_Z'])

    self.unetdim = len(self.dim)
  
    # Seed for randomization.
    self.seed = int(os.environ['CG_SEED'])
  
    # Number of Classes (Including Background)
    self.num_classes = int(os.environ['CG_NUM_CLASSES'])

    # Whether relabel of LVOT is necessary
    if int(os.environ['CG_RELABEL_LVOT']) == 1:
      self.relabel_LVOT = True
    else:
      self.relabel_LVOT = False

    # normalize
    if int(os.environ['CG_NORMALIZE']) == 1:
      self.normalize = True
    else:
      self.normalize = False
      
  
    # UNet Depth
    self.unet_depth = 5
  
    # Depth of convolutional feature maps
    self.conv_depth_multiplier = int(os.environ['CG_CONV_DEPTH_MULTIPLIER'])
    self.ii = int(os.environ['CG_FEATURE_DEPTH'])
    self.conv_depth = [2**(self.ii-4),2**(self.ii-3),2**(self.ii-2),2**(self.ii-1),2**(self.ii),2**(self.ii),
                      2**(self.ii-1),2**(self.ii-2),
                      2**(self.ii-3),2**(self.ii-4),2**(self.ii-4)]
    #self.conv_depth = [16, 32, 64, 128, 256, 256, 128, 64, 32, 16, 16]
    self.conv_depth = [self.conv_depth_multiplier*x for x in self.conv_depth]
    print(self.conv_depth)
  
    assert(len(self.conv_depth) == (2*self.unet_depth+1))
  
    # How many images should be processed in each batch?
    self.batch_size = int(os.environ['CG_BATCH_SIZE'])

    # How many cases should be read in each batch?
    self.patients_in_one_batch = int(os.environ['CG_PATIENTS_IN_ONE_BATCH'])
  
    # Translation Range
    self.xy_range = float(os.environ['CG_XY_RANGE'])
  
    # Scale Range
    self.zm_range = float(os.environ['CG_ZM_RANGE'])

    # Rotation Range
    self.rt_range=float(os.environ['CG_RT_RANGE'])
  
    # Should Flip
    self.flip = False

    # SPACING:
    self.spacing = float(os.environ['CG_SPACING'])

    # Total number of epochs to train
    self.epochs = int(os.environ['CG_EPOCHS'])

    # Number of epochs to train before decreasing learning rate
    self.lr_epochs = int(os.environ['CG_LR_EPOCHS'])


    # # folders
    # for VR dataset
    self.main_data_dir = os.environ['CG_MAIN_DATA_DIR']
    self.image_data_dir = os.environ['CG_IMAGE_DATA_DIR']
    self.seg_data_dir = os.environ['CG_SEG_DATA_DIR']
    self.spreadsheet_dir = os.environ['CG_SPREADSHEET_DIR']
    self.partition_dir = os.environ['CG_PARTITION_DIR']
    self.local_dir = os.environ['CG_LOCAL_DIR']
    self.model_save_dir = os.environ['CG_MODEL_SAVE_DIR']





    # # for Davis dataset:
    # # raw data
    # self.raw_dir = os.environ['CG_RAW_DIR']

    # # Input data (annotations).
    # self.base_dir = os.environ['CG_INPUT_DIR']

    # # Output data (models, image lists).
    # self.data_dir = os.environ['CG_DERIVED_DIR']

    # # data saved in the octomore local
    # self.local_dir = os.environ['CG_LOCAL_DIR']

    # # folder of FC's NAS drive
    # self.fc_dir = os.environ['CG_FCNAS_DIR']

    # # Input directory names
    # self.img_dir = os.path.normpath('img-nii-0.625-adapted/')
    # self.seg_dir = os.path.normpath('seg-nii-0.625-adapted/')  
    
    # self.matrix_dir=os.path.normpath('matrix/')
  

    # # Output directory name
    # self.pred_dir = os.path.normpath('seg-pred')
    # self.pred_dir_2=os.path.normpath('matrix-pred')

  
