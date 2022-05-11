## to run this in terminal, type:
# chmod +x pre_step1_set_defaults.sh
# . ./pre_ste1_set_defaults.sh   

## parameters
# define GPU you use
export CUDA_VISIBLE_DEVICES="0"

# pixel size, 
# for high resolution segmentation, set it to be 0.625, 
# for low resolution (used in 3D segmentation), set it to be 1.5
export CG_SPACING=0.625

# volume dimension: this should be set based on the dimension of every resampled volume
# use tool_check_image_size.py to check the dimension
export CG_CROP_X=352 # has to be divisible by 2^5 = 32
export CG_CROP_Y=352 # has to be divisible by 2^5 = 32
export CG_CROP_Z=256 # has to be divisible by 2^5 = 32

# define whether we are going to use pre-adapted (i.e., pre-processed) image 
# turn it on to facilitate the training
export CG_ADAPTED_ALREADY=1

# set the batch:
# in one batch, the model will read n slices from N patients
export CG_PATIENTS_IN_ONE_BATCH=2  # set it larger will slow down the training speed.
# n should be divisible by CG_CROP_Z
# let's set n = 16

# batch_size = N * n
export CG_BATCH_SIZE=32 # N = 2 patients in one batch, n = 16 slices from each patient


# set the number of classes in the output
export CG_NUM_CLASSES=4 # 2 for LV only, 3 for LV+LA, 4 for LV+LA+LVOT (recommend to use this one if you want to do LV segmentation, adding LA and LVOT segmentation will improve the LV segmentation performance), 10 for Left-sided, 14 for Right-sided
export CG_RELABEL_LVOT=1 # in the manual ground truth segmentation, LVOT = class 4; while in DL prediction, LVOT = class 3. so need to re-label the DL output LVOT

# set U-NET feature depth
export CG_CONV_DEPTH_MULTIPLIER=1 # default = 1 
export CG_FEATURE_DEPTH=8 
# depth=8 is up to 2^8 = 256, 9 is up to 512 and 10 is up to 1024 (1024 is reported to use in many papers), larger number means deeper U-net, slower training and larger GPU capacity required

# set learning epochs
export CG_EPOCHS=100
export CG_LR_EPOCHS=26 # the number of epochs for learning rate change 

export CG_SEED=1

# set number of partitioning groups, = 5 means we will do 5-fold cross-validation
export CG_NUM_PARTITIONS=5

# set data augmentation range
export CG_XY_RANGE="0.1"   #0.1
export CG_ZM_RANGE="0.1"  #0.1
export CG_RT_RANGE="10"   #15


export CG_NORMALIZE=0 #default = 0


# folders for Zhennong's dataset (change based on your folder paths)
export CG_MAIN_DATA_DIR="/Data/McVeighLabSuper/wip/zhennong/"   # main folder in NAS
export CG_IMAGE_DATA_DIR="${CG_MAIN_DATA_DIR}nii-images/"       # folder in NAS to save all nii images 
export CG_SEG_DATA_DIR="${CG_MAIN_DATA_DIR}predicted_seg/"      # folder in NAS to save all segmentations 
export CG_PARTITION_DIR="${CG_MAIN_DATA_DIR}partition/"         # folder in NAS to save partition files
export CG_LOCAL_DIR="/Data/local_storage/Zhennong/VR_Data_0.625/"   # folder in lab workstation's local storage to save all image files (it's faster to use local data when train the model)
export CG_MODEL_SAVE_DIR="/Data/ContijochLab/workspaces/zhennong/Volume_Rendering_segmentation/"  # folder in NAS to save all files of deep learning model weights
export CG_SPREADSHEET_DIR="/Data/McVeighLabSuper/wip/zhennong/spreadsheets/"    # folder in NAS to save all spreadsheets



