# U-Net-2D

## Description:
This repo enables the deep learning (DL) heart chamber segmentation of CT volumes in a 2D slice-by-slice fashion.

It includes scripts to pre-process the image for DL training, DL model training and DL model prediction.

## Install:
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up environment swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. <br />
- The docker file is included in this repo. <br />
- The code relies on dvpy python package, make sure you have installed the latest version of it. If not, do:
pip uninstall dvpy; pip install git+https://github.com/zhennongchen/dvpy.git#egg=dvpy <br />

## Data Preparation:
To train the model, make sure you have prepared your images and manual/ground truth segmentation. Here is a list of things you need to do for data preparation.
1. ```tool_resample_by_c3d.sh```: resample the CT volumes/manual segmentations to a uniform pixel dimension (required for U-Net input). default = 0.625mm^3
2. ```pre_step1_adapt_image.py```: pre-process the image for model *training*
3. ```pre_step2_partition.py```: randomly split the patient list to do *n-fold cross-validation*.

## Main Script:
- ```set_defaults.sh```: define the parameters&folders for DL experiments.
- ```main_train.py```: to train the model, using n-1 subsamples for training and the rest 1 subsample for validation.
- ```main_validate.py```: to validate the model for n-fold cross-validation
- ```main_predict.py```: to predict segmentation on new cases by trained DL model
    - in folder ```post_processing```: post_process the predicted segmentation (optional), mainly to exclude the disconnected parts. first run ```tool_exclude_disconnectivity.m```, and then run ```tool_mat_to_nii.ipynb``` to turn mat file to nii file.

## Additional Guidelines
see comments in the script

Please contact zhc043@eng.ucsd.edu or chenzhennong@gmail.com for any further questions.
