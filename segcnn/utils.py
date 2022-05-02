# System
import os

# Third Party
from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
import numpy as np
import nibabel as nb
import pandas as pd
import pylab as plt
from PIL import Image
import scipy.ndimage.interpolation as itp
from scipy.ndimage.measurements import center_of_mass

# Internal
import dvpy as dv
import segcnn
import os

cg = segcnn.Experiment()
#fs = segcnn.FileSystem(cg.base_dir, cg.data_dir,cg.local_dir)

adapt_size = (int(os.environ['CG_CROP_X']), int(os.environ['CG_CROP_Y']), int(os.environ['CG_CROP_Z']))

def in_adapt(x, target = adapt_size):
  x = nb.load(x).get_data()
  x = dv.crop_or_pad(x, target)
  x = np.expand_dims(x, axis = -1)
  return x

def relabel(x):
    # flip the label of LAA and LVOT
    x[x==3] = 10000 # original label of LAA
    x[x==4] = 3 # original label of LVOT
    x[x==10000] = 4
    return x


def out_adapt_raw(x, relabel_LVOT, target = adapt_size, n = cg.num_classes):
    x = nb.load(x).get_data()
    if relabel_LVOT == True:
        x = relabel(x)
    elif relabel_LVOT == False:
        a = 1
    else:
        raise ValueError('have not defined relabel_LVOT')
    x[x >= n] = 0
    x = dv.crop_or_pad(x, target)
    return x


def out_adapt(x,relabel_LVOT, target = adapt_size, n = cg.num_classes):
  return dv.one_hot(out_adapt_raw(x, relabel_LVOT, target), n)
