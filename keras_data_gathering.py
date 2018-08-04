# Base CNN imports
import pandas as panda
import imageio
import os
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug
from numbers import Number
import pdb
import re
# For parameter tuning
import math

# Keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input, BatchNormalization
from tensorflow.python.keras.layers import Reshape, MaxPooling2D, Lambda
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, ConvLSTM2D
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import save_model, load_model
from tensorflow.contrib import rnn
from itertools import chain

# Scikit Optimizer
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

# Hyperparameter tuning
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                         name='learning_rate')


dim_num_dense_layers = Integer(low=10, high=100, name='num_dense_layers')

dim_num_conv_layers = Integer(low=1, high=4, name='num_conv_layers')

dim_kernel_size = Integer(low=3, high=10, name='kernel_size')

dim_num_filters = Integer(low=16, high=64, name='num_filters')

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_conv_layers,
              dim_kernel_size,
              dim_num_filters]

default_parameters = [1e-3, 25, 2, 5, 32]

# We know that MNIST images are 128 pixels in each dimension.
img_width = 128

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_width * img_width

# Tuple with height and width of images used to reshape arrays.
# This is used for plotting the images.
img_shape = (img_width, img_width)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 0

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = (img_width, img_width)


# Set image directory, only png's
paths = ('april/orthoimage', 'mayone/orthoimage', 'maytwo/orthoimage')

# For the imageList, every index will represent
# the image counts for the samples contained under
# that index
imageList = []
correspondingTorque = []
correspondingRoll = []
correspondingPitch = []
oldsampleid = -1
index = 0

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# Walk through our directory and ready images into input vector
#for root, dirs, files in chain.from_iterable(os.walk(path) for path in paths):
for root, dirs, files in chain.from_iterable(os.walk(path) for path in paths):

    prefix_fname = root.replace('/', '_')
    imagePath = '{}_timeseries_images'.format(prefix_fname)
    torquePath = '{}_timeseries_torque'.format(prefix_fname)
    pitchPath = '{}_timeseries_pitch'.format(prefix_fname)
    rollPath = '{}_timeseries_roll'.format(prefix_fname)

    index = 0
    # Get the torques for this new directory of mages
    tp=os.path.join(root,'motor_DRF.csv')
    pose=os.path.join(root,'pose.csv')
    motorDRF = panda.read_csv(tp, sep="\t")
    pose = panda.read_csv(pose, sep="\t")

    csvtimestamp = motorDRF['TIMESTAMP'].values
    csvtorque = motorDRF['EST_POWER'].values
    csvroll = pose['ROLL'].values
    csvpitch = pose['PITCH'].values
    print len(csvtorque)
    print len(csvroll)
    for filename in sorted(files):
	if (filename.endswith('png')) :
	    p=os.path.join(root,filename)
            # Read the current image and update the previous sample id var
            #im = imageio.imread(p)
            #gray = rgb2gray(im)
	  
            # Get the sample id
            sampleid = (filename.split('_'))[0]
            sampleid = (int)(re.sub('(s0*)', '', sampleid))
	   
	    if (sampleid != oldsampleid) :
		    print p
	
#		    print gray[0][0].shape
#		    print gray[0][0]
	            #imageList.append(gray)
 		    correspondingTorque.append(float(csvtorque[sampleid]))
#		    if (index > 0) :
#	            	correspondingRoll.append(float(csvroll[sampleid]))
#		    	correspondingPitch.append(float(csvpitch[sampleid]))
#		    else: 
#                	correspondingRoll.append(float(0))
#                	correspondingPitch.append(float(0))
#		    if (index > 5830):
#			print sampleid
#			print (float(csvroll[sampleid]))
#			print (float(csvpitch[sampleid]))
#			print float(csvtorque[sampleid])
#			correspondingTorque.append(float(csvtorque[sampleid]))
	         
#                    index += 1

	    oldsampleid = sampleid
            index += 1

    # Convert imagelist into numpy array
    #imageArray = np.asarray(imageList, dtype=np.float32)

    #np.save(imagePath, imageArray)

    # Convert our torquelist to numpuy array
    torque = np.asarray(correspondingTorque, dtype=np.float32)

    # Need to transpose to create row by column matrix
    torque = np.transpose(torque)
    torque = np.reshape(torque, (-1, 1))

    # Take the absolute value because we have negative torque values
    torque = np.absolute(torque)

    np.save(torquePath, torque)

#    roll = np.asarray(correspondingRoll, dtype=np.float32)
#    pitch = np.asarray(correspondingPitch, dtype=np.float32)
#    print len(roll)
#    print len(pitch)

#    np.save(rollPath, roll)
#    np.save(pitchPath, pitch)

