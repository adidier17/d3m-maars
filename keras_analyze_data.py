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

from multiprocessing import Pool, cpu_count

# Set image directory, only png's
paths = ('merged_2018-05-10-14-51-43/orthoimage')

# For the imageList, every index will represent
# the image counts for the samples contained under
# that index
imageList = []
correspondingTorque = []
correspondingRoll = []
correspondingPitch = []
oldsampleid = -1
index = 0

def readImage(filepath):
    im = imageio.imread(filepath)
    return np.dot(im[...,:3], [0.299, 0.587, 0.114])
 
# Walk through our directory and ready images into input vector
#for root, dirs, files in chain.from_iterable(os.walk(path) for path in paths):
for root, dirs, files in (os.walk(paths)):

    index = 0

    filteredFiles = []
    for filename in sorted(files):
	if (filename.endswith('.png')):
	 # Get the sample id
 	        sampleid = (filename.split('_'))[0]
                sampleid = (int)(re.sub('(s0*)', '', sampleid))
		if ((sampleid != oldsampleid) and sampleid > 510 and sampleid < 11733) :
			filteredFiles.append(os.path.join(root, filename))	

			if (sampleid != oldsampleid+1):
				print oldsampleid;
				print oldsampleid-sampleid;
	 	oldsampleid = sampleid
	 	index += 1

    print len(filteredFiles) 
    #for chunkIndex in xrange(0, len(sorted(filteredFiles)), 32):
    #for filename in sorted(files):
	
#	if (True) :
            # Read the current image and update the previous sample id var
#     imageList = pool.map(readImage, filteredFiles)
#     imageList = pool.map(rgb2gray, imageList)
	  
            # Get the sample id
            # sampleid = (filename.split('_'))[0]
            # sampleid = (int)(re.sub('(s0*)', '', sampleid))
	   
#	    if (sampleid != oldsampleid) :
	
#		    print gray[0][0].shape
#		    print gray[0][0]
#	            imageList.append(gray)
# 		    correspondingTorque.append(float(csvtorque[sampleid]))
#		    if (sampleid > 510 and index < 11733) :
#	            	correspondingRoll.append(float(csvroll[sampleid]))
#		    	correspondingPitch.append(float(csvpitch[sampleid]))
#       	    imageList.append(gray)
#                    	correspondingTorque.append(float(csvtorque[sampleid]))
#		    else: 
#	               	correspondingRoll.append(float(0))
#                	correspondingPitch.append(float(0))
#		    if (index > 5830):
#			print sampleid
#			print (float(csvroll[sampleid]))
#			print (float(csvpitch[sampleid]))
#			print float(csvtorque[sampleid])
#			correspondingTorque.append(float(csvtorque[sampleid]))
	         
#                    index += 1

#	    oldsampleid = sampleid
