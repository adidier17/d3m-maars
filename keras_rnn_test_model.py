# Base CNN imports
import pandas as panda
import imageio
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.python import debug as tf_debug
from numbers import Number
import pdb

# For parameter tuning
import math

# Keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input, concatenate, BatchNormalization
from tensorflow.python.keras.layers import Reshape, MaxPooling2D, Lambda, TimeDistributed
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, CuDNNLSTM, ConvLSTM2D 
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import save_model, load_model, Model
from tensorflow.python.keras.utils import multi_gpu_model
from sklearn.preprocessing import MinMaxScaler


index = 0

# Convert imagelist into numpy array
imageArray = np.load('april_orthoimage_timeseries_images.npy')
torque = np.load('april_orthoimage_timeseries_torque.npy')
roll = np.load('april_orthoimage_timeseries_roll.npy')
pitch = np.load('april_orthoimage_timeseries_pitch.npy')

#imageArray = np.expand_dims(imageArray, axis=1)
real_image_array = []
real_roll_array = []
real_pitch_array = []
real_torque_array = []

sequence_length = 30
scaler = MinMaxScaler(feature_range=(0.0, 1.0))

for i in xrange(0,len(imageArray)-1,sequence_length):
    if (i > 29):
        sample_batch = imageArray[i-sequence_length:i]
        sample_batch = np.expand_dims(sample_batch, axis=3)
        sample_roll = roll[i-sequence_length:i]
        sample_roll = np.absolute(np.expand_dims(sample_roll, axis=1))
        scaler.fit(sample_roll)
        sample_pitch = (pitch[i-sequence_length:i])
        sample_pitch = np.absolute(np.expand_dims(sample_pitch, axis=1))
        scaler.fit(sample_pitch)
        real_image_array.append(sample_batch)
        real_roll_array.append(sample_roll)
        real_pitch_array.append(sample_pitch)
        sample_torque = torque[i-sequence_length:i]
        real_torque_array.append(sample_torque)

imageArray = np.asarray(real_image_array, dtype=np.float32)
roll = np.asarray(real_roll_array, dtype=np.float32)
pitch = np.asarray(real_pitch_array, dtype=np.float32)
torque = np.asarray(real_torque_array, dtype=np.float32)

index = len(torque)
print index
print len(imageArray)
print len(roll)
print len(pitch)
print imageArray.shape
print roll.shape
print pitch.shape
print torque.shape

testingSet = index-50

pathy = os.environ["HOME"]
path_best_model = '{}/08_23_keras_rnn_pose_best_model_august23.keras'.format(pathy)
best_accuracy = 1

model = load_model(path_best_model)
parallel_model = multi_gpu_model(model, gpus=4)
optimizer = Adam(lr=1e-3, clipnorm=1.0, clipvalue=5)
parallel_model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mse'])

test_inputs = {}
test_inputs['roll'] = roll[testingSet:index]
test_inputs['pitch'] = pitch[testingSet:index]
test_inputs['images'] = imageArray[testingSet:index]
score = parallel_model.predict(x=test_inputs)

print torque[testingSet:index]
print score


