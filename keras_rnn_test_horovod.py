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

import horovod.keras as hvd

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

index = 0

# Convert imagelist into numpy array
imageArray = np.load('merged_2018-08-29-12-36-26_orthoimage_timeseries_images.npy')
torque = np.load('merged_2018-08-29-12-36-26_orthoimage_timeseries_torque.npy')
roll = np.load('merged_2018-08-29-12-36-26_orthoimage_timeseries_roll.npy')
pitch = np.load('merged_2018-08-29-12-36-26_orthoimage_timeseries_pitch.npy')

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

crossValidation = index-10
testingSet = index-5

valid_inputs = {}
valid_inputs['images'] = imageArray[crossValidation:testingSet]
valid_inputs['roll'] = roll[crossValidation:testingSet]
valid_inputs['pitch'] = pitch[crossValidation:testingSet]   
validation_data = (valid_inputs, torque[crossValidation:testingSet])

pathy = os.environ["HOME"]
path_best_model = '{}/08_23_keras_rnn_pose_best_model_mayonedata.keras'.format(pathy)
best_accuracy = 1

data_inputs = {}
data_inputs['images'] = imageArray[0:crossValidation]
data_inputs['roll'] = roll[0:crossValidation]
data_inputs['pitch'] = pitch[0:crossValidation]

epochs = int(math.ceil(12.0 / hvd.size()))

print len(pitch[0:crossValidation])
print len(torque[0:crossValidation])

model = load_model(path_best_model)
opt = Adam(lr=1.0 * hvd.size(), clipnorm=1.0, clipvalue=5)

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

#parallel_model = model
model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['mse'])

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

history = model.fit(x=data_inputs,
		        y=torque[0:crossValidation],
                epochs=epochs,
                batch_size=128,
		        callbacks=callbacks,
                validation_data=validation_data)

path_new_model = '{}/08_31_keras_rnn_pose_horovod_best_model_merged_2018-08-29-12-36-26.keras'.format(pathy)
model.save(path_new_model)


