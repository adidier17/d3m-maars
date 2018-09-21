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
from tensorflow.python.keras.utils import plot_model, multi_gpu_model
from sklearn.preprocessing import MinMaxScaler
# Scikit Optimizer
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

#import horovod.keras as hvd

# Horovod: initialize Horovod.
#hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = str(hvd.local_rank())
#K.set_session(tf.Session(config=config))

index = 0
paths = ('merged_2018-04-09-14-28-11/orthoimage', 'merged_2018-05-10-14-30-07/orthoimage', 'merged_2018-07-13-13-03-27/orthoimage', 'merged_2018-07-13-13-39-03/orthoimage')

for (path in paths):
	imageArray = np.load('{}_orthoimage_timeseries_images.npy').format(path)
	torque = np.load('{}_orthoimage_timeseries_torque.npy').format(path)
	roll = np.load('{}_orthoimage_timeseries_roll.npy').format(path)
	pitch = np.load('{}_orthoimage_timeseries_pitch.npy').format(path)

	print len(imageArray);
	#imageArray = np.expand_dims(imageArray, axis=1)
	real_image_array = []
	real_roll_array = []
	real_pitch_array = []
	real_torque_array = []

	sequence_length = 30
	scaler = MinMaxScaler(feature_range=(0.001, 1.0))

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

	crossValidation = index-75
	testingSet = index-25

	#np.save('april_orthoimage_timeseries_images', imageArray)

	valid_inputs = {}
	valid_inputs['images'] = imageArray[crossValidation:testingSet]
	valid_inputs['roll'] = roll[crossValidation:testingSet]
	valid_inputs['pitch'] = pitch[crossValidation:testingSet]
	#validation_data = (imageArray[testingSet:crossValidation], torque[testingSet:crossValidation])
	validation_data = (valid_inputs, torque[crossValidation:testingSet])

	pathy = os.environ["HOME"]
	#path_best_model = '{}/08_23_keras_rnn_pose_best_model_mayonedata.keras'.format(pathy)
	path_best_model = '{}/09_14_keras_rnn_pose_model_reevaluation.keras'.format(pathy)
	#print path_best_model
	#best_hyperparameters = '{}/7_24_best_rnn_pose_hyperparameters'.format(pathy)
	best_accuracy = 1

	data_inputs = {}
	data_inputs['images'] = imageArray[0:crossValidation]
	data_inputs['roll'] = roll[0:crossValidation]
	data_inputs['pitch'] = pitch[0:crossValidation]

	print len(pitch[0:crossValidation])
	print len(torque[0:crossValidation])

	model = load_model(path_best_model)
	parallel_model = multi_gpu_model(model, gpus=4)
	optimizer = Adam(lr=1e-3, clipnorm=1.0, clipvalue=5)

	# Horovod: add Horovod Distributed Optimizer.
	#opt = hvd.DistributedOptimizer(opt)

	#parallel_model = model
	parallel_model.compile(optimizer=optimizer,
			  loss='mean_squared_error',
			  metrics=['mse'])
	history = parallel_model.fit(x=data_inputs,
			y=torque[0:crossValidation],
			epochs=500,
			batch_size=20,
			shuffle=False,
			validation_data=validation_data)

	path_new_model = '{}/09_17_keras_rnn_with_{}.keras'.format(pathy, path)
	model.save(path_new_model)


	#test_inputs = {}
	#test_inputs['roll'] = roll[testingSet:index]
	#test_inputs['pitch'] = pitch[testingSet:index]
	#test_inputs['images'] = imageArray[testingSet:index]
	#score = parallel_model.predict(x=test_inputs)

	#print score
	# Need to transpose to create row by column matrix
	#torque = np.transpose(torque)
	#torque = np.reshape(torque, (-1, 1))

	# Take the absolute value because we have negative torque values
	#torque = np.absolute(torque)

	#print torque[testingSet:index]
	#print score


