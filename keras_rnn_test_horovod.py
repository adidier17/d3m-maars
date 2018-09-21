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
# Scikit Optimizer
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

import horovod.keras as hvd

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))


# Hyperparameter tuning
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                         name='learning_rate')


dim_num_dense_layers = Integer(low=50, high=250, name='num_dense_layers')

dim_num_conv_layers = Integer(low=1, high=2, name='num_conv_layers')

dim_kernel_size = Integer(low=3, high=10, name='kernel_size')

dim_num_filters = Integer(low=16, high=64, name='num_filters')

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_conv_layers,
              dim_kernel_size,
              dim_num_filters]

def log_dir_name(learning_rate, num_dense_layers,
                 num_conv_layers, kernel_size, num_filters):

    # The dir-name for the TensorBoard log-dir.
    s = "./725_keras_rnn_test_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}_{4}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_conv_layers,
                       kernel_size,
                       num_filters)

    return log_dir

default_parameters = [1e-3, 50, 1, 5, 32]

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

index = 0

# Convert imagelist into numpy array
#imageArray = np.asarray(imageList, dtype=np.float32)
#np.save('grayscale_image_array_timeseries.out', imageArray)
imageArray = np.load('merged_2018-08-29-12-36-26_orthoimage_timeseries_images.npy')
torque = np.load('merged_2018-08-29-12-36-26_orthoimage_timeseries_torque.npy')
roll = np.load('merged_2018-08-29-12-36-26_orthoimage_timeseries_roll.npy')
pitch = np.load('merged_2018-08-29-12-36-26_orthoimage_timeseries_pitch.npy')

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
		real_torque_array.append(torque[i])

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

#np.save('april_orthoimage_timeseries_images', imageArray)

valid_inputs = {}
valid_inputs['images'] = imageArray[crossValidation:testingSet]
valid_inputs['roll'] = roll[crossValidation:testingSet]
valid_inputs['pitch'] = pitch[crossValidation:testingSet]
#validation_data = (imageArray[testingSet:crossValidation], torque[testingSet:crossValidation])
validation_data = (valid_inputs, torque[crossValidation:testingSet])

def create_model(learning_rate, num_dense_layers,
                 num_conv_layers, kernel_size, num_filters):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_conv_layers:   Number of conv layers.
    kernel_size:       kernel_size function for conv layers
    num_filters        Number of Filters
    """
    

    # Add an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.
    input_first = (Input(shape=(sequence_length, img_width, img_width, 1), name='images'))
    input_next = input_first
    # model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x)))

    # for i in range(num_conv_layers):
    #     name = 'layer_conv_{0}'.format(i+1)
    #     # First convolutional layer.
    #     # There are many hyper-parameters in this layer, but we only
    #     # want to optimize the kernel_size-function in this example.

    #     factor = 1
    #     filters = num_filters
    #     if (i == 0):
    #         factor = 1
    #         filters = 1
    #     else:
    #         factor = 2*i

    filters = 0
    for i in range(num_conv_layers):
        name = 'layer_convlstm_{0}'.format(i+1)
        
        if (i == 0):
            filters = 1
        else:
            filters = num_filters

        input_next = (ConvLSTM2D(kernel_size=kernel_size, input_shape=(sequence_length, img_width, img_width, filters), strides=1, 
                        filters=num_filters, padding='same', activation="relu", name=name,
                        return_sequences=True))(input_next)
        input_next = (BatchNormalization(input_shape=(sequence_length, img_width, img_width, num_filters)))(input_next)

    #input_shape=(img_width, img_width, num_filters)

    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    #input_next = (Reshape((100, img_width*img_width*num_filters), input_shape=(100, img_width, img_width, num_filters)))(input_next)
    input_next = (TimeDistributed(Flatten()))(input_next)
    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    # for i in range(num_dense_layers):
    #     # Name of the layer. This is not really necessary
    #     # because Keras should give them unique names.
    #     name = 'layer_dense_{0}'.format(i+1)

    #     # Add the dense / fully-connected layer to the model.
    #     # This has two hyper-parameters we want to optimize:
    #     # The number of nodes
    #     # Add a fully-connected / dense layer
    #     model.add(Dense((num_filters*img_width*img_width/(4**num_conv_layers)),
    #                     activation="relu", name=name,input_shape=(None, (num_filters*img_width*img_width/(4**num_conv_layers)))))

    # Last fully-connected / dense layer with linear-activation
    # for use in classification. 
    input_next = (TimeDistributed(Dense(1, activation="linear"), input_shape=(sequence_length, num_filters*img_width*img_width)))(input_next)

    def scaleDown(x):
        import tensorflow as tf
        return tf.div(
                tf.subtract(
                    x, 
                    tf.reduce_min(x)
                ), 
                tf.subtract(
                    tf.reduce_max(x), 
                    tf.reduce_min(x)
                ) 
        )
 
    input_next = (Lambda(scaleDown))(input_next)
    roll = Input(shape=(sequence_length,1), name='roll')
    pitch = Input(shape=(sequence_length,1), name='pitch')
#    roll = (Lambda(scaleDown))(roll)
#    pitch = (Lambda(scaleDown))(pitch)  
    x = concatenate([input_next, roll, pitch], axis=2)
    input_next = CuDNNLSTM(3, input_shape=(sequence_length, 3), name="LSTM_aggregation")(x)
    predictions = (Dense(1, activation="linear", name="predictions", input_shape=(None,3)))(input_next)

    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=[input_first, roll, pitch], outputs=predictions)    
    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mse'])
    
    return model

pathy = os.environ["HOME"]
path_best_model = '{}/08_23_keras_rnn_pose_best_model_mayonedata.keras'.format(pathy)
#print path_best_model
#best_hyperparameters = '{}/7_24_best_rnn_pose_hyperparameters'.format(pathy)
best_accuracy = 1

data_inputs = {}
data_inputs['images'] = imageArray[0:crossValidation]
data_inputs['roll'] = roll[0:crossValidation]
data_inputs['pitch'] = pitch[0:crossValidation]

epochs = int(math.ceil(12.0 / hvd.size()))

print len(pitch[0:crossValidation])
print len(torque[0:crossValidation])

model = load_model(path_best_model)
#parallel_model = multi_gpu_model(model, gpus=4)
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


test_inputs = {}
test_inputs['roll'] = roll[testingSet:index]
test_inputs['pitch'] = pitch[testingSet:index]
test_inputs['images'] = imageArray[testingSet:index]
score = parallel_model.predict(x=test_inputs)
#torque = np.load('april_orthoimage_DEBUG_torque.npy')

#print score
# Need to transpose to create row by column matrix
#torque = np.transpose(torque)
#torque = np.reshape(torque, (-1, 1))

# Take the absolute value because we have negative torque values
#torque = np.absolute(torque)

print torque[testingSet:index]
print score


#fitness(x=default_parameters)
#search_result = gp_minimize(func=fitness,
#                             dimensions=dimensions,
#                             acq_func='EI', # Expected Improvement.
#                             n_calls=50,
#                             x0=default_parameters)

#print search_result.x

