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
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import InputLayer, Input, concatenate, BatchNormalization
from tensorflow.python.keras.layers import Reshape, MaxPooling2D, Lambda
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, ConvLSTM2D
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import save_model, load_model
from tensorflow.contrib import rnn
from tensorflow.python.keras.utils import multi_gpu_model

# Scikit Optimizer
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

# Hyperparameter tuning
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                         name='learning_rate')


dim_num_dense_layers = Integer(low=50, high=200, name='num_dense_layers')

dim_num_conv_layers = Integer(low=1, high=4, name='num_conv_layers')

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
    s = "./keras_cnn_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}_{4}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_conv_layers,
                       kernel_size,
                       num_filters)

    return log_dir

default_parameters = [1e-3, 50, 2, 5, 32]

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

crossValidation = 15000
testingSet = 5000

imageArray = np.load('all_grayscale_image_array.out.npy')
torque = np.load('all_torque.out.npy')
roll = np.load('all_roll_values.npy')
pitch = np.load('all_pitch_values.npy')
index = len(torque)
print index
print len(roll)
print len(pitch)
# Create validation data
valid_inputs = {}
valid_inputs['images'] = imageArray[testingSet:crossValidation]
valid_inputs['roll'] = roll[testingSet:crossValidation]
valid_inputs['pitch'] = pitch[testingSet:crossValidation]
validation_data = (valid_inputs, torque[testingSet:crossValidation])


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
    
    # Start construction of a Keras Sequential model.

    # Add an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.
    input_first = (Input(shape=(img_width, img_width, 1), name='images'))
    input_next = input_first

    for i in range(num_conv_layers):
        name = 'layer_conv_{0}'.format(i+1)
        # First convolutional layer.
        # There are many hyper-parameters in this layer, but we only
        # want to optimize the kernel_size-function in this example.

        factor = 1
        filters = num_filters
        if (i == 0):
            factor = 1
            filters = 1
        else:
            factor = 2*i

        input_next = (Conv2D(kernel_size=kernel_size, input_shape=(img_width/factor, img_width/factor, filters), strides=1, 
                        filters=num_filters, padding='same', activation="relu", name=name))(input_next)
        input_next = (MaxPooling2D(pool_size=2, input_shape=(img_width/factor, img_width/factor, num_filters), strides=2))(input_next)
        input_next = (BatchNormalization(input_shape=(img_width/(factor*2), img_width/(factor*2), num_filters)))(input_next)


    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    input_next = (Flatten())(input_next)

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
    input_next = (Dense(1, activation="linear", input_shape=(None, (num_filters*img_width*img_width/(4**num_conv_layers)))))(input_next)

    def scaleDown(x):
        import tensorflow as tf
        return tf.scalar_mul(
            10e-2,
            tf.div(
                tf.subtract(
                    x, 
                    tf.reduce_min(x)
                ), 
                tf.subtract(
                    tf.reduce_max(x), 
                    tf.reduce_min(x)
                )
            )   
        )
 
    input_next = (Lambda(scaleDown))(input_next)
    roll = Input(shape=(1,), name='roll')
    pitch = Input(shape=(1,), name='pitch')  
    x = concatenate([input_next, roll, pitch])  
    predictions = Dense(1, activation='linear', name='predictions', input_shape = (None, 3))(x) 

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
path_best_model = '{}/07_23_keras_cnn_pose_best_model.keras'.format(pathy)
print path_best_model
best_hyperparameters = '{}/7_23_best_pose_hyperparameters'.format(pathy)
best_accuracy = 1

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers,
            num_conv_layers, kernel_size, num_filters):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_conv_layers:   Number of nodes in each conv layer.
    kernel_size:        kernel_size function for all layers.
    num_filters:       Number of Filters
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_conv_layers:', num_conv_layers)
    print('kernel_size:', kernel_size)
    print('num_filters:', num_filters)
    print()
    
    # Create the neural network with these hyper-parameters.
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_conv_layers=num_conv_layers,
                         kernel_size=kernel_size,
                         num_filters=num_filters)

    
    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, num_dense_layers,
                           num_conv_layers, kernel_size, num_filters)
    
    # Create a callback-function for Keras which will be
    # run after each epoch has ended during training.
    # This saves the log-files for TensorBoard.
    # Note that there are complications when histogram_freq=1.
    # It might give strange errors and it also does not properly
    # support Keras data-generators for the validation-set.
    callback_log = TensorBoard(
        log_dir=log_dir,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False)
   
    # Use Keras to train the model.
    #checker = torque[crossValidation:index]
    parallel_model = multi_gpu_model(model, gpus=4)
    optimizer = Adam(lr=learning_rate)
    parallel_model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mse'])
    data_inputs = {}
    data_inputs['images'] = imageArray[crossValidation:index]
    data_inputs['roll'] = roll[crossValidation:index]
    data_inputs['pitch'] = pitch[crossValidation:index]
    history = parallel_model.fit(x=data_inputs,
                        y=torque[crossValidation:index],
                        epochs=num_dense_layers,
                        batch_size=32,
                        validation_data=validation_data,
                        callbacks=[callback_log])

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.

    accuracy = history.history['val_loss'][-1]

    # Print the classification accuracy.
    print()
    print("Error: {}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy < best_accuracy:
        # Save the new model to harddisk.
	print 'BEST NEW MODEL SAVED'
        model.save(path_best_model)
        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.

    K.clear_session()
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return accuracy



model = load_model(path_best_model)
parallel_model = multi_gpu_model(model, gpus=4)
optimizer = Adam(lr=1e-3, clipnorm=1.0, clipvalue=0.5)
parallel_model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mse'])

test_inputs = {}
test_inputs['roll'] = roll[testingSet:index]
test_inputs['pitch'] = pitch[testingSet:index]
test_inputs['images'] = imageArray[testingSet:index]
score = parallel_model.predict(x=test_inputs)

print score
print torque[testingSet:index]

#fitness(x=default_parameters)
#search_result = gp_minimize(func=fitness,
#                             dimensions=dimensions,
#                             acq_func='EI', # Expected Improvement.
#                             n_calls=50,
#                             x0=default_parameters)
#f1=open(best_hyperparameters, 'w+')
#f1.write(search_result.x)
#f1.close()
#
