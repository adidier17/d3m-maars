# Base CNN imports
import pandas as panda
import imageio
import os
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tf.python import debug as tf_debug

# For parameter tuning
import math

# Keras
from tf.python.keras import backend as K
from tf.python.keras.models import Sequential
from tf.python.keras.layers import InputLayer, Input
from tf.python.keras.layers import Reshape, MaxPooling2D
from tf.python.keras.layers import Conv2D, Dense, Flatten
from tf.python.keras.callbacks import TensorBoard
from tf.python.keras.optimizers import Adam
from tf.python.keras.models import load_model

# Scikit Optimizer
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

# Hyperparameter tuning
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                         name='learning_rate')


dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')

dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')

dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation]

default_parameters = [1e-3, 2, 32, 'relu']



# Set image directory, only png's
imageDir = 'image/'

# Get timestamp and torque of right front drive motor
motorDRF = panda.read_csv('motor/motor_DRF.csv')
timestamps = motorDRF.TIMESTAMP
# Convert torque into list
torqueList = (motorDRF.TORQUE).tolist()

imageList = []
i = 0
# Skip first 50 images because torque for these images in the csv is 0
# Leave the next 40 images (Nos. 50-90) for testing the model 
missing_data_DRF = 90

# Walk through our directory and ready images into input vector
for root, dirs, files in (sorted(os.walk(imageDir))):
    for filename in files:
            p=os.path.join(root,filename)
            im = imageio.imread(p)
            i += 1
            imageList.append(im) 

# We're going to use i as an index so it needs to be one less
# than the total length of the array 
i -= 1

# Convert imagelist into numpy array
imageList = imageList
imageArray = np.array(imageList, dtype=np.float32)

# The CSV file has more torque values than we have
# images, so delete all unnecessary torque listings
torqueList = torqueList
torque = np.array(torqueList, dtype=np.float32)

# Need to transpose to create row by column matrix
torque = np.transpose(torque)
torque = np.reshape(torque, (-1, 1))

# Take the absolute value because we have negative torque values
torque = np.absolute(torque)

# Input x is an image of size 480 * 640 with 3 RGB channels
x = tf.placeholder(tf.float32, [None, 480, 640, 3])
# Output is a single torque value
y = tf.placeholder(tf.float32, [None, 1])

# 32 convolutions, each is through 5x5x3 set of neurons
W1 = tf.Variable(tf.random_uniform([5, 5, 3, 32]))
# Bias
b1 = tf.Variable(tf.random_uniform([32]))

# Another 32 convolutions
W2 = tf.Variable(tf.random_uniform([5, 5, 32, 32]))
b2 = tf.Variable(tf.random_uniform([32]))

# After two maxpool operations, we are operating on data
# that is 1/4 the size of our original image, so divide
# width and height by 4. In this case we started with 
# 480 x 640 ---> 120 x 160 x 32 output channels (from
# the above convolution)
 
W_out = tf.Variable(tf.random_uniform([120*160*32, 1]))
b_out = tf.Variable(tf.random_uniform([1]))

def conv_layer(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_with_b = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(conv_with_b)
    return conv_out

def maxpool_layer(conv, k=2):
    return tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1],
     padding='SAME')

def model():
    # Flatten our batch of images
    x_reshaped = tf.reshape(x, shape=[-1, 480, 640, 3])
   
    # Run convolutuion
    conv_out1 = conv_layer(x_reshaped, W1, b1)
    # Maxpool to reduce size of image
    maxpool_out1 = maxpool_layer(conv_out1)
    # Local Response Normalization on the data
    norm1 = tf.nn.lrn(maxpool_out1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
   
    # Repeat above process
    conv_out2 = conv_layer(norm1, W2, b2)
    norm2 = tf.nn.lrn(conv_out2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    maxpool_out2 = maxpool_layer(norm2)

    # Flatten feature vector
    maxpool_reshaped = tf.reshape(maxpool_out2, [-1,
    W_out.get_shape().as_list()[0]])
 
    # Matrix multiply to get our 1D output
    out = tf.add(tf.matmul(maxpool_reshaped, W_out), b_out)
    # Normalize the data to range between 0 and 0.01
    out = tf.scalar_mul(
            10e-2,
            tf.div(
                tf.subtract(
                    out, 
                    tf.reduce_min(out)
                ), 
                tf.subtract(
                    tf.reduce_max(out), 
                    tf.reduce_min(out)
                )
            )   
    )
    return out

model_op = model()

# Calculate mean squared error between our actual torque and our model
# Loss function for the model, metrics function for user to see via terminal
cost = tf.losses.mean_squared_error(labels=y,predictions=model_op)
metrics = tf.losses.mean_squared_error(labels=y,predictions=model_op)

# Train function will seek to reduce the cost
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    
    sess.run(init)
    # Run debugger by uncommenting below line
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # Batch size can be changed
    batch_size = 32
    print('batch size', batch_size)
    for j in range(0, 1000):
        print('EPOCH', j)
 
        r = np.random.permutation(len(imageArray))
        imageArray = imageArray[r]
        torque = torque[r]

        for k in range(missing_data_DRF, i, batch_size):
            batch_data = imageArray[k:k+batch_size]
            batch_onehot_vals = torque[k:k+batch_size]
            accuracy = sess.run([train_op, metrics], feed_dict={x:batch_data, y: batch_onehot_vals})
            print(accuracy[1])
        
        if (j % 100 == 0):
            # Save the variables to disk.
            save_path = saver.save(sess, os.getcwd() + "/tensorflow_cnn_model.ckpt")
            print("Model saved") 

    print('DONE WITH EPOCH')




