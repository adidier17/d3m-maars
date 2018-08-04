# Base CNN imports
import pandas as panda
import imageio
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from numbers import Number
import pdb

# For parameter tuning
import math

# Keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input, BatchNormalization
from tensorflow.python.keras.layers import Reshape, MaxPooling2D, Lambda
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import save_model, load_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.utils import multi_gpu_model

# Scikit Optimizer
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

# model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
# model_vgg16_conv.summary()

# input = Input(shape=(128,128,3),name = 'image_input')

# for l in model_vgg16_conv.layers:
#     l.trainable = False

# #Use the generated model 
# output_vgg16_conv = model_vgg16_conv(input)


# def scaleDown(x):
#     import tensorflow as tf
#     return tf.scalar_mul(
#         10e-2,
#         tf.div(
#             tf.subtract(
#                 x, 
#                 tf.reduce_min(x)
#             ), 
#             tf.subtract(
#                 tf.reduce_max(x), 
#                 tf.reduce_min(x)
#             )
#         )   
#     )

# #Add the fully-connected layers 
# x = Flatten(name='flatten')(output_vgg16_conv)
# x = Dense(1, activation='linear', name='predictions')(x)
# x = Lambda(scaleDown)(x)

# #Create your own model 
# my_model = Model(inputs=input, outputs=x)

# #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
# my_model.summary()


# # We know that MNIST images are 128 pixels in each dimension.
# img_width = 128

# # Images are stored in one-dimensional arrays of this length.
# img_size_flat = img_width * img_width

# # Tuple with height and width of images used to reshape arrays.
# # This is used for plotting the images.
# img_shape = (img_width, img_width)

# # Number of colour channels for the images: 1 channel for gray-scale.
# num_channels = 3

# # Tuple with height, width and depth used to reshape arrays.
# # This is used for reshaping in Keras.
# img_shape_full = (img_width, img_width, num_channels)


# # Set image directory, only png's
# imageDir = 'orthoimage/'

# # Get timestamp and torque of right front drive motor
# motorDRF = panda.read_csv('motor/motor_DRF.csv')
# timestamps = motorDRF.TIMESTAMP

# csvtorque = motorDRF['TORQUE'].values

# # For the imageList, every index will represent
# # the image counts for the samples contained under
# # that index
# imageList = []
# correspondingTorque = []

# index = 0
# crossValidation = 800
# testingSet = 400

# # Walk through our directory and ready images into input vector
# for root, dirs, files in (os.walk(imageDir)):
#     for filename in sorted(files):
#             p=os.path.join(root,filename)

#             # Read the current image and update the previous sample id var
#             im = imageio.imread(p)
#             imageList.append(im)

#             # Get the sample id
#             sampleid = (filename.split('_'))[0]
#             sampleid = int(sampleid.replace('s000', ''))
#             correspondingTorque.append(float(csvtorque[sampleid]))
#             index += 1

# # Convert imagelist into numpy array
# imageArray = np.asarray(imageList, dtype=np.float32)
# # Convert our torquelist to numpuy array
# torque = np.asarray(correspondingTorque, dtype=np.float32)

# # Need to transpose to create row by column matrix
# torque = np.transpose(torque)
# torque = np.reshape(torque, (-1, 1))

# # Take the absolute value because we have negative torque values
# torque = np.absolute(torque)

# # Create validation data
# validation_data = (imageArray[testingSet:crossValidation], torque[testingSet:crossValidation])

# # Use the Adam method for training the network.
# # We want to find the best learning-rate for the Adam method.
# optimizer = Adam(lr=1e-3)

# # In Keras we need to compile the model so it can be trained.
# # my_model.compile(optimizer=optimizer,
# #                 loss='mean_squared_error',
# #                 metrics=['mse'])

testingSet = 5000
crossValidation = 15000

imageArray = np.load('all_grayscale_image_array.out.npy')
torque = np.load('all_torque.out.npy')
index = len(torque)
index2 = len(imageArray)
print index
print index2

validation_data = (imageArray[testingSet:crossValidation], torque[testingSet:crossValidation])

pathy = os.environ["HOME"]
path_best_model = '{}/07_18_keras_cnn_best_model.keras'.format(pathy)
model = load_model(path_best_model)
#model.summary()
i = 0
for layer in model.layers:
	layer.name = 'layer'+str(i)+'_1K_test_719'
	i+=1

pmodel = multi_gpu_model(model, gpus=4)

history = model.fit(x=imageArray[crossValidation:index],
                     y=torque[crossValidation:index],
                     epochs=1000,
                     batch_size=32,
                     validation_data=validation_data)

new_path_best_model = '{}/07_19_keras_cnn_best_model.keras'.format(pathy)
model.save(new_path_best_model)


