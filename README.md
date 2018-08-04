Dependencies:
Tensorflow
Numpy
h5py
scikit-optimize
keras

MAARS-D3M Collaboration Team Code

In this repository, there are several scripts for running and analyzing the MAARS image dataset 
(Link: https://s3.console.aws.amazon.com/s3/buckets/athena-energy-dataset/?region=us-east-1&tab=overview)

The models include CNNs and RNNs. To create/train a model, use one of the 3 run scripts:
        1) keras_rnn_run : Keras program to optimize hyperparameters and create a Keras RNN
                           model with ConvLSTM2D layers (LSTM cell with convolutional IO). Currently 
                           accepts RGB 128x128 images from the orthoimage directory and torque values 
                           from the motor_DRF spreadsheet in the motor/ directory. Rather than loop over
                           the images and spreadsheet every time, I have saved the input data as two vectors: grayscale_image_array_timeseries_out.npy and torque.out.npy. Number of calls to gp_minimize can be modified as well as the # of samples to set aside for
                           testing/validation.
        2) keras_cnn_run : Keras program to optimize hyperparameters and create a CNN model from 
                           best parameters. Currently accepts RGB 128x128 images from the orthoimage
                           directory and torque values from the motor_DRF spreadsheet in the motor/
                           directory. Rather than loop over the images and spreadsheet every time,
                           I have saved the input data as two vectors: grayscale_image_array_out.npy
                           and torque.out.npy. Number of calls to gp_minimize can be modified
                           as well as the # of samples to set aside for testing/validation.
        3) tensorflow_cnn_run : Runs Tensorflow custom CNN model with 2 layers

To test your models, use either the test_tensorflow or test_keras scripts and make sure to specify 
the appropriate model file and dataset.