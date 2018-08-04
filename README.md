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

a) keras_rnn_run : Keras program to optimize hyperparameters and create a Keras RNN
                   model with ConvLSTM2D layers (LSTM cell with convolutional IO). Currently 
                   accepts RGB 128x128 images and torque/roll/pitch data from a spreadsheet.
                           
# How to use keras_rnn_run:

1. Clone this repository

2. Inside this repo, download any of the MAARS image datasets from the MAARS AWS. Make sure you download the "orthoimages" tarball. Unzip the orthoimage tarball in your repo so you now have a folder called "orthoimage" containing all the orthoimages. If you download multiple datasets, make sure to name the extracted folders differently.

3. You also need to download the "motor" tarfile as well as the "pose.csv" file from AWS. Extract the "motor.tar.gz" and find "motor_DRF.csv". Move both of these csv files inside of your orthoimage directory.

4. You will now need to preprocess your data with keras_data_gathering.py. You must first edit this script and find
the variable "paths." Change this variable to the paths of all the orthoimage datasets you downloaded. If you downloaded
all 3 datasets, you can link all three, or else you can link to just the ones you downloaded. If you only download one however, you will need to change your for loop to look like this:

 ```
  for root, dirs, files in (os.walk('orthoimage')): 
  
```
 
Once you've set your "paths" variable you can run the script. It will take a while because it's converting all the images into a numpy array and will save them in your current directory.

5. Once you've generated your numpy array files, you can run hyperparameter optimization via keras_rnn_run.py. Edit the script and find the variables "imageArray", "torque", "roll", and "pitch". Set each of these to the name of the numpy arrays generated in Step d. You may also need to tune the variables "crossValidation" and "testingSet." Depending on which dataset you chose,
it may have more or less images than the others. You may want to add a print statement to view the length of your data and
decide how to make your train/test splits accordingly. The last variable to change is path_best_model and best_hyperparameters. These variables set the path for saving your best model and the optimal hyperparameters as found
by scikit-optimize.

6. Once you've made some adjustments to keras_rnn_run.py, you should be able to run it on your data!

7. For training your optimized model, you can use keras_rnn_test.py with some of the same adjustments you made to 
keras_rnn_run.py. The main difference between the two scripts is at the bottom, where keras_test.py simply
runs Keras' training function, "fit", for however many epochs you specify and then saves the model.

                           
b) keras_cnn_run : Keras program to optimize hyperparameters and create a CNN model from 
                           best parameters. Currently accepts RGB 128x128 images from the orthoimage
                           directory and torque values from the motor_DRF spreadsheet in the motor/
                           directory. Rather than loop over the images and spreadsheet every time,
                           I have saved the input data as two vectors: grayscale_image_array_out.npy
                           and torque.out.npy. Number of calls to gp_minimize can be modified
                           as well as the # of samples to set aside for testing/validation.
                           
c) tensorflow_cnn_run : Runs Tensorflow custom CNN model with 2 layers

To test your CNN models, use either the test_tensorflow or test_keras scripts and make sure to specify 
the appropriate model file and dataset.
