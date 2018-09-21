# Base CNN imports
import pandas as panda
import imageio
import os
import numpy as np
import random
from numbers import Number
import pdb
import re
# For parameter tuning
import math

from itertools import chain

# Set orthoimage directory
#paths = ('merged_2018-08-29-12-36-26/orthoimage')
paths = ('merged_2018-04-09-14-28-11/orthoimage', 'merged_2018-05-10-14-30-07/orthoimage', 'merged_2018-07-13-13-03-27/orthoimage', 'merged_2018-07-13-13-39-03/orthoimage')
# For the imageList, every index will represent
# the image counts for the samples contained under
# that index
def readImage(filepath):
    im = imageio.imread(filepath)
    return np.dot(im[...,:3], [0.299, 0.587, 0.114])
 
# Walk through our directory and ready images into input vector
for root, dirs, files in chain.from_iterable(os.walk(path) for path in paths):
#for root, dirs, files in (os.walk(paths)):
    print root
    prefix_fname = root.replace('/', '_')
    imagePath = '{}_timeseries_images'.format(prefix_fname)
    torquePath = '{}_timeseries_torque'.format(prefix_fname)
    pitchPath = '{}_timeseries_pitch'.format(prefix_fname)
    rollPath = '{}_timeseries_roll'.format(prefix_fname)

    imageList = []
    correspondingTorque = []
    correspondingRoll = []
    correspondingPitch = []
    oldsampleid = -1
    index = 0

    print root;

    # Manually check your data to see where the image sequences start and end
    # Check the motor_DRF file for 0/NaN values at the beginning and end
    # Check the pose.csv file for 0/NaN values at the beginning and end
    # Select a start and finish index accordingly
    mindex = (int)(raw_input('Enter first index to start collecting data: '))
    maxdex = (int)(raw_input('Enter maximum index to crawl to: '))

    # Motor_DRF and Pose.csv files need to be in the path specified in your 
    # "paths" variable above
    # Get the torques for this new directory of mages
    tp=os.path.join(root,'motor_DRF.csv')
    pose=os.path.join(root,'pose.csv')
    motorDRF = panda.read_csv(tp, sep="\t")
    pose = panda.read_csv(pose, sep="\t")

    csvtimestamp = motorDRF['TIMESTAMP'].values
    csvtorque = motorDRF['EST_POWER'].values
    csvroll = pose['ROLL'].values
    csvpitch = pose['PITCH'].values
    print len(csvtorque)
    print len(csvroll)

    pool = Pool(processes=cpu_count())

    runningSequenceRoll = []
    runningSequencePitch = []
    runningSequenceTorque = []
    runningSequenceFilteredFiles = []
    sequenceCounter = 0
    filteredFiles = []
    for filename in sorted(files):
	if (filename.endswith('.png')):
	 # Get the sample id
 	        sampleid = (filename.split('_'))[0]
                sampleid = (int)(re.sub('(s0*)', '', sampleid))

		if ((sampleid == oldsampleid+1) and sampleid > mindex and sampleid < maxdex) :
			sequenceCounter += 1
			print("sampleid {} oldsampleid {}".format(sampleid, oldsampleid))
			if (sequenceCounter == 29) :
				correspondingRoll.extend(runningSequenceRoll)
				correspondingPitch.extend(runningSequencePitch)
				correspondingTorque.extend(runningSequenceTorque)
				filteredFiles.extend(runningSequenceFilteredFiles)

				print os.path.join(root, filename);
                                del runningSequenceRoll[:]
                                del runningSequencePitch[:]
                                del runningSequenceTorque[:]
                                del runningSequenceFilteredFiles[:]
				sequenceCounter = 0
			else :
				print sequenceCounter
				runningSequenceRoll.append(float(csvroll[sampleid]))
				runningSequencePitch.append(float(csvpitch[sampleid]))
				runningSequenceTorque.append(float(csvtorque[sampleid]))
				runningSequenceFilteredFiles.append(os.path.join(root, filename))

		elif (sampleid != oldsampleid) :
                	del runningSequenceRoll[:]
                        del runningSequencePitch[:]
                        del runningSequenceTorque[:]
                        del runningSequenceFilteredFiles[:]
			sequenceCounter = 0

	 	oldsampleid = sampleid

    print len(filteredFiles) 
    imageList = pool.map(readImage, filteredFiles)
	
    # Convert imagelist into numpy array
    imageArray = np.asarray(imageList, dtype=np.float32)
    print(len(imageArray))
    np.save(imagePath, imageArray)

    # Convert our torquelist to numpuy array
    torque = np.asarray(correspondingTorque, dtype=np.float32)

    # Need to transpose to create row by column matrix
    torque = np.transpose(torque)
    torque = np.reshape(torque, (-1, 1))

    # Take the absolute value because we have negative torque values
    torque = np.absolute(torque)

    np.save(torquePath, torque)

    roll = np.asarray(correspondingRoll, dtype=np.float32)
    pitch = np.asarray(correspondingPitch, dtype=np.float32)

    np.save(rollPath, roll)
    np.save(pitchPath, pitch)

    pool.close()
    pool.terminate()
