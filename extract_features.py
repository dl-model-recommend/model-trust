'''
Extracting features (flatten layer output) from trained models

'''

import argparse
import os

import numpy as np
from keras.applications import DenseNet121
from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--task', default="all", choices=['all', 'color', 'shape', 'size', 'quadrant', 'background'],
						help='Extract features from task color, shape, qudrant, size, background')
parser.add_argument('--prefix', help='add prefix to uniquely identify the each experiments', required=True)
parser.add_argument('--layer', help='layer name from which features to be extracted', required=True)
args = parser.parse_args()

task_mapping = { "shape":0, "color":1, "size":2, "quadrant":3, "background":4}

model_dir = "./models"
data_dir = "data"
train_data_file = "synthetic_train_data.npz"
test_data_file = "synthetic_test_data.npz"

feat_dir = "./features"
if(not os.path.exists(feat_dir)):
        os.makedirs(feat_dir)

def get_features():

	task = args.task

	for task_key in task_mapping.keys():

		if(task == 'all' or task == task_key):

			print("Loading the pretrained model")
			trained_model = os.path.join(model_dir, args.prefix + "_" + task_key + "_final.hdf5")

			model = load_model(trained_model)
			print(model.summary())
			dense_model = Model(inputs=model.input, outputs=model.get_layer(args.layer).output)

			# loading trianing dataset
			train_data = np.load(os.path.join(data_dir, train_data_file))
			train_x = train_data["data"]
			train_x = train_x.astype('float32')
			train_x /= 255
			train_y = train_data["lables"]
			print("\nTrain data shape", train_x.shape)
			print("Train lables shape", train_y.shape)

			dense_feature = dense_model.predict(train_x)
			print("\nFeatures of train data (shape)", dense_feature.shape)
			np.savez_compressed(os.path.join(feat_dir,  args.prefix + "_" + task_key + '_train.npz'), data=dense_feature, lables=train_y)

			# loading test dataset
			test_data = np.load(os.path.join(data_dir, test_data_file))
			test_x = test_data["data"]
			test_x = test_x.astype('float32')
			test_x /= 255
			test_y = test_data["lables"]
			print("\nTest data shape", test_x.shape)
			print("Test lables shape", test_y.shape)

			dense_feature = dense_model.predict(test_x)
			print("\nFeatures of test data (shape)", dense_feature.shape)
			np.savez_compressed(os.path.join(feat_dir,  args.prefix + "_" + task_key + '_test'), data=dense_feature, lables=test_y)

if __name__ == '__main__':
	get_features()
