'''
Traning as baseline models (Densenet121) on dataset

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
						help='train model on all the task color, shape, qudrant, size, background')
parser.add_argument('--prefix', help='add prefix to uniquely identify the each experiments', required=True)
parser.add_argument('--epoch', help='number of epoch to train the model', required=True)
args = parser.parse_args()

# index for task's label in data
task_mapping = { "shape":0, "color":1, "size":2, "quadrant":3, "background":4}

label_num = { "shape":5, "color":7, "size":3, "quadrant":4, "background":3}
batch_size = 32
epochs = args.epoch

data_dir = "data"
train_data_file = "synthetic_train_data.npz"
test_data_file = "synthetic_test_data.npz"

model_dir = "models"
if not os.path.exists(model_dir):
	os.makedirs(model_dir)


########################################################################################
#
# Change build_model
#
#
#
#
########################################################################################

def build_model(num_lables):

	base_model = DenseNet121(include_top=False, weights=None, input_shape=(256,256,3))
	flat_1 = Flatten()(base_model.output)

	cus_dense_1 = Dense(512, activation='relu', name='cus_dense_1')(flat_1)
	cus_dense_do_1 = Dropout(0.5, name='cus_dense_do_1')(cus_dense_1)
	cus_dense_2 = Dense(100, activation='relu', name='cus_dense_2')(cus_dense_do_1)
	cus_dense_do_2 = Dropout(0.3, name='cus_dense_do_2')(cus_dense_2)
	cus_dense_3 = Dense(num_lables, activation='softmax', name='cus_dense_3')(cus_dense_do_2)

	model = Model(base_model.input, cus_dense_3)
	return model



def training():

	task = args.task

	# shape - 0, color - 1, size - 2, quadrant - 3, background - 4
	for task_key, task_value in task_mapping.items():

		if(task == 'all' or task == task_key):

			print("\nBuilding model for task", task_key)

			lables_ind = task_value
			num_lables = label_num[task_key]
			verbose = 1

			# model_name = "densenet_base_" + task_key + "_lr04"
			model_name = args.prefix + "_" + task_key

			model = build_model(num_lables)
			print(model.summary())

			print("\nLoading dataset...")
			X = np.load(os.path.join(data_dir, train_data_file))
			x_train = X["data"]
			answer = X["lables"][:,lables_ind]
			print("Train data shape", x_train.shape)
			print("Train lables shapes ", answer.shape)
			print("Train lables set", set(answer))

			print("Pre-processing the data...")
			x_train = x_train.astype('float32')
			x_train /= 255
			total_data_point = x_train.shape[0]
			data_indices = np.arange(total_data_point)
			np.random.shuffle(data_indices)
			x_train = x_train.take(data_indices, axis=0)
			answer = answer.take(data_indices, axis=0)

			y_train = to_categorical(answer, num_lables)

			filepath = os.path.join(model_dir, model_name + "_e{epoch:02d}-acc{val_acc:.5f}.hdf5")
			checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
			callbacks_list = [checkpoint]

			print("Submitting for training...")

			adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
			model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
			model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=0.2, callbacks=callbacks_list)

			print("\nTraining has finished")
			model.save(os.path.join(model_dir, model_name  + "_final.hdf5"))
			print("Trained model store at", os.path.join(model_dir, model_name  + "_final.hdf5"))

			print("\nLoading test dataset...")
			test_data = np.load(os.path.join(data_dir, test_data_file))
			test_data_x = test_data["data"]
			test_data_answer = test_data["lables"][:,lables_ind]
			print("Test data shape", test_data_x.shape)
			print("Test lables shapes ", test_data_answer.shape)
			print("Test lables set", set(test_data_answer))

			test_data_x = test_data_x.astype('float32')
			test_data_x /= 255
			test_data_answer_one_hot = to_categorical(test_data_answer, num_lables)

			score = model.evaluate(x=test_data_x, y=test_data_answer_one_hot, verbose=1)
			print("\nTest Score", score)

if __name__ == "__main__":
	training()
