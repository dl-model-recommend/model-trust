'''
Training only classification network using the features

'''

import argparse
import csv
import os
import sys

import keras
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

task_mapping = {"shape":0,  "color":1, "size":2, "quadrant":3, "background":4}

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--task', default="all", choices=['all', 'color', 'shape', 'size', 'quadrant', 'background'],
						help='Use features from task color, shape, qudrant, size, background. Or all')
parser.add_argument('--prefix', help='add prefix to uniquely identify the each experiments', required=True)
parser.add_argument('--epoch', help='number of epoch to train the model', required=True)
args = parser.parse_args()

MATRIX_FILE_NAME = "trust_matrix.csv"

feat_dir = "./features"
epochs = args.epoch
verbose = 1
batch_size = 32
header_row = ["task"] + list(task_mapping.keys())

def train_models():

	trained_task = args.task

	fd = open(MATRIX_FILE_NAME,'w')
	writer = csv.writer(fd)
	writer.writerow(header_row)

	# shape - 0, color - 1, size - 2, quadrant - 3, background - 4
	for task_key in task_mapping.keys():

		if(trained_task == 'all' or trained_task == task_key ):

			print("\n\n########################################")
			print("Feature prefix -", args.prefix)
			print("Using features trained on", task_key, "\n")

			print("\nLoading data please wait...")
			train_data = np.load(os.path.join(feat_dir, args.prefix + "_" + task_key + "_train.npz"))
			print("Loading test dataset...")
			test_data = np.load(os.path.join(feat_dir, args.prefix + "_" + task_key + "_test.npz"))

			accuracy_list = []
			for predict_task_key, predict_task_value in task_mapping.items():

				lable_index = predict_task_value

				print("Predict lable index -", lable_index, "of", predict_task_key)

				# load trianing data
				train_x = train_data["data"]
				print("Traning data shape", train_x.shape)

				train_y = train_data["lables"][:,lable_index]
				num_lables = len(set(train_y))
				print("Traning lable shape", train_y.shape)
				print("Number of actual lables", num_lables)
				print("Train lables set", set(train_y))

				print("\nData pre-processing and shuffling...")
				total_data_point = train_x.shape[0]
				data_indices = np.arange(total_data_point)
				np.random.shuffle(data_indices)
				train_x = train_x.take(data_indices, axis=0)
				train_y = train_y.take(data_indices, axis=0)
				print("Done! Data pre-processing\n")

				data_shape = train_x.shape[1]
				train_y_one_hot = to_categorical(train_y, num_lables)

				# building small classification network
				model = Sequential()
				model.add(Dense(512, input_shape=(data_shape, ), activation='relu', name="cus_dense_1"))
				model.add(Dropout(0.5, name='cus_dense_do_1'))
				model.add(Dense(100, activation='relu', name='cus_dense_2'))
				model.add(Dropout(0.3, name='cus_dense_do_2'))
				model.add(Dense(num_lables, activation='softmax', name='cus_dense_3'))

				print(model.summary())

				# filepath = model_dir + model_name + "_e{epoch:02d}-acc{val_acc:.5f}.hdf5"
				# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
				# callbacks_list = [checkpoint]

				print("Submitting for training...")

				adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
				model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

				model.fit(train_x, train_y_one_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=0.2)

				print("\nTraining has finished")
				# model.save(model_dir + model_name  + "_final.hdf5")
				# print(model_dir + model_name  + "_final.hdf5")

				test_data_x = test_data["data"]
				test_data_answer = test_data["lables"][:,lable_index]
				num_lables = len(set(test_data_answer))

				print("Test data shape", test_data_x.shape)
				print("Test lables shapes ", test_data_answer.shape)
				print("Test lables set", set(test_data_answer))

				test_data_answer_one_hot = to_categorical(test_data_answer, num_lables)

				score = model.evaluate(x=test_data_x, y=test_data_answer_one_hot, verbose=2)
				print("\n######## Note this thing ########\nTest Score on features from",
						task_key, "and predicted", predict_task_key, "is -", score, "\n")

				accuracy_list.append(round(score[1],4))

			writer.writerow([task_key] + accuracy_list)

if __name__=="__main__":
	train_models()
