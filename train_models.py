'''
Traning as baseline models (Densenet121) on dataset
Change the build_model() function to use your custom deep learning model
This function should written a valid keras or tf.keras Model object
'''

import model_params
import util_functions as utils
import os, numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

########################################################################################
# Hard-coded rules. Please change this if you are changing the dataset format
########################################################################################
# 1. Provide the order in which the labels are provided in the label matrix
task_mapping = {"shape": 0, "color": 1, "size": 2, "quadrant": 3, "background": 4}

# 2. Provide the max number of classes in each task
label_num = { "shape": 5, "color": 7, "size": 3, "quadrant": 4, "background": 3}

########################################################################################
# Function that defines the DL model to be trained. 
# Edit this function to use custom DL model.
# This function should written a valid keras or tf.keras Model object
########################################################################################
def build_model(num_lables):
	"""Define the deep learning model to be trained"""
	from keras.applications import DenseNet121
	from keras.layers import (Dense, Dropout, Flatten)
	from keras.models import Model, Sequential, load_model
	
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
	"""This is the main function that trains the model for one or many tasks"""

	task = model_params.task
	for task_key, task_value in task_mapping.items():
		if(task == 'all' or task == task_key):
			print("==================================================")
			print("Building model for task ..... ", task_key)

			num_task_labels = label_num[task_key]
			model_name = model_params.output_name_prefix + "_" + task_key

			# Edit this method to use custom DL models.
			model = build_model(num_task_labels)
			print("Printing model summary for task", task_key)
			print(model.summary())

			print("\nLoading training dataset ...... ")
			data_type = "train"
			X = np.load(model_params.train_data_path)
			x_train = X["data"]
			answer = X["lables"][:,task_value]
			print("Train data shape", x_train.shape)
			print("Train labels shapes ", answer.shape)
			print("Train labels set", set(answer))

			print("Pre-processing training dataset ...... ")
			x_train = x_train.astype('float32')
			x_train /= 255
			total_data_point = x_train.shape[0]
			data_indices = np.arange(total_data_point)
			np.random.shuffle(data_indices)
			x_train = x_train.take(data_indices, axis=0)
			answer = answer.take(data_indices, axis=0)
			y_train = to_categorical(answer, num_task_labels)

			utils.create_directory(model_params.model_dir)
			filepath = os.path.join(model_params.model_dir, model_name + "_e{epoch:02d}-acc{val_acc:.5f}.hdf5")
			checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
			callbacks_list = [checkpoint]

			print("Submitting the model for training ...... ")
			adam = Adam(lr=model_params.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
			model.compile(optimizer=adam, loss=model_params.loss_function, metrics=['accuracy'])
			model.fit(x_train, y_train, batch_size=model_params.batch_size, epochs=model_params.epoch, verbose=model_params.verbose, validation_split=model_params.validation_split, callbacks=callbacks_list)

			print("\nTraining has finished ...... ")
			model.save(os.path.join(model_params.model_dir, model_name  + "_final.hdf5"))
			print("Trained model store at: ", os.path.join(model_params.model_dir, model_name  + "_final.hdf5"))

			print("\nLoading test dataset ...... ")
			data_type = "test"
			test_data = np.load(model_params.test_data_path)
			test_data_x = test_data["data"]
			test_data_answer = test_data["lables"][:,task_value]
			print("Test data shape", test_data_x.shape)
			print("Test lables shapes ", test_data_answer.shape)
			print("Test lables set", set(test_data_answer))

			print("Pre-processing test dataset ...... ")
			test_data_x = test_data_x.astype('float32')
			test_data_x /= 255
			test_data_answer_one_hot = to_categorical(test_data_answer, num_task_labels)

			print("Evaluating the model using test data ...... ")
			score = model.evaluate(x=test_data_x, y=test_data_answer_one_hot, verbose=model_params.verbose)
			print("\nTest Score", score)

	return None

if __name__ == "__main__":
	training()
