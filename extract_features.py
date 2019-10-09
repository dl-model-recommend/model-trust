'''
Extracting features (from a predefined layer) from trained models
'''

import model_params
import util_functions as utils
import os, numpy as np
from keras.models import Model, load_model

########################################################################################
# Hard-coded rules. Please change this if you are changing the dataset format
########################################################################################
# 1. Provide the order in which the labels are provided in the label matrix
task_mapping = {"shape": 0, "color": 1, "size": 2, "quadrant": 3, "background": 4}

def extract_features():
	"""Extract features from a trained DL model (from a predefined layer)"""

	utils.create_directory(model_params.feat_dir)
	task = model_params.task
	for task_key in task_mapping.keys():
		if(task == 'all' or task == task_key):
			print("Loading the pretrained model ..... ")
			trained_model_path = model_params.trained_model_path
			if not trained_model_path:
				trained_model_path = os.path.join(model_params.model_dir, model_params.output_name_prefix + "_" + task_key + "_final.hdf5")
			
			model = load_model(trained_model_path)
			print("Printing model summary ....... ")
			print(model.summary())
			dense_model = Model(inputs=model.input, outputs=model.get_layer(model_params.layer).output)

			# loading Trianing dataset
			print("\nLoading and preprocessing training dataset ...... ")
			train_data = np.load(model_params.train_data_path)
			train_x = train_data["data"]
			train_x = train_x.astype('float32')
			train_x /= 255
			train_y = train_data["lables"]
			print("Train data shape: ", train_x.shape)
			print("Train lables shape: ", train_y.shape)

			print("Extracting features from training dataset ...... ")
			dense_feature = dense_model.predict(train_x)
			print("Features of train data (shape): ", dense_feature.shape)
			print("Saving features from training dataset ...... ")
			np.savez_compressed(os.path.join(model_params.feat_dir,  model_params.output_name_prefix_feat + "_" + task_key + '_train.npz'), data=dense_feature, lables=train_y)

			# loading test dataset
			print("\nLoading and preprocessing testing dataset ...... ")
			test_data = np.load(model_params.test_data_path)
			test_x = test_data["data"]
			test_x = test_x.astype('float32')
			test_x /= 255
			test_y = test_data["lables"]
			print("Test data shape: ", test_x.shape)
			print("Test lables shape: ", test_y.shape)

			print("Extracting features from testing dataset ...... ")
			dense_feature = dense_model.predict(test_x)
			print("Features of test data (shape): ", dense_feature.shape)
			print("Saving features from testing dataset ...... ")
			np.savez_compressed(os.path.join(model_params.feat_dir,  model_params.output_name_prefix_feat + "_" + task_key + '_test.npz'), data=dense_feature, lables=train_y)

if __name__ == '__main__':
	extract_features()
