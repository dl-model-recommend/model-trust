###########################################
# Classifcation of Auxillary Tasks Params
###########################################

# What task to perform while training the model
# Options are ['all', 'color', 'shape', 'size', 'quadrant', 'background']
task = 'shape' 

# Input data path for the .npz file (refer to make_dataset.py to generate the .npz file)
train_features_path = "./features/my_densenet_feat_shape_train.npz"
test_features_path = "./features/my_densenet_feat_shape_test.npz"

# Path where the output trust matrix is of the current DL model is saved 
OUTPUT_MATRIX_FILE_NAME = "trust_matrix.csv"

# Add a name prefix to uniquely identify the each experiments
output_name_prefix = 'my_densenet_classifiers'

########################################
# Training Params
#######################################

# Should the model be trained in verbose mode
verbose = 1 # or 0

# Should the model be trained in verbose mode
validation_split = 0.2

# Number of epochs to train the model
epoch = 1

# Batch size for training
batch_size = 32