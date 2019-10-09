########################################
# Model Params
#######################################

# Input data path for the .npz file (refer to make_dataset.py to generate the .npz file)
train_data_path = "./data/synthetic_train_data.npz"
test_data_path = "./data/synthetic_train_data.npz"

# What task to perform while training the model
# Options are ['all', 'color', 'shape', 'size', 'quadrant', 'background']
task = 'shape' 

# Add a name prefix to uniquely identify the each experiments
output_name_prefix = 'my_densenet'

# Output model directory to save model checkpoints
model_dir = "models"

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

# Adam optimizer learning rate
learning_rate = 0.0001

# Loss function 
loss_function = 'categorical_crossentropy'

########################################
# Feature Extraction Params
#######################################

# Which trained model to use
trained_model_path = ''

# Which layer to extract features from (the default value is provided for DenseNet121 model)
layer = 'flatten_1'

# Path where the extracted features are saved
feat_dir = "features"

# Add a name prefix to uniquely identify the each experiments
output_name_prefix_feat = 'my_densenet_feat'