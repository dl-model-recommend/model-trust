'''
Main wrapper function to run the entire pipeline:
1. Generate synthetic dataset (Can skip this with custom dataset)
2. Train a deep learning model for one task (currently training DenseNet model, can replace this with custom DL model)
3. Extract features of dataset on the DL model (can chose any custom layer)
4. Train neural network classifier model for other auxillary tasks (can replace this with custom classifier)
5. Computer Model trust score

Overall Input:
-------------
Images shape: (N, h, w, 3) where 'N' is the number of images, 'hxw' is the dimension of each image
Images label: (N, m) where 'm' is the number of tasks performed using the same image - m_1 can be a 3-class classification task, m_2 maybe a 7-class classification task etc. One of the 'm' is a primary task and the rest of the 'm' becomes auxillary tasks.
DL model: 'keras' or 'tf.keras' compatible DL model object (trained or vanilla). The DL model is trained on the primary task.

Overall Output:
--------------
Trust score: [0-1], on how much the input DL model performs trustworthy learning on the input image data
'''


from make_dataset import make_dataset
from train_DL_model import train_models
from train_DL_model import extract_features
from train_classifier_model import classification_net
from compute_model_trust import calculate_model_trust

def main():
    # 1. Generate synthetic dataset (Can skip this with custom dataset)
    # All the controlling parameters are available in make_dataset/dataset_params.py
    print("########################################################################")
    print("1. Generate synthetic dataset (Can skip this with custom dataset)")
    make_dataset.make_dataset()

    # 2. Train a deep learning model for one task (currently training DenseNet model, can replace this with custom DL model)
    # All the controlling parameters are available in train_DL_model/model_params.py
    print("########################################################################")
    print("2. Train a deep learning model for one task (currently training DenseNet model, can replace this with custom DL model)")
    train_models.train_models()

    # 3. Extract features of dataset on the DL model (can chose any custom layer)
    # All the controlling parameters are available in train_DL_model/model_params.py
    print("########################################################################")
    print("3. Extract features of dataset on the DL model (can chose any custom layer)")
    extract_features.extract_features()

    # 4. Train neural network classifier model for other auxillary tasks (can replace this with custom classifier)
    # All the controlling parameters are available in train_classifier_model/classification_params.py
    print("########################################################################")
    print("4. Train neural network classifier model for other auxillary tasks (can replace this with custom classifier)")
    classification_net.train_models()

    # 5. Computer Model trust score
    # All the controlling parameters are available in compute_model_trust/trust_params.py
    print("########################################################################")
    print("5. Computer Model trust score")
    calculate_model_trust.calculate()


if __name__ == '__main__':
	main()