# [Reducing Model Overlearning]

This repository contains the original implementation of the paper - **[Reducing Overlearning through Disentangled Representations by Suppressing Unknown Tasks]()**.


## Train Deep Learning Model

 - This is `Step 2` and `Step 3` in the overall pipeline.
 - Train a deep learning model for one task (currently training DenseNet model, can replace this with custom DL model)
 - Extract features of dataset on the DL model (can chose any custom layer)

### Install Prerequisites

```
$ git clone https://github.com/dl-model-recommend/model-trust.git

$ cd model-trust

$ pip install -r requirements.txt
```

### Run the entire code

```
$ cd train_DL_model
$ python train_models.py
$ python extract_features.py
```
The paramters are available for customization:
 - `train_DL_model/model_params.py`


## Overall Output:
 - Trained DL model (DenseNet, by default)
 - Extracted features for the provided dataset 

## Questions/Bugs

Please submit a Github issue if you have any questions or find any bugs.
