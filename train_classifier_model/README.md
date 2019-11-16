# [Reducing Model Overlearning]

This repository contains the original implementation of the paper - **[Reducing Overlearning through Disentangled Representations by Suppressing Unknown Tasks]()**.


## Train Classifier Model

 - This is `Step 4` in the overall pipeline.
 - Train neural network classifier model for other auxillary tasks (can replace this with custom classifier)

### Install Prerequisites

```
$ git clone https://github.com/dl-model-recommend/model-trust.git

$ cd model-trust

$ pip install -r requirements.txt
```

### Run the entire code

```
$ cd train_classifier_model
$ python classification_net.py
```
The paramters are available for customization:
 - `train_classifier_model/classification_params.py`


## Overall Output:
 - Trained classifier model
 - Performance of the classifier model on the multiple auxillary tasks

## Questions/Bugs

Please submit a Github issue if you have any questions or find any bugs.
