# [ICLR-2020](https://openreview.net/forum?id=B1lf4yBYPr)

This repository contains the original implementation of the paper - **[Is my deep learning model learning more than I want it to?](https://openreview.net/forum?id=B1lf4yBYPr)**.

## Compute Trust Score

 - This is `Step 5` in the overall pipeline.
 - Computer Model trust score

### Install Prerequisites

```
$ git clone https://github.com/dl-model-recommend/model-trust.git

$ cd model-trust

$ pip install -r requirements.txt
```

### Run the entire code

```
$ cd compute_model_trust
$ python calculate_model_trust.py
```
The paramters are available for customization:
 - `compute_model_trust/trust_params.py`


## Overall Output:
 - Trust Score of the DL model on the input dataset

## Questions/Bugs

Please submit a Github issue if you have any questions or find any bugs.