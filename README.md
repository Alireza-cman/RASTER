# RASTER: Representation Learning for Time Series Classification using Scatter Score and Randomized Threshold Exceedance Rate

**RASTER** is an advanced version of the MiniROCKET method, designed to capture temporal changes more effectively. This approach introduces a new temporal-aware downsampling strategy named **Randomized Threshold Exceedance Rate (rTER)**. Our method demonstrates significant enhancements in classification performance over state-of-the-art methods, including ROCKET, miniROCKET, ResNet, and InceptionTime. These improvements are evident across 30 different datasets.

For more detailed insights, refer to our paper: [RASTER: Time Series Random Representation Method](https://ieeexplore.ieee.org/abstract/document/10285973).

## Requirements

The recommended requirements for TS2Vec are specified as follows:
* Python 3.8
* numba==0.56.4
* numpy==1.23.5
* pandas==1.5.2
* scikit_learn==1.2.0
* sktime==0.24.1

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```



## Data

The datasets can be obtained and put into `datasets/` folder in the following way:

* [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) should be put into `datasets/UCR/` so that each data file can be located by `datasets/UCR/<dataset_name>/<dataset_name>_*.csv`.

  
## Code Example

```python

import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import RASTER
import utils


x_train,y_train, x_test, y_test = utils.load_dataset('CinCECGTorso')
x_train_trans, x_test_trans = RASTER.RASTER(x_train ,y_train  ,x_test,y_test, n_features=10_000)
accuracy , _, clf = utils.classic_classifier(x_train_trans,y_train , x_test_trans,y_test)
print(accuracy)

```
