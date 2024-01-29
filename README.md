# RASTER: Representation Learning for Time Series Classification using Scatter Score and Randomized Threshold Exceedance Rate

**RASTER** is an advanced version of the MiniROCKET method, designed to capture temporal changes more effectively. This approach introduces a new temporal-aware downsampling strategy named **Randomized Threshold Exceedance Rate (rTER)**. Our method demonstrates significant enhancements in classification performance over state-of-the-art methods, including ROCKET, miniROCKET, ResNet, and InceptionTime. These improvements are evident across 30 different datasets.

For more detailed insights, refer to our paper: [RASTER: Time Series Random Representation Method](https://ieeexplore.ieee.org/abstract/document/10285973).


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
