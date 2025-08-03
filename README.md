# RASTER: Time Series Classification Framework

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE Xplore](https://img.shields.io/badge/IEEE-10285973-blue.svg)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10285973)

RASTER is a comprehensive time series classification framework that implements multiple state-of-the-art algorithms including RASTER, MiniROCKET, and various multivariate extensions. This repository provides a complete toolkit for time series classification with pre-trained models, evaluation metrics, and benchmark datasets.

## 🚀 Features

- **Multiple Algorithms**: RASTER, MiniROCKET, and their multivariate variants
- **Benchmark Datasets**: ACSF1 and CinCECGTorso datasets included
- **Comprehensive Evaluation**: Performance comparison across multiple algorithms
- **Easy-to-use Interface**: Simple API for training and evaluation
- **Jupyter Notebooks**: Interactive examples and tutorials

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithms](#algorithms)
- [Datasets](#datasets)
- [Usage Examples](#usage-examples)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/alireza-cman/RASTER-github.git
cd RASTER-github
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
import numpy as np
from utils import classifier as CLF
from utils import RASTER
from utils import dataset as DS
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset = 'ACSF1'
x_train, y_train, x_test, y_test = DS.load_dataset(dataset, verbose=True)

# Preprocess data
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
enc = LabelEncoder()
y_train = enc.fit_transform(y_train)
y_test = enc.transform(y_test)

# Run RASTER
n_features = 10000
sizes = x_train.shape[1]//200 + 1
x_train_trans_raster, x_test_trans_raster, parameter_raster = RASTER.RASTER(
    x_train, y_train, x_test, y_test, 
    n_features=n_features, sizes=sizes, shuffle_quant=False
)

# Evaluate
accuracy, (x_train_raster_scl, x_test_raster_scl), clf_raster = CLF.classic_classifier(
    x_train_trans_raster, y_train, x_test_trans_raster, y_test
)
print("RASTER accuracy:", accuracy)
```

## 🧠 Algorithms

### RASTER
RASTER (RAndom Kitchen Sink with randomized Threshold Exceeda-
nce Rate summarizer) is a novel time series classification algorithm that provides efficient feature extraction and classification.

### MiniROCKET 

MiniROCKET is a fast and accurate time series classification algorithm based on random convolutional kernels.

- **Repository:** [https://github.com/angus924/minirocket](https://github.com/angus924/minirocket)

### Multivariate Extensions
- **RASTER_MV**: Multivariate version of RASTER
- **MiniROCKET_MV**: Multivariate version of MiniROCKET

### Additional Variants
- **PDRASTER**: Parameter-dependent RASTER
- **ISD_RASTER**: Improved energy Distribution RASTER

## 📊 Datasets

### ACSF1 Dataset
- **Description**: Power consumption signatures of home appliances
- **Classes**: 10 categories (mobile phones, coffee machines, computers, etc.)
- **Train/Test**: 100/100 samples
- **Length**: 1460 time steps
- **Source**: ACS-F1 database of appliance consumption signatures

### CinCECGTorso Dataset
- **Description**: ECG data from multiple torso-surface sites
- **Classes**: 4 different persons
- **Train/Test**: 40/1380 samples
- **Length**: 1639 time steps
- **Source**: Computers in Cardiology challenge

## 📖 Usage Examples

### Basic Classification
```python
# Load and preprocess data
dataset = 'ACSF1'
x_train, y_train, x_test, y_test = DS.load_dataset(dataset)

# Run RASTER classification
accuracy = run_raster_classification(x_train, y_train, x_test, y_test)
print(f"RASTER Accuracy: {accuracy:.3f}")
```

### Compare Multiple Algorithms
```python
# Compare RASTER vs MiniROCKET
raster_acc = run_raster_classification(x_train, y_train, x_test, y_test)
minirocket_acc = run_minirocket_classification(x_train, y_train, x_test, y_test)

print(f"RASTER: {raster_acc:.3f}")
print(f"MiniROCKET: {minirocket_acc:.3f}")
```

### Multivariate Time Series
```python
# For multivariate datasets
x_train_trans, x_test_trans, params = RASTER.RASTER_MV(
    x_train, y_train, x_test, y_test, n_features=10000
)
```

## 📈 Results

The repository includes comprehensive evaluation results comparing RASTER against other state-of-the-art algorithms:

- **RASTER**: Our proposed algorithm
- **MiniROCKET**: Baseline comparison
- **ROCKET**: Original ROCKET algorithm
- **Hydra**: Hydra algorithm
- **ResNet**: Deep learning baseline
- **InceptionNet**: CNN-based approach

Results are stored in CSV format in the `results/` directory for detailed analysis.

## 📁 Project Structure

```
RASTER-github/
├── utils/                          # Core algorithm implementations
│   ├── RASTER.py                   # Main RASTER algorithm
│   ├── miniROCKET.py              # MiniROCKET implementation
│   ├── raster_multivariate.py     # Multivariate RASTER
│   ├── minirocket_multivariate.py # Multivariate MiniROCKET
│   ├── classifier.py              # Classification utilities
│   ├── dataset.py                 # Dataset loading utilities
│   └── checker.py                 # Validation utilities
├── datasets/                       # Benchmark datasets
│   ├── ACSF1/                     # ACSF1 dataset
│   ├── CinCECGTorso/              # CinCECGTorso dataset
│   └── TSCDescription.csv         # Dataset descriptions
├── results/                        # Evaluation results
│   ├── raster.csv                 # RASTER results
│   ├── minirocket.csv             # MiniROCKET results
│   └── ...                        # Other algorithm results
├── raster.ipynb                   # Main demonstration notebook
├── Analysis.ipynb                 # Analysis notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{keshavarzian2023raster,
  title={Raster: Representation learning for time series classification using scatter score and randomized threshold exceedance rate},
  author={Keshavarzian, Alireza and Valaee, Shahrokh},
  booktitle={2023 IEEE 33rd International Workshop on Machine Learning for Signal Processing (MLSP)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-
- The ACSF1 dataset creators: Gisler, Christophe, et al.
- The CinCECGTorso dataset from PhysioNet
- The MiniROCKET authors for the baseline implementation
- The time series classification community for valuable feedback

## 📞 Contact

For questions and support, please open an issue on GitHub or contact us at [alireza.keshavarzian@mail.utoronto.ca].

---

**Note**: This is a research implementation. For production use, please ensure proper testing and validation. 