import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from pyLSHash import LSHash
from sklearn.metrics import accuracy_score
from utils import miniROCKET as mr
from utils import minirocket_multivariate as mrm
from utils import raster_multivariate as rsm
from utils import classifier as CLF
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from utils import checker



def misclassified_ones(x_train, y_train, classifier):
    predict = classifier.predict(x_train)
    result = []
    for i in range(x_train.shape[0]):
        if predict[i] != y_train[i]:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)


def find_range(number, ranges_list):
    for i, (start, end) in enumerate(ranges_list):
        if start <= number < end:
            return i
    return None


def update_biases(P_biases, ranges, n_feature):
    normalized_weight = P_biases / np.sum(P_biases)
    chosen_ranges = np.random.choice(
        np.arange(len(ranges)), p=normalized_weight, size=n_feature
    )
    biases = []
    for r in chosen_ranges:
        chosen_range = ranges[r]
        lower, upper = chosen_range
        a = np.random.uniform(lower, upper)
        biases.append(a)
    return biases





def assign_probability(x, beta=1.4):
    size = len(x)
    mid_size = size // 2
    # print(mid_size)
    result = []
    result2 = []
    default = 50
    adder = 1

    if size % 2 == 0:
        adder = 0
    for i in range(1, mid_size + 1 + adder):
        result.append(default // (i**beta))
    for i in range(1, mid_size + 1):
        result2.append(default // (i**beta))

    result.extend(result2[::-1])

    return result / np.sum(result)


def dilation_finder(x_train, n_dilation=5):
    dilation, num_per_dilation, _, biases = mr.fit(x_train, num_features=10_000)
    prob = assign_probability(dilation)
    result = np.random.choice(dilation, size=n_dilation, p=prob, replace=False)

    return np.sort(result)


def create_histogram(arr, num_bins):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    bin_size = (arr_max - arr_min) / num_bins
    histogram = np.zeros(num_bins + 1, dtype=int)
    ranges = []
    tmp_min = arr_min
    for i in range(num_bins):
        ranges.append([tmp_min, tmp_min + bin_size])
        tmp_min = tmp_min + bin_size

    for b in arr:
        histogram[find_range(b, ranges)] += 1

    return histogram, ranges


def train_classifier(x_train, y_train):
    # Normalize the training data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    # Train the classifier
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(x_train_scaled, y_train)

    # Make predictions on the normalized test data
    predictions = classifier.predict(x_train_scaled)
    accuracy = accuracy_score(y_train, predictions)
    return accuracy, x_train_scaled, classifier



def idx_finder(d, r, i, repeat=4):
    if d == 0:
        return r * 84 + i
    else:
        return d * repeat * 84 + r * 84 + i


#########
from numba import njit, jit


def Scatter_score(x_train, y_train):
    n_sample, n_feature = x_train.shape[0], x_train.shape[1]
    y_train = np.array(y_train)
    result = []
    for i in range(n_feature):
        ss = misc.scatter_score(x_train[:, i], y_train)
        result.append(ss)
    return np.array(result)


def elbow(x, y):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x)
    # Train the classifier
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(x_train_scaled, y)
    # n_targets, n_features = classifier.coef_.shape
    aggregated_coef = np.mean(classifier.coef_, axis=0)
    absoluted_coef = np.abs(aggregated_coef)
    return absoluted_coef


def erocket(x, y):
    sclr = StandardScaler()
    sclr.fit(x)
    X_training_transform_scaled = sclr.transform(x)
    # X_test_transform_scaled = sclr.transform(X_test_transform)
    clf = RidgeClassifierCV(np.logspace(-3, 3, 10))
    clf.fit(X_training_transform_scaled, y)
    w_ridgecv = clf.coef_
    u_tilde, first_point_non_neg_weight, knees_minus, knees_plus = (
        improved_multi_curve_feature_pruner_exp(w_ridgecv)
    )
    return u_tilde


def apply_pca(x_train, x_test, variance_ratio):
    """
    Apply PCA to the training and test datasets.

    :param x_train: Training dataset
    :param x_test: Test dataset
    :param variance_ratio: Ratio of variance to preserve in PCA
    :return: Transformed training and test datasets
    """
    pca = PCA(n_components=variance_ratio)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    return x_train_pca, x_test_pca


# Example usage
# x_train_pca, x_test_pca = apply_pca(x_train_scl, x_test_scl, 0.98)


# @jit(cache=True)
def select_best(x, y, k=100, method="MI"):
    n_sample, n_feature = x.shape[0], x.shape[1]
    # if method == 'MI':
    #     sorted_index = np.argsort(mutual_info_classif(x,y))[::-1]
    if method == "SS":
        sorted_index = np.argsort(Scatter_score(x, y))[::-1]
    elif method == "random":
        indexes = np.arange(n_feature)
        sorted_index = np.random.choice(indexes, k, replace=False)
    elif method == "elbow":
        sorted_index = np.argsort(elbow(x, y))[::-1]
    elif method == "erocket":
        best = erocket(x, y)
        return x[:, best], best

    best_index = sorted_index[0:k]
    return x[:, best_index], best_index
    # mutual_info_classif(x_train_transformed,y_train)


def MiniROCKET(
    x_train,
    y_train,
    x_test,
    y_test,
    function_type="ter",
    n_features=10000,
    shuffle_quant=False,
    parameter=None,
    ali=False,
):
    # parameter = mr.fit(x_train,num_features =n_features)
    if type(parameter) == type(None):
        if shuffle_quant:
            parameter = mr.fit_shuffled_quantiles(x_train, num_features=n_features)
        else:
            parameter = mr.fit(x_train, num_features=n_features)

    if ali == True:
        parameter = mr.ali_fit(x_train, num_features=n_features)

    dilations, num_features_per_dilation, _, biases = parameter
    parameter = dilations, num_features_per_dilation, biases

    x_train_trans_org = mr.transform(x_train, parameter, function_type)
    x_test_trans_org = mr.transform(x_test, parameter, function_type)
    return x_train_trans_org, x_test_trans_org, parameter





def RASTER(
    x_train,
    y_train,
    x_test,
    y_test,
    sizes=4,
    n_features=10000,
    shuffle_quant=False,
    parameter=None,
    fixed=False,
):
    if type(parameter) == type(None):
        if shuffle_quant:
            parameter = mr.fit_shuffled_quantiles(
                x_train, num_features=n_features, sizes=sizes
            )
        else:
            parameter = mr.fit(x_train, num_features=n_features, sizes=sizes)

    dilation, num_feature_dilation, my_size, biases = parameter

    if fixed == True:
        my_size = np.ones(len(my_size)) * sizes
        my_size = my_size.astype(np.int64)
        parameter = (dilation, num_feature_dilation, my_size, biases)

    x_train_trans_org = mr.transform_refined(x_train, parameter, "ter")
    x_test_trans_org = mr.transform_refined(x_test, parameter, "ter")
    return x_train_trans_org, x_test_trans_org, parameter



def MiniROCKET_MV(
    x_train,
    y_train,
    x_test,
    y_test,
    n_features=10000,
    shuffle_quant=False,
    parameter=None,
):
    # parameter = mr.fit(x_train,num_features =n_features)
    if len(x_train.shape) < 3:
        raise TypeError("it is not multi variate")

    # if type(parameter) == type(None):
    #     dilations, num_features_per_dilation,_, biases = parameter
    #     parameter = dilations, num_features_per_dilation, biases
    parameter = mrm.fit(x_train, num_features=n_features)
    x_train_trans_org = mrm.transform(x_train, parameter)
    x_test_trans_org = mrm.transform(x_test, parameter)
    return x_train_trans_org, x_test_trans_org, parameter


def RASTER_MV(
    x_train,
    y_train,
    x_test,
    y_test,
    n_features=10000,
    shuffle_quant=False,
    parameter=None,
):
    # parameter = mr.fit(x_train,num_features =n_features)
    if len(x_train.shape) < 3:
        raise TypeError("it is not multi variate")

    # if type(parameter) == type(None):
    #     dilations, num_features_per_dilation,_, biases = parameter
    #     parameter = dilations, num_features_per_dilation, biases
    parameter = rsm.fit(x_train, num_features=n_features)
    x_train_trans_org = rsm.transform(x_train, parameter)
    x_test_trans_org = rsm.transform(x_test, parameter)
    return x_train_trans_org, x_test_trans_org, parameter




def PDRASTER(x_train, y_train, x_test, y_test, n_features=10000, shuffle_quant=False):
    """
    PD stands for Probabilistic dilation
    """

    portion_size = n_features
    my_dict = {}
    ddd, _ = mr._fit_dilations(x_train.shape[1], 9996, max_dilations_per_kernel=32)
    num_dilation = len(ddd)
    for i in range(num_dilation):
        parameter_probDilation = mr.ProbabilisticDilation_fit(
            x_train, num_features=10000, index=i
        )
        x_train_trans_new, x_test_trans_new, parameter_new = MiniROCKET(
            x_train,
            y_train,
            x_test,
            y_test,
            shuffle_quant=False,
            parameter=parameter_probDilation,
        )

        accuracy, (x_train_new_scl, x_test_new_scl), clf_new = CLF.classic_classifier(
            x_train_trans_new, y_train, x_test_trans_new, y_test
        )
        my_dict[ddd[i]] = accuracy

    max_value = np.sum(list(my_dict.values()))
    probability_accuracy = {k: v / max_value for k, v in my_dict.items()}
    ####
    num_kernels = 84
    portion_size = n_features // num_kernels
    dilations, num_features_per_dilation = np.unique(
        np.random.choice(
            list(probability_accuracy.keys()),
            size=portion_size,
            p=list(probability_accuracy.values()),
        ),
        return_counts=True,
    )
    num_features_per_dilation = num_features_per_dilation.astype(np.int32)
    num_features_per_kernel = np.sum(num_features_per_dilation)
    quantiles = mr._quantiles(num_kernels * num_features_per_kernel)

    biases = mr._fit_biases(x_train, dilations, num_features_per_dilation, quantiles)
    random_size = np.random.randint(1, 5, len(biases))
    parameter_setareh = dilations, num_features_per_dilation, random_size, biases
    x_train_trans_new, x_test_trans_new, parameter_new = RASTER(
        x_train,
        y_train,
        x_test,
        y_test,
        n_features=n_features,
        shuffle_quant=shuffle_quant,
        parameter=parameter_setareh,
    )
    return x_train_trans_new, x_test_trans_new, parameter_new




def ISD_RASTER(
    x_train, y_train, x_test, y_test, n_features=10000, sizes=5, shuffle_quant=False
):

    avg_spec = checker.average_spectogram(x_train, verbose=False)
    should_use_miniROCKET = checker.is_evenly_distributed(avg_spec, verbose=False)

    if should_use_miniROCKET:
        x_train_trans_mini, x_test_trans_mini, parameter_mini = MiniROCKET(
            x_train, y_train, x_test, y_test, n_features=n_features
        )

        return (
            x_train_trans_mini,
            x_test_trans_mini,
            parameter_mini,
            should_use_miniROCKET,
        )
    else:
        x_train_trans_raster, x_test_trans_raster, parameter_raster = RASTER(
            x_train,
            y_train,
            x_test,
            y_test,
            n_features=n_features,
            sizes=sizes,
            shuffle_quant=shuffle_quant,
        )

        return (
            x_train_trans_raster,
            x_test_trans_raster,
            parameter_raster,
            should_use_miniROCKET,
        )
