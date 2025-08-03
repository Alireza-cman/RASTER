import sys

import numpy as np


from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC


def normalize(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled


def SVC_classifier(x_train, y_train, x_test, y_test, verbose=False, alpha=None):
    # Normalize the training data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    classifier = SVC(kernel="linear")
    classifier.fit(x_train_scaled, y_train)
    pred_train = classifier.predict(x_train_scaled)
    train_accuracy = accuracy_score(y_train, pred_train)
    if verbose == True:
        print("train: ", train_accuracy)
    # Make predictions on the normalized test data
    predictions = classifier.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    return accuracy, (x_train_scaled, x_test_scaled), classifier


def classic_classifier(
    x_train, y_train, x_test, y_test, balanced=True, verbose=False, alpha=None
):
    # Normalize the training data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train the classifier
    if type(None) == type(alpha):
        if balanced == True:
            classifier = RidgeClassifierCV(
                alphas=np.logspace(-3, 3, 10), class_weight="balanced"
            )
        else:
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    else:
        if balanced == True:
            classifier = RidgeClassifierCV(alphas=alpha, class_weight="balanced")
        else:
            classifier = RidgeClassifierCV(alphas=alpha)
    classifier.fit(x_train_scaled, y_train)
    pred_train = classifier.predict(x_train_scaled)
    train_accuracy = accuracy_score(y_train, pred_train)
    if verbose == True:
        print("train: ", train_accuracy)
    # Make predictions on the normalized test data
    predictions = classifier.predict(x_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    return accuracy, (x_train_scaled, x_test_scaled), classifier
