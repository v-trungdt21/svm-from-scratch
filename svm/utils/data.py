import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_gaussian_quantiles

import svm


def generate_linear_separable_dataset_old(
    n_samples=200, n_features=2, n_classes=2, seed=8080
):
    X, Y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_classes,
        cluster_std=1.0,
        center_box=[0.0, 8.0],
        random_state=seed,
    )
    return X, Y


def generate_linear_separable_dataset():
    path = os.path.dirname(svm.__file__)
    data_path = os.path.join(path, "assets/data")
    X = np.load(data_path + "/X_linear_separable.npy")
    Y = np.load(data_path + "/Y_linear_separable.npy")

    return X, Y


def generate_nonlinear_separable_dataset_2():
    path = os.path.dirname(svm.__file__)
    data_path = os.path.join(path, "assets/data")
    X = np.load(data_path + "/X_non_linear_separable_2.npy")
    Y = np.load(data_path + "/Y_non_linear_separable_2.npy")

    return X, Y


def generate_linear_separable_dataset_overlap():
    path = os.path.dirname(svm.__file__)
    data_path = os.path.join(path, "assets/data")
    X = np.load(data_path + "/X_linear_separable_overlap.npy")
    Y = np.load(data_path + "/Y_linear_separable_overlap.npy")

    return X, Y


def generate_nonlinear_separable_dataset(
    n_samples=200, n_features=2, n_classes=2, mean=[0.0, 0.0], seed=50
):
    path = os.path.dirname(svm.__file__)
    data_path = os.path.join(path, "assets/data")
    X = np.load(data_path + "/X_non_linear_separable.npy")
    Y = np.load(data_path + "/Y_non_linear_separable.npy")

    return X, Y


def generate_nonlinear_separable_dataset_old(
    n_samples=200, n_features=2, n_classes=2, mean=[0.0, 0.0], seed=50
):
    X, Y = make_gaussian_quantiles(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        mean=mean,
        random_state=seed,
    )
    return X, Y


def plot_2d_dataset(X, Y):
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=25, edgecolor="k")
    plt.show()
