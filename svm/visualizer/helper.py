from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

from svm.core.svm import SVM
from svm.core.svm_cvxopt import SVM_cvxopt
from svm.core.svm_cvxopt_backup import SVM_cvx_backup


class Param:
    """Param class
    This class contains all the parameters involved in the plotting process.
    In order to minimize the number of params pass to the callback.
    """

    def __init__(self):
        self.allow_press = True
        self.allow_infer = True
        self.showed_tooltip = False

        self.ax_x_lim = [0.0, 10.0]
        self.ax_y_lim = [0.0, 10.0]

        self.kernel_list = ["linear", "poly", "rbf", "sigmoid"]
        self.kernel_idx = 0
        self.model_list = ["sklearn", "cvxopt", "svm_backup", "smo"]
        self.model_idx = 0
        self.model_dict = {
            "sklearn": SVC,
            "cvxopt": SVM_cvxopt,
            "svm_backup": SVM_cvx_backup,
            "smo": SVM,
        }

        # Model parameters
        self.C = 1.0
        self.tol = 0.001
        self.degree = 3
        self.gamma = "scale"
        self.max_iter = -1
        self.coef0 = 0.0

    def get_kernel(self):
        return self.kernel_list[self.kernel_idx]

    def get_model_name(self):
        return self.model_list[self.model_idx]

    def get_model(self):
        return self.model_dict[self.get_model_name()]

    def get_model_params(self):
        return dict(
            C=self.C,
            tol=self.tol,
            degree=self.degree,
            gamma=self.gamma,
            max_iter=self.max_iter,
            coef0=self.coef0,
            kernel=self.get_kernel(),
        )


def plot_scatter(ax, X1, X2, Y):
    ax.scatter(
        X1,
        X2,
        marker="o",
        c=Y,
        s=25,
        edgecolor="k",
    )


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the contour of the meshgrid after infer.

    Args:
        ax (Axes): the ax of the figure.
        clf (SVM): SVM model
        xx (array): The coordinates of the x-axis.
        yy (array): The coordinates of the y-axis.

    Returns:
        [type]: [description]
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, **params)


def generate_data_then_plot(fig, ax, X, Y, plot_param, generate_func):
    reset(fig, ax, X, Y)
    plot_param.allow_press = False
    plot_param.allow_infer = True
    plot_param.showed_tooltip = False

    a, b = generate_func()
    X.extend(a)
    Y.extend(b)
    plot_scatter(ax, a[:, 0], a[:, 1], b)
    fig.canvas.draw()


def make_meshgrid(xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0, h=0.02):
    """Make meshgrid for SVM color plot given a region.

    Args:
        x{min,max} (float, optional): x_limit.
        y{min,max} (float, optional): y_limit.
        h (float, optional): step size for each grid. Defaults to 0.02.

    Returns:
        xx, yy (np.array, np.array): meshgrid
    """
    x_min, x_max = xmin - 1, xmax + 1
    y_min, y_max = ymin - 1, ymax + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    )
    return xx, yy


def init_ax(fig, ax, refresh=True):
    """Clear the axes and redraw

    Args:
        refresh: True to refresh the plot immediately.
    """
    ax.clear()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    if refresh:
        fig.canvas.draw()


def reset(fig, ax, X, Y):
    """Remove all points and clear the plot"""
    init_ax(fig, ax)
    X.clear()
    Y.clear()


def redraw(fig, ax, X, Y, refresh=True):
    """Redraw the plot with the given data points."""
    init_ax(fig, ax)
    plot_scatter(ax, X[:, 0], X[:, 1], Y)

    if refresh:
        fig.canvas.draw()


def plot_svc_decision_function(model, ax, plot_support=True):
    """Plot the SVM decision function and bold the support vectors

    Args:
        model (SVM): The model to classify
        ax (Axes): ax of the figure
        plot_support (bool, optional): True to bold the support vectors. Defaults to True.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(
        X,
        Y,
        P,
        # colors="k",
        levels=[-1, 0, 1],
        alpha=1,
        linestyles=["--", "-", "--"],
    )

    # plot support vectors
    if plot_support:
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=300,
            linewidth=1,
            facecolors="none",
            edgecolor="black",
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def infer(fig, ax, clf, X, Y, plot_param):
    print("Fitting....")
    clf.fit(X, Y)
    print("Plotting....")
    xx, yy = make_meshgrid()

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    plot_svc_decision_function(clf, ax)
    fig.canvas.draw()
