from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt
import glob
import os 
import numpy as np 
import time
import cv2
import sys
from statsmodels.distributions.empirical_distribution import ECDF
from noise_model import *

# See https://en.wikipedia.org/wiki/Confusion_matrix for more info
CONFUSION_MATRIX = np.array(
        [[0.85714286,0.,0.14285714,0.,0.,0.,0.],
        [0.06666667,0.8,0.06666667,0.,0.,0.06666667,0.],
        [0.,0.,1.,0.,0.,0.,0.],
        [0.03636364,0.,0.01818182,0.94545455,0.,0.,0.],
        [0.,0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,0.,1.,0.],
        [0.,0.,0.,0.01408451,0.,0.,0.98591549]]).T
            


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

GENERATE_NEW_BBS = True
GENERATE_IMGS = False


CLASSES = ['mountainbike',
           'car',
           'truck',
           'dog',
           'horse',
           'sheep',
           'giraffe']


def get_overlap_fraction(bb1, bb2, getAreaDiff=False):
    """Returns the fraction of overlap (as a fraction of
    the smallest of the BBs) between two BBs.
    
    Parameters
    ----------
    bb1 : tuple(int, int, int, int)
        BB extent defined by min and max x-coordinates and
        min and max y-coordinates.
    bb2 : tuple(int, int, int, int)
        BB extent defined by min and max x-coordinates and
        min and max y-coordinates.
    getAreaDiff : bool, optional
        if true, will return the difference in area, by default False
    
    Returns
    -------
    float
        The overlap fraction, or area difference.
    """

    xmin1, xmax1, ymin1, ymax1, cls1 = bb1
    xmin2, xmax2, ymin2, ymax2, cls2 = bb2

    xmin1 = int(xmin1)
    xmax1 = int(xmax1)
    ymin1 = int(ymin1)
    ymax1 = int(ymax1)
    xmin2 = int(xmin2)
    xmax2 = int(xmax2)
    ymin2 = int(ymin2)
    ymax2 = int(ymax2)

    dx = np.min((xmax1, xmax2)) - np.max((xmin1, xmin2))
    dy = np.min((ymax1, ymax2)) - np.max((ymin1, ymin2))

    dx = float(dx)
    dy = float(dy)

    if (dx>=0) and (dy>=0):
        overlapping = dx*dy
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        not_overlapping = area1 + area2 - 2 * overlapping
        if getAreaDiff: return float(np.abs(area1 - area2))/area1 if area1 != 0 else 0
        if overlapping == 0: return 0
        return overlapping /(not_overlapping+overlapping)
    else:
        return 0


def dist(p1, p2):
    return np.linalg.norm(p2-p1)


def center(bb):
    xmin, xmax, ymin, ymax, label = bb
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)

    x = xmax - xmin
    x /= 2
    y = ymax - ymin
    y /= 2

    return np.array([x,y])


def area(bb):
    xmin, xmax, ymin, ymax, label = bb
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
    return (xmax - xmin) * (ymax - ymin)


def get_min_dim(bb):
    xmin, xmax, ymin, ymax, label = bb
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
    return np.min((xmax-xmin, ymax-ymin))


def get_max_dim(bb):
    xmin, xmax, ymin, ymax, label = bb
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
    return np.max((xmax-xmin, ymax-ymin))


def plot_confusion_matrix(cm,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    plt.show() must be run to view the plot, this is not done
    by this function.
    
    Parameters
    ----------
    cm : np.ndarray
        the confusion matrix to plot
    normalize : bool, optional
        whether to normalise the matrix, by default True
    title : string, optional
        title of the plot, otherwise otherwise a sensible title is generated. By default None
    cmap : matplotlib.colormap, optional
        matplotlib colormap, by default plt.cm.Blues
    
    Returns
    -------
    matplotlib.Axes
        axes object of the plot generated
    """
    if not title:
        title = 'Normalized confusion matrix' if normalize else 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(4,2))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=CLASSES, yticklabels=CLASSES,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    return ax


def analyse(get_ps = True, lerr=0.8, cerr=0.8, plot=False):
    # Make a sensible function which shows the
    # distribution of errors given an array of BBs
    raise NotImplementedError("Currently in development")


if __name__ == '__main__':
    lerr = 0.175
    cerr = 0.7
    analyse(True, lerr, cerr, True)
