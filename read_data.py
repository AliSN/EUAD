import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.utils import shuffle as sh

from sklearn.datasets import fetch_openml
# from .fetch_ml_mieux import fetch_spambase, fetch_annthyroid, fetch_arrhythmia
# from .fetch_ml_mieux import fetch_pendigits, fetch_pima, fetch_wilt
# from .fetch_ml_mieux import fetch_internet_ads, fetch_adult
from sklearn.datasets import fetch_kddcup99
from sklearn.datasets import fetch_covtype

from sklearn.preprocessing import LabelBinarizer, scale
from sklearn.utils import shuffle as sh

import synthetic_dataset as sd

from scipy.io import loadmat
import arff
from sklearn.utils import check_array

def read_arff(file_path):

    file = arff.load(open(file_path))
    data_value = np.asarray(file['data'])
    attributes = file['attributes']

    X = data_value[:, 0:-2]
    y = data_value[:, -1]

    y[y == 'no'] = 0
    y[y == 'yes'] = 1
    y = y.astype('float').astype('int').ravel()

    if y.sum() > len(y):
        print(attributes)
        raise ValueError('wrong sum')

    return X, y, attributes

# Define data file and read X and y

__all__ = ["read_data"]


def read_data(dat, scaling=True, shuffle=True, percent10_kdd=False, continuous=True, anomaly_max=0.1):

    print('loading ' + dat)

    if dat in ['Abnormal_close', 'Abnormal_far']:
        path_csv = os.path.join('./data', dat + '.csv')
        csv_file = pd.read_csv(path_csv)
        X = csv_file.drop(columns = 'Label')
        y = csv_file['Label']
        

    if dat in ['level1_far', 'level1_close', 'level1_cluster', 'level1_cluster2', 'level2_close', 'level3_far', 'level3_close',
              'level4_far', 'level4_close'] :
        X, y = sd.syntheticdataset(dat)

    if dat in ['mammography', 'thyroid', 'satimage-2', 'speech']:
        path_mat = os.path.join('./data/ODDS', dat + '.mat')
        mat = loadmat(path_mat)
        X = mat['X']
        y = mat['y'].ravel()

        
    if dat in ['Wilt', 'Glass', 'PenDigits', 'Shuttle']:
        path_arff = os.path.join('./data/DAMI', dat + '.arff')
        X, y, attributes = read_arff(path_arff)
        X = check_array(X).astype('float64')
        y = y.ravel()

    # take max 10 % of abnormal data:
    if anomaly_max is not None:
        index_normal = (y == 0)
        index_abnormal = (y == 1)
        if index_abnormal.sum() > anomaly_max * index_normal.sum():
            X_normal = X[index_normal]
            X_abnormal = X[index_abnormal]
            n_anomalies = X_abnormal.shape[0]
            n_anomalies_max = int(0.1 * index_normal.sum())
            r = sh(np.arange(n_anomalies))[:n_anomalies_max]
            X = np.r_[X_normal, X_abnormal[r]]
            y = np.array([0] * X_normal.shape[0] + [1] * n_anomalies_max)
    
    X = X.astype(float)
    
    # scale dataset:
    if scaling:
        X = scale(X)

    # shuffle dataset:
    if shuffle is True:
        X, y = sh(X, y)
        

    return X, y