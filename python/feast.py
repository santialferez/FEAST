#########################################################################
##  Based on https://github.com/EvgeniDubov/FEAST/tree/python_wrapper  ##
#########################################################################

import os
import sys
import pandas as pd
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes as c
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing

class feast(BaseEstimator, TransformerMixin):
    """ Class for FEAST features selection
        based on C implementation in Feature Selection Tools box
        https://github.com/Craigacp/FEAST

        Parameters
        ----------
        num_of_features : integer, default = 20
            the amount of features to choose.
        scale_method : {None, 'softmax', 'quantile', 'robust'}, str, default = None
            Feast needs discretize the continuous values (Real or float type),
            The method discretize the values to 10 values (0 to 9). This parameter
            controls the preprocessing method before this discretization.

        Attributes
        ----------
        feature_names : list of strings
            Names of the selected features.
            * will be available only in case data for fit is pandas DataFrame
        feature_scores : list of floats
            Scores of selected feature. The score is calculated by min{I(Xn ; Y | Xk)}
        feature_indexes : list of floats
            Indexes of selected feature.
        """

    def __init__(self, num_of_features=20, criterion="CMIM", scale_method = None):
        if sys.platform.startswith('linux'):
            # in case of Linux OS load feast linux library (.so file)
            # self.libFSToolbox = c.CDLL('{}/{}/{}'.format(os.path.dirname(__file__), 'lib', 'libFSToolbox.so'))
            self.libFSToolbox = c.CDLL(os.path.dirname(__file__) + "/../libFSToolbox.so")
        elif sys.platform.startswith('win'):
            # in case of Windows OS load feast windows library (.dll file)
            self.libFSToolbox = c.WinDLL('{}/{}/{}'.format(os.path.dirname(__file__), 'lib', 'libFSToolbox.dll'))
        else:
            # OSs other than Linux and Windows are not supported
            raise NotImplementedError('Feast is not supported on {} operating system'.format(sys.platform))

        self.num_of_features = num_of_features
        self.feature_indexes = None
        self.feature_scores = None
        self.feature_names = None
        self.criterion = criterion # it could be CMIM, JMI or mRMR_D
        self.scale_method = scale_method

    def fit(self, data, labels):
        """
        Fits a defined filter depending of the criterion to a given data set.
        data and labels are expected to be discretized

        Parameters
        ----------
        data : pandas or numpy data object
            The training input samples.
        labels : pandas or numpy data object. labels should contain a binary label.
            The label of the training samples.
        """

        # in case the data is a pandas dataframe we can store features by name
        data_labels = None
        if isinstance(data, pd.DataFrame):
            data_labels = data.columns.tolist()

        ####### CODIFICATION AND DISCRETIZATION NECESSARY TO USE FEAST #########

        # Codification from a string label to intengers
        if labels.dtype != int:
            encoder = preprocessing.LabelEncoder()
            labels = encoder.fit_transform(labels)

        # Normalization of data, necesary to the correct use of Feast

        if self.scale_method == "softmax":
            # Softmax
            data = 1/( 1 - np.exp(-preprocessing.scale(data)) )

        if self.scale_method == "quantile":
            # Quantile transform
            data = preprocessing.quantile_transform(data)

        if self.scale_method == "robust":
           # Robust scale_method
           data = preprocessing.robust_scale(data)

        # normalize each feature
        data = preprocessing.minmax_scale(data)

        # discretize each feature
        n_bins = 10 # number of bins by default
        X_n_samples, X_n_features = data.shape
        X_discrete = np.zeros((X_n_samples, X_n_features))
        bins = np.linspace(0, 1, n_bins)
        for i in range(X_n_features):
            X_discrete[:, i] = np.digitize(data[:, i], bins)
        X_discrete = X_discrete-1

        data = X_discrete
        ########################################################################

        # python variables adaptation for C parameters initialization
        data = np.array(data, dtype=np.uint32, order="F")
        labels = np.array(labels, dtype=np.uint32)
        n_samples, n_features = data.shape
        output = np.zeros(self.num_of_features).astype(np.uint)
        selected_features_score = np.zeros(self.num_of_features)

        # cast as C types
        _uintpp = ndpointer(dtype=np.uintp, ndim=1, flags='F')
        c_k = c.c_uint(self.num_of_features)
        c_no_of_samples = c.c_uint(n_samples)
        c_no_of_features = c.c_uint(n_features)
        c_feature_matrix = (data.__array_interface__['data'][0] + np.arange(data.shape[1]) * (data.strides[1])).astype(
            np.uintp)
        c_class_column = labels.ctypes.data_as(c.POINTER(c.c_uint))
        c_output_features = output.ctypes.data_as(c.POINTER(c.c_uint))
        c_feature_scores = selected_features_score.ctypes.data_as(c.POINTER(c.c_double))

        getattr(self.libFSToolbox,self.criterion).argtypes = [c.c_uint, c.c_uint, c.c_uint, _uintpp, c.POINTER(c.c_uint), c.POINTER(c.c_uint), c.POINTER(c.c_double)]

        getattr(self.libFSToolbox,self.criterion).restype = c.POINTER(c.c_uint)

        # call the C lib implementation
        c_selected_features = getattr(self.libFSToolbox,self.criterion)(
            c_k, c_no_of_samples, c_no_of_features, c_feature_matrix,
            c_class_column, c_output_features, c_feature_scores)

        # result transition from C to Python
        features_iterator = np.fromiter(c_selected_features, dtype=np.uint, count=self.num_of_features)
        selected_features = []
        for c_selected_feature_index in features_iterator:
            selected_features.append(c_selected_feature_index)

        # store the selection results
        self.feature_scores = [c_feature_scores[idx] for idx in range(self.num_of_features)]
        self.feature_indexes = selected_features

        if data_labels is not None:
            self.feature_names = [data_labels[idx] for idx in self.feature_indexes]

        return self

    def transform(self, X, y = None):
        if isinstance(X, np.ndarray):
            return(X[:, self.feature_indexes])
        elif isinstance(X, pd.DataFrame):
            return(X.loc[:, self.feature_names])
        else:
            raise AttributeError('Transform accepts data of type numpy.ndarray or pandas.DataFrame')
