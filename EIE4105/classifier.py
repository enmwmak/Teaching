# Source: https://github.com/tgy/mnist-em-bmm-gmm

import numpy as np
import gmm

def _model_class_from_type(model_type):

    if model_type == 'gmm':
        return gmm.gmm

    raise ValueError('Unknown model type: {}'.format(model_type))

class classifier:

    def __init__(self, n_components,
                 means_init_heuristic='kmeans',
                 covariance_type='diag',
                 model_type='gmm', means=None, verbose=False):

        self.n_components = n_components
        self.means_init_heuristic = means_init_heuristic
        self.covariance_type = covariance_type
        self.model_class = _model_class_from_type(model_type)
        self.means = means
        self.verbose = verbose
        self.models = dict()

    def fit(self, x, labels):

        label_set = set(labels)

        for label in label_set:

            x_subset = x[np.in1d(labels, label)]
            self.models[label] = self.model_class(
                self.n_components, covariance_type=self.covariance_type,
                verbose=self.verbose)

            print('Training label {} ({} samples)'
                  .format(label, x_subset.shape[0]))

            self.models[label].fit(
                x_subset, means_init_heuristic=self.means_init_heuristic,
                means=self.means)

    def predict(self, x):

        n = x.shape[0]
        n_classes = len(self.models)

        likelihoods = np.ndarray(shape=(n_classes, n))

        for label in range(n_classes):
            likelihoods[label] = self.models[label].predict(x)

        predicted_labels = np.argmax(likelihoods, axis=0)

        return predicted_labels
