from abc import ABC, abstractmethod

import numpy
from rdkit.ML.Data import DataUtils

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, forest


class ScoringModel(ABC):
    """
    Abstract class that defines the interface for different scoring methods
    """
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def train(self, fingerprints, classifications):
        """

        :param fingerprints: nparray of fingerprints
        :param classifications: nparray of binary classifications (active/inactive),
            must map to the 'fingerprints' elements
        :return:
        """
        pass

    @abstractmethod
    def predict(self, fingerprints):
        """
        After an eventual training, predict the

        :param fingerprint:
        :return: The scores of the input fingerprints
        """
        pass


class LogisticScoring(ScoringModel):

    def __init__(self, sklearn_kwargs):
        # just pass all arguments forward, since this class acts mainly as a proxy
        # no argument checking is done beforehand, they are delegated as is to the sklearn implementation
        self._ml = LogisticRegression(**sklearn_kwargs)

    @property
    def name(self):
        # TODO include params
        # TODO build intermediate parent class with self._ml and name etc
        return self._ml.__class__.__name__

    def train(self, fingerprints, classifications):
        self._ml.fit(fingerprints, classifications)

    def predict(self, fingerprints):
        return self._ml.predict_proba(fingerprints)


class RandomForestScoring(ScoringModel):

    def __init__(self, sklearn_kwargs):
        # just pass all arguments forward, since this class acts mainly as a proxy
        # no argument checking is done beforehand, they are delegated as is to the sklearn implementation
        self._ml = RandomForestClassifier(**sklearn_kwargs)

    @property
    def name(self):
        # TODO include params
        return self._ml.__class__.__name__

    def train(self, fingerprints, classifications):
        # self._ml.fit(fingerprints, classifications)
        raise NotImplementedError

    def predict(self, fingerprints):
        # return self._ml.predict_proba(fingerprints)
        raise NotImplementedError

    @staticmethod
    def _balanced_parallel_build_trees(n_trees, forest, X, y, sample_weight, sample_mask, X_argsorted, seed, verbose):
        # TODO refactor, update to newer version
        # XXX seems like they did this in order to balance the number of actives/inactives to the same amount
        """Private function used to build a batch of trees within a job"""
        from sklearn.utils import check_random_state
        from sklearn.utils.fixes import bincount
        import random
        MAX_INT = numpy.iinfo(numpy.int32).max
        random_state = check_random_state(seed)

        trees = []
        for i in xrange(n_trees):
            if verbose > 1:
                print("building tree %d of %d" % (i+1, n_trees))
            seed = random_state.randint(MAX_INT)

            tree = forest._make_estimator(append = False)
            tree.set_params(compute_importances=forest.compute_importances)
            tree.set_params(random_state = check_random_state(seed))

            if forest.bootstrap:
                n_samples = X.shape[0]
                if sample_weight is None:
                    curr_sample_weight = numpy.ones((n_samples,), dtype=numpy.float64)
                else:
                    curr_sample_weight = sample_weight.copy()

                ty = list(enumerate(y))
                indices = DataUtils.FilterData(ty, val=1, frac=0.5, col=1, indicesToUse=0, indicesOnly=1)[0]
                indices2 = random_state.randint(0, len(indices), len(indices))
                indices = [indices[j] for j in indices2]
                sample_counts = bincount(indices, minlength=n_samples)

                curr_sample_weight *= sample_counts
                curr_sample_mask = sample_mask.copy()
                curr_sample_mask[sample_counts==0] = False

                tree.fit(X, y, sample_weight=curr_sample_weight, sample_mask=curr_sample_mask, X_argsorted=X_argsorted, check_input=False)
                tree.indices = curr_sample_mask
            else:
                tree.fit(X, y, sample_weight=sample_weight, sample_mask=sample_mask, X_argsorted=X_argsorted, check_input=False)
            trees.append(tree)
        return trees