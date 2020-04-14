from sklearn.base import BaseEstimator


class NearestNeighbourModel(BaseEstimator):
    # CHECKME should we inherit from the BaseEstimator?
    """
    This is just a dummy replacement for an sklearn model.
    It is used to not process the test and training molecules in the ML script,
    and to recognize when to calculate nearest neighbor distances.
    """

    def fit(self, x, Y):
        return None

    def predict_proba(self, x):
        return None