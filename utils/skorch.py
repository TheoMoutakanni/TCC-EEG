import numpy as np

from skorch import NeuralNet, NeuralNetRegressor
from skorch.callbacks import EpochScoring, ProgressBar
from skorch.helper import predefined_split
from skorch.utils import to_numpy

from sklearn.base import TransformerMixin

from braindecode import EEGClassifier, EEGRegressor


class EEGTransformer(EEGClassifier, TransformerMixin):
    def __init__(self, *args, **kwargs):
        super(EEGTransformer, self).__init__(*args, **kwargs)
    
    def get_loss(self, y_pred, y_true, X, **kwargs):
        if len(y_pred) == 2:
            y_pred, _ = y_pred
        return super().get_loss(y_pred, y_true=y_true, X=X, **kwargs)
    
    def transform(self, X):
        out = []
        for outs in self.forward_iter(X, training=False):
            outs = outs[1] if isinstance(outs, tuple) else outs
            out.append(to_numpy(outs))
        transforms = np.concatenate(out, 0)
        return transforms