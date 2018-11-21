from sklearn.pipeline import FeatureUnion, _transform_one
from sklearn.utils import Parallel, delayed
from scipy import sparse
import numpy as np

class FeatureUnionOnAxis(FeatureUnion):

        def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, axis=0):
            super().__init__(transformer_list, n_jobs, transformer_weights)
            self.axis = axis
        
        def transform(self, X):
            Xs = Parallel(n_jobs=self.n_jobs)(
                delayed(_transform_one)(trans, X, None, weight)
                for name, trans, weight in self._iter())
            if not Xs:
                # All transformers are None
                return np.zeros((X.shape[0], 0))
            Xs = np.concatenate(Xs, axis=self.axis)
            return Xs