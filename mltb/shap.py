"""Shap tools."""

import shap
import numpy as np

def tree_feature_importance(model, x):
    """Calculate feature importance for tree models based on SHAP values.

    Parameters
    ----------
    model : model object
        The tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost,
        and most tree-based scikit-learn models are supported.
    x : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
        A matrix of samples (# samples x # features) on which to explain the model's output.

    Returns
    -------

    See Also
    --------
    * `SHAP (SHapley Additive exPlanations): <https://github.com/slundberg/shap>`

    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    feature_importance = np.sum(np.abs(shap_values), axis=0)
    return feature_importance
