# Machine Learning Tool Box
This is the machine learning tool box. A collection of userful machine learning tools intended for reuse and extension.

The toolbox contains the following modules:
* keras - Keras callbacks for metrics and various Keras tools
* lightgbm - metric tool functions for LightGBM
* metrics - several metric implementations 
* plot - plot and visualisation tools
* tools - various statistical tools

## Module: lightgbm
This module implements metric functions that are not included in LightGBM. 
At the moment this is the F1- and accuracy-score for binary and multi class problems.
The usage looks like this:
```
bst = lgb.train(param, 
                train_data, 
                valid_sets=[validation_data]
                early_stopping_rounds=10,
                evals_result=evals_result,
                feval=mltb.lightgbm.multi_class_f1_score_factory(num_classes, 'macro'),
               )
```
