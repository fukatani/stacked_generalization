## Purpose
Implemented machine learning ***stacking technic[1]*** as handy  library.


## feature:

#####1) Any scikit-learn model is availavle for Stage 0 and Stage 1 model. And stacked model itself has the same interface as scikit-learn library.

ex.
```python
from stacked_generalization.lib.stacking import StackedClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn import datasets, metrics
iris = datasets.load_iris()

# Stage 1 model
bclf = LogisticRegression(random_state=1)

# Stage 0 models
clfs = [RandomForestClassifier(n_estimators=40, criterion = 'gini', random_state=1),
        ExtraTreesClassifier(n_estimators=30, criterion = 'gini', random_state=3),
        GradientBoostingClassifier(n_estimators=25, random_state=1),
        RidgeClassifier(random_state=1)]

# same interface as scikit-learn
sl = StackedClassifier(bclf, clfs)
sl.fit(iris.target, iris.data)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)
```

Stacked learning model is used as sk-learn model, so you can replace model such as *RandomForestClassifier* to *stacked model* easily in your scripts.

#####2) Evaluation model by out-of-bugs score.
In general, CV (cross validation) is used for evaluting models.
But stacking technic itself uses CV to stage0. So if you use CV for entire stacked model, *each stage 0 model are fitted n_folds squared times.*
Recent data analysis competitor sometimes uses over 1000 models for stage0, computational cost can be significent.
Therefore, as refered in [2], we implemented CV only for stage1 model.
For example, when we get 3 blends (n_folds=3), 2 blends are used for stage 1 fitting. The remaining one data is used for model test. Repitation this cycle for all 3 blends, and averaging these scores, you can get score as oob (out-of-bugs) *with only n_fold times stage0 fitting.*

ex.
```python
sl = StackedClassifier(bclf, clfs, oob_score_flag=True)
sl.fit(iris.target, iris.data)
print("Accuracy: %f" % sl.oob_score_)
```

## Software Requirement

* Python (2.7 or later)
* scikit-learn

## Installation

```
git clone https://github.com/fukatani/stacked_generalization.git
```

## Example

Here is iris classification example.
https://github.com/fukatani/stacked_generalization/blob/master/example/cross_validation_for_iris.py

## Todo
* Regression
* Gridsearch for stacking architecture
* Save blend sample
* Feature Weighted Linear Stacking

## License

MIT License.
(http://opensource.org/licenses/mit-license.php)


## Copyright

Copyright (C) 2016, Ryosuke Fukatani

Many part of the implementation is based on the following. Thanks!
https://github.com/log0/vertebral/blob/master/stacked_generalization.py

## Other
Any contributions (implement, documentation, test or idea...) are welcome.

## References
[1] L. Breiman, "Stacked Regressions", Machine Learning, 24, 49-64 (1996).
[2] J. Sill1 et al, "Feature Weighted Linear Stacking", https://arxiv.org/abs/0911.0460, 2009.