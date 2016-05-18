[![Build Status](https://travis-ci.org/fukatani/stacked_generalization.svg?branch=master)](https://travis-ci.org/fukatani/stacked_generalization)

# stacked_generalization
Implemented machine learning ***stacking technic[1]*** as handy library in Python.


## feature:

#####1) Any scikit-learn model is availavle for Stage 0 and Stage 1 model. And stacked model itself has the same interface as scikit-learn library.

ex.
```python
from stacked_generalization.lib.stacking import StackedClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn import datasets, metrics
iris = datasets.load_iris()

# Stage 1 model
bclf = LogisticRegression(random_state=1)

# Stage 0 models
clfs = [RandomForestClassifier(n_estimators=40, criterion = 'gini', random_state=1),
        GradientBoostingClassifier(n_estimators=25, random_state=1),
        RidgeClassifier(random_state=1)]

# same interface as scikit-learn
sl = StackedClassifier(bclf, clfs)
sl.fit(iris.target, iris.data)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)
```

More detail example is here.
https://github.com/fukatani/stacked_generalization/blob/master/stacked_generalization/example/cross_validation_for_iris.py

https://github.com/fukatani/stacked_generalization/blob/master/stacked_generalization/example/simple_regression.py

Stacked learning model itself is used as sk-learn model, so you can replace model such as *RandomForestClassifier* to *stacked model* easily in your scripts.

#####2) Evaluation model by out-of-bugs score.
Stacking technic itself uses CV to stage0. So if you use CV for entire stacked model, ***each stage 0 model are fitted n_folds squared times.***
Sometimes its computational cost can be significent,therefore we implemented CV only for stage1[2].

For example, when we get 3 blends (stage0 prediction), 2 blends are used for stage 1 fitting. The remaining one blend is used for model test. Repitation this cycle for all 3 blends, and averaging scores, we can get oob (out-of-bugs) score ***with only n_fold times stage0 fitting.***

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