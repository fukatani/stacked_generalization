from sklearn import datasets
from sklearn.utils.validation import check_random_state
from stacked_generalization.lib.stacking import StackedClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE

iris = datasets.load_iris()
rng = check_random_state(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# Stage 1 model
bclf = LogisticRegression(random_state=1)

# Stage 0 models
clfs = [RandomForestClassifier(n_estimators=40, criterion = 'gini', random_state=1),
        ExtraTreesClassifier(n_estimators=30, criterion = 'gini', random_state=3),
        GradientBoostingClassifier(n_estimators=25, random_state=1),
        GradientBoostingClassifier(n_estimators=30, random_state=2),
        #GradientBoostingClassifier(n_estimators=30, random_state=3),
        KNeighborsClassifier(),
        RidgeClassifier(random_state=1),
        Ridge(),
        TSNE(n_components=2)
        ]

sc = StackedClassifier(bclf,
                       clfs,
                       n_folds=3,
                       verbose=0,
                       stack_by_proba=True,
                       oob_score_flag=True,
                       )

gb = GradientBoostingClassifier(n_estimators=25, random_state=1)

# cross validation
sc_score = 0
gb_score = 0
n_folds = 3
for train_idx, test_idx in StratifiedKFold(n_folds).split(iris.data, iris.target):
    xs_train = iris.data[train_idx]
    y_train = iris.target[train_idx]
    xs_test = iris.data[test_idx]
    y_test = iris.target[test_idx]

    sc.fit(xs_train, y_train)
    print('oob_score: {0}'.format(sc.oob_score_))
    sc_score += sc.score(xs_test, y_test)
    gb.fit(xs_train, y_train)
    gb_score += gb.score(xs_test, y_test)

sc_score /= n_folds
print('Stacked Classfier score: {0}'.format(sc_score))
gb_score /= n_folds
print('Gradient Boosting Classfier score: {0}'.format(gb_score))
