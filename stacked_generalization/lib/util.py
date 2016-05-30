from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.externals import joblib
import pandas as pd

def multiple_feature_weight(self, blend, feature):
    assert blend.shape[0] == feature.shape[0]
    result = None
    for a_vec in blend.T:
        for b_vec in feature.T:
            if result is None:
                result = a_vec * b_vec
            else:
                result = np.c_[result, a_vec * b_vec]
    return result

def numpy_c_concatenate(A, B):
    if A is None:
        return B
    else:
        return np.c_[A, B]

def saving_predict_proba(model, X, index):
    csv_file = "{0}_{1}_{2}.csv".format(model.id, min(index), max(index))
    try:
        df = pd.read_csv(csv_file)
        proba = df.values[:, 1:]
        print("**** prediction is loaded from {0} ****".format(csv_file))
    except IOError:
        proba = model.predict_proba(X)
        df = pd.DataFrame({'index': index})
        for i in range(proba.shape[1]):
            df["prediction" + str(i)] = proba[:, i]
        #print(df)
        df.to_csv(csv_file, index=False)
    return proba

def get_model_id(model):
    model_type = str(type(model))
    return model_type[model_type.rfind(".")+1: model_type.rfind("'")]

##def saving_fit(learner, X, y, index):
##    import os
##    pkl_file = "{0}_{1}_{2}.pkl".format(learner.id, min(index), max(index))
##    try:
##        learner = joblib.load(pkl_file)
##        print("**** learner is loaded from {0} ****".format(pkl_file))
##    except IOError:
##        learner.fit(X, y)
##        joblib.dump(learner, pkl_file)
##    return learner

if __name__ == '__main__':
    temp = {'index': [0, 1], 'value': [2, 3]}
    df = pd.DataFrame(temp)
    print(df)
    df.to_csv('dum.csv', index=False)