from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.externals import joblib
import pandas as pd

def multiple_feature_weight(blend, X):
    result = None
    for a_vec in blend.T:
        for b_vec in X.T:
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

def saving_predict_proba(model, X, index, cache_dir=''):
    try:
        csv_file = get_cache_file(model.id, index, cache_dir)
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

def saving_predict(model, X, index, cache_dir=''):
    csv_file = get_cache_file(model.id, index,cache_dir)
    try:
        df = pd.read_csv(csv_file)
        prediction = df.values[:, 1:]
        prediction = prediction.reshape([prediction.size,])
        print("**** prediction is loaded from {0} ****".format(csv_file))
    except IOError:
        prediction = model.predict(X)
        df = pd.DataFrame({'index': index})
        prediction.reshape([prediction.shape[-1],])
        df["prediction"] = prediction
        #print(df)
        df.to_csv(csv_file, index=False)
    return prediction

def get_model_id(model):
    model_type = str(type(model))
    model_type = model_type[model_type.rfind(".")+1: model_type.rfind("'")]
    param_dict = model.get_params()
    ignore_list = ('n_jobs', 'oob_score', 'verbose', 'warm_start')
    new_param_dict = {}
    for key, value in sorted(param_dict.items(), key=lambda x: x[0]):
        i = 0
        if key in ignore_list:
            continue
        while True:
            new_key = key[0] + str(i)
            if not new_key in new_param_dict:
                new_param_dict[new_key] = value
                break
            i += 1
    model_type += str(new_param_dict)
    replace_dict = {'{': '_',
                    '}': '',
                    "'": "",
                    '.': 'p',
                    ',': '__',
                    ':': '_',
                    ' ': '',
                    'True': '1',
                    'False': '0',
                    'None': 'N',
                    '=': '_',
                    '(': '_',
                    ')': '_',
                    '\n': '_'}
    for key, value in replace_dict.items():
        model_type = model_type.replace(key, value)
    if len(model_type) > 150:
        model_type = model_type[:150]
    return model_type

def get_cache_file(model_id, index, cache_dir='', suffix='csv'):
    # Identify index trick.
    # If sum of first 20 index, recognize as the same index.
    if index is None:
        raise IOError
    if len(index) < 20:
        sum_index = sum(index)
    else:
        sum_index = sum(index[:20])
    return "{0}{1}_{2}.{3}".format(cache_dir,
                                   model_id,
                                   sum_index,
                                   suffix)

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