from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelBinarizer
from stacked_generalization.lib.stacking import StackedClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np
import re

class DataReader(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def disp_hist(self, data, label, bins):
        temp = [i[label].dropna() for i in data]
        plt.hist(temp, histtype='barstacked', bins=bins)
        plt.show()

    def pre_process(self, drop=True, title_to_onehot=True, norm_fare=True):
        def get_title(name):
            title_search = re.search(' ([A-Za-z]+)\.', name)
            if title_search:
                return title_search.group(1)
            return ""

        def normalize_fare(data):
            new_data = None
            for embarked in (0, 1, 2):
                temp = data[data.Embarked == embarked]
                temp['Fare'] /= temp['Fare'].values.mean()
                if new_data is None:
                    new_data = temp
                else:
                    new_data = pd.concat([new_data, temp])
            new_data = new_data.sort('PassengerId')
            return new_data

        data = pd.read_csv(self.file_name).replace('male',0).replace('female',1)
        data['Age'].fillna(data.Age.median(), inplace=True)
        data['Fare'].fillna(data.Fare.median(), inplace=True)
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data['Embarked'] = data['Embarked'].replace('S',0).replace('C',1).replace('Q',2)
        data['Embarked'].fillna(0, inplace=True)
        if norm_fare:
            data = normalize_fare(data)

        # Get all the titles and print how often each one occurs.
        titles = data["Name"].apply(get_title)
        print(pd.value_counts(titles))

        # Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
        title_mapping = {"Dona": 1, "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
        for k,v in title_mapping.items():
            titles[titles == k] = v

        # Add in the title column.
        data['Title'] = titles
        data['Title'].fillna(1, inplace=True)
        #data['Pos'] = data["Title"] + data['Pclass']
        if drop:
            #data = data.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Pclass', 'Cabin', 'Embarked'], axis=1)
            data = data.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
            #data = data.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'Pclass', 'Title'], axis=1)
        print(data.keys())
        if title_to_onehot:
            self.encode(data, 'Title', [i for i in range(1, 11)])
            data = data.drop(['Title'], axis=1)
        return data

    def encode(self, data, label, value_set=None):
        le =LabelBinarizer()
        if value_set is None:
            encoded = le.fit_transform(data[label])
        else:
            le.fit(value_set)
            encoded = le.transform(data[label])
        for i in range(encoded.shape[1]):
            new_label = '{0}_is_{1}'.format(label, i)
            data[new_label] = encoded[:,i]

    def split_by_label(self, data, label='Survived'):
        split_data = []
        for element in set(data[label]):
            split_data.append(data[data[label]==element])
        return split_data

    def get_sample(self, N=600, scale=False):
        all_data = self.pre_process(self.file_name)
        #print('data_type: ' + str(all_data.dtypes))
        all_data = all_data.values
        xs = all_data[:, 2:]
        y = all_data[:, 1]
        if scale:
            xs = preprocessing.scale(xs)
        if N != -1:
            perm = np.random.permutation(xs.shape[0])
            xs = xs[perm]
            y = y[perm]
            xs_train, xs_test = np.split(xs, [N])
            y_train, y_test = np.split(y, [N])
            return xs_train, xs_test, y_train, y_test
        else:
            return xs, y

    def summarize_about_same_ticket(self, data):
        data = data.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Cabin', 'FamilySize'], axis=1)
        for num in data[data.Age <= 5.0]['Ticket']:
            print('num:' + num)
            print(data[data.Ticket == num])


class TestDataReader(DataReader):
    def get_sample(self, N=-1):
        all_data = self.pre_process(self.file_name)
        all_data = all_data.values
        xs = all_data[:, 1:]
        pid = all_data[:, 0]
        return pid, xs

def write_result(pid, output, suffix=''):
    import csv
    import datetime
    suffix += datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    with open("predict_result_data_{0}.csv".format(suffix), "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["PassengerId", "Survived"])
        for pid, survived in zip(pid.astype(int), output.astype(int)):
            writer.writerow([pid, survived])

if __name__ == '__main__':
    import os
    if not os.path.isfile('train.csv'):
        raise Exception('This example is data analysis for Kaggle Titanic Competition.' +
                        'For trying this example, you should download "train.csv" from https://www.kaggle.com/c/titanic.')

    train = True
    full_cv = True
    test = False

    train_dr = DataReader('train.csv')
    bclf = LogisticRegression(random_state=1)
    clfs = [
            RandomForestClassifier(n_estimators=50, criterion = 'gini', random_state=1),
            ExtraTreesClassifier(n_estimators=50, criterion = 'gini', random_state=1),
            ExtraTreesClassifier(n_estimators=50, criterion = 'gini', random_state=2),
            GradientBoostingClassifier(n_estimators=25, random_state=1),
            GradientBoostingClassifier(n_estimators=40, random_state=1),
            Ridge(random_state=1),
            #KNeighborsClassifier(n_neighbors=4)
            #LogisticRegression(random_state=1)
            ]
    sl = StackedClassifier(bclf, clfs, n_folds=3, verbose=2)
    #fsl = FWSLClassifier(bclf, clfs, feature=xs_train[:, 0])
    if train:# evalute by hold-out and out-of-bugs
        sl = StackedClassifier(bclf, clfs, n_folds=3, verbose=2, oob_score_flag=True)
        xs_train, xs_test, y_train, y_test = train_dr.get_sample()
        sl.fit(xs_train, y_train)
        score = sl.score(xs_test, y_test)
        print('score: {0}'.format(score))
        print('oob_score: {0}'.format(sl.oob_score_))
    if full_cv: #cross validation
        sl = StackedClassifier(bclf, clfs, oob_score_flag=False,verbose=2)
        xs_train, y_train = train_dr.get_sample(-1)
        score = []
        for train_index, test_index in StratifiedKFold(3).split(xs_train, y_train):
            sl.fit(xs_train[train_index], y_train[train_index])
            score.append(sl.score(xs_train[test_index], y_train[test_index]))
        print('full-cv score: {0}'.format(score))
    if test: #to make pb leader board data.
        xs_train, y_train = train_dr.get_sample(-1)
        sl.fit(xs_train, y_train)
        test_dr = TestDataReader('test.csv')
        pid, xs_test = test_dr.get_sample(-1)
        output = sl.predict(xs_test)
        write_result(pid, output, sl.tostr())