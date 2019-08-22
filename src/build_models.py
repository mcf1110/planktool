import sys
import os
sys.path.append('./libs')

from joblib import dump

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import dataset as d

def build_models():
    classifiers = {
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'naive_bayes': GaussianNB(),
        '1nn': KNeighborsClassifier(1),
        '3nn': KNeighborsClassifier(1),
        '5nn': KNeighborsClassifier(1),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'svm': SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
            max_iter=-1, probability=False, random_state=42, shrinking=True,
            tol=0.001, verbose=False)
    }
    general = d.remove_extras(d.general(d.read('./dataset.csv')))
    Xg = general[general.columns[:-1]]
    yg = general[general.columns[-1]]

    specific = d.remove_extras(d.specific(d.read('./dataset.csv')))
    Xs = specific[specific.columns[:-1]]
    ys = specific[specific.columns[-1]]

    for clf in classifiers:
        pipelined = make_pipeline(StandardScaler(), classifiers[clf])
        pipelined.fit(Xg, yg)
        get_path = lambda p: os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/%s/%s.joblib' % (p, clf))
        dump(pipelined, get_path('general'))
        pipelined.fit(Xs, ys)
        dump(pipelined, get_path('specific'))