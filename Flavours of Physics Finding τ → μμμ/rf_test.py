import numpy as np
import pandas as pd

train = pd.read_csv("./training.csv")
test  = pd.read_csv("./check_agreement.csv")

features = list(train.columns[1:-5])

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard'
)
voting_clf.fit(train[features], train["signal"])

from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(train[features], train["signal"])
    y_pred = clf.predict(test[features])
    print(clf.__class__.__name__, accuracy_score(test['signal'], y_pred))



