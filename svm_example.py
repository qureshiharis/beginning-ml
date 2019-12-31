from sklearn import svm

# Creating classifier with linear decision boundary
clf = svm.SVC(kernel='linear')

# Training classifier
clf.fit(features_train, labels_train)

# Predicting
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score

# Calculating evaluation metric; accuracy
print(accuracy_score(labels_test, pred))
