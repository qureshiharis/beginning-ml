from sklearn import svm

# Creating classifier with linear decision boundary
clf = svm.SVC(kernel='linear')

# Training dataset down to 1% of its original size, tossing out 99% of the training data
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]


# Training classifier
clf.fit(features_train, labels_train)

# Predicting
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score

# Calculating evaluation metric; accuracy
print(accuracy_score(labels_test, pred))
