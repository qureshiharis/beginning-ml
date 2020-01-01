from sklearn.neighbours import KNeighborsClassifier

# Create classifier with 3 nearest neighbor
clf = KNeighborsClassifier(n_neighbors=3)

# Train classifier
clf = clf.fit(features_train, labels_train)

# Predict
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score

# Accuracy
accuracy = accuracy_score(labels_test, pred)
print(accuracy)
