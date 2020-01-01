from sklearn import tree

# Create classifier
clf = tree.DecisionTreeClassifier(min_samples_split=40)

# Train classifier
clf = clf.fit(features_train, labels_train)

# Predict
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score

# Calculate accuracy (an evaluation metric)
accuracy = accuracy_score(labels_test, pred)
print("My DT accuracy;")
print(accuracy)
