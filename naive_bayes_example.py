from sklearn.naive_bayes import GaussianNB

# Creating classifier
clf = GaussianNB()

# Training data line 9, check time taken line 8, 10
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

# Predicting data line 15, check time taken line 14, 16
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score

# Checking metrics: accuracy 
accuracy = accuracy_score(pred, labels_test)
print(accuracy)
