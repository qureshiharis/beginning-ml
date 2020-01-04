from sklearn.cluster import KMeans

# Start with two clusters and fit data
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Predict
kmeans.predict(...)
