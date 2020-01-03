from sklearn.linear_model import LinearRegression

# Create classifier and fit data
reg = LinearRegression().fit(X, y)

# Calculate score metric
reg.score(X, y)

# Calculate slope of the line
reg.coef_

# Calculate intercept of the line
reg.intercept_

# Predict
reg.predict(features_test)
