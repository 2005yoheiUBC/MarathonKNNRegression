"""
Marathon KNN Regression
Author: Yohei Ono
Description:
Predict marathon race times (in hours) based on maximum distance run per week (in miles)
using k-Nearest Neighbors Regression and cross-validation.
"""

import altair as alt
import numpy as np
import pandas as pd
import os
from sklearn import set_config
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Enable Altair large dataset mode
alt.data_transformers.enable("vegafusion")
set_config(transform_output="pandas")

# Load data
marathon = pd.read_csv("data/marathon.csv")

# Sample visualization subset
marathon_50 = marathon.sample(n=50, random_state=300)
marathon_50_plot = (
    alt.Chart(marathon_50)
    .mark_circle()
    .encode(
        x=alt.X("max", title="Maximum distance per week (miles)", scale=alt.Scale(zero=False)),
        y=alt.Y("time_hrs", title="Race time (hours)", scale=alt.Scale(zero=False))
    )
)
marathon_50_plot.save("results/marathon_50_scatter.html")
os.system("open results/marathon_50_scatter.html")

# Predict race time for 100-mile runner using 4-NN
marathon_100mile_time_prediction = (
    marathon_50.assign(diff=(100 - marathon_50["max"]).abs())
    .nsmallest(4, "diff")["time_hrs"]
    .mean()
)
print(f"Predicted race time for 100-mile/week runner: {marathon_100mile_time_prediction:.2f} hours")

# Split data
marathon_training, marathon_testing = train_test_split(marathon, test_size=0.25, random_state=2000)
X_train, y_train = marathon_training[["max"]], marathon_training["time_hrs"]
X_test, y_test = marathon_testing[["max"]], marathon_testing["time_hrs"]

# Create pipeline
pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())

# Cross-validation
cv_results = cross_validate(
    pipe,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error",
    return_train_score=True
)
print("Cross-validation results:", pd.DataFrame(cv_results).mean())

# Grid search for best k
param_grid = {"kneighborsregressor__n_neighbors": range(1, 201)}
tuned = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring="neg_root_mean_squared_error")
tuned.fit(X_train, y_train)

best_k = tuned.best_params_["kneighborsregressor__n_neighbors"]
best_rmspe = -tuned.best_score_
print(f"Best k = {best_k}, Best RMSPE = {best_rmspe:.3f}")

# Evaluate on test data
pred = tuned.predict(X_test)
test_rmspe = mean_squared_error(y_test, pred) ** 0.5
print(f"Test RMSPE = {test_rmspe:.3f}")

# Visualization with predictions
marathon_preds = marathon_training.assign(predictions=tuned.predict(X_train))
marathon_plot = (
    alt.Chart(marathon_preds)
    .mark_circle(opacity=0.4)
    .encode(
        x=alt.X("max", title="Max distance per week (miles)", scale=alt.Scale(zero=False)),
        y=alt.Y("time_hrs", title="Race time (hours)", scale=alt.Scale(zero=False))
    )
    + alt.Chart(marathon_preds)
    .mark_line(color="black")
    .encode(x="max", y="predictions")
)
marathon_plot.save("results/marathon_prediction_fit.html")
os.system("open results/marathon_prediction_fit.html")

print("✅ Marathon regression completed successfully.")
