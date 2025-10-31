# %%
import altair as alt
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Simplify working with large datasets in Altair
alt.data_transformers.enable('vegafusion')

# Output dataframes instead of arrays
set_config(transform_output="pandas")

# import data
# %%
marathon = pd.read_csv("data/marathon.csv")
marathon
# %%
marathon_50 = marathon.sample(n= 50, random_state = 300)
marathon_50_plot = alt.Chart(marathon_50).mark_circle().encode(
    x = alt.X("max").title("maximum distance ran per week (in miles) ").scale(zero = False),
    y = alt.Y("time_hrs").title("Race time").scale(zero = False)
)
marathon_50_plot
# %%
# We want to predict the race time for someone who 
# ran a maximum distance of 100 miles per week during training.
# With KNN Regression(k = 4)
marathon_100mile_time_prediction = (
    marathon_50.assign(diff = (100 - marathon_50["max"]).abs())
    .nsmallest(4,"diff")["time_hrs"]
    .mean()
)
marathon_100mile_time_prediction
# %%
# Find the best k for estimation
# Cross Validation

# split the data
marathon_training, marathon_testing = train_test_split(
    marathon,
    test_size=0.25,
    random_state = 2000
)
# Predictors and Target
X_train = marathon_training[["max"]]  #dataframe [[]]
y_train = marathon_training["time_hrs"]   #A series []
X_test = marathon_testing[["max"]]
y_test = marathon_testing["time_hrs"]

# Preprocessor(Scaler)
preprocessor = StandardScaler()
# Pipeline
marathon_pipe = make_pipeline(preprocessor,KNeighborsRegressor())

# Cross Validation Results
marathon_cv = pd.DataFrame(
    cross_validate(
        marathon_pipe,
        X_train,
        y_train,
        cv = 5,
        scoring="neg_root_mean_squared_error",
        return_train_score = True
    )
)
marathon_cv

#Find the best k(GridSearchCV)
np.random.seed(2019)
param_grid = {"kneighborsregressor__n_neighbors" :range(1,201,1)}
marathon_tuned = GridSearchCV(
    marathon_pipe, param_grid, cv = 5, n_jobs= -1, 
    scoring="neg_root_mean_squared_error"
)
marathon_results = pd.DataFrame(
    marathon_tuned.fit(X_train, y_train).cv_results_
)
marathon_results
# %%
marathon_min = marathon_tuned.best_params_
marathon_best_RMSPE = -marathon_tuned.best_score_

print(marathon_min, "\n ",marathon_best_RMSPE)
# %%

# applying to Testing Data
np.random.seed(1234)
marathon_prediction = marathon_tuned.predict(X_test)
#get the error
marathon_rmspe = mean_squared_error(y_test, marathon_prediction)**0.5
marathon_rmspe
# %%

#The error of train set is 0.5669298834669159
#The error of test set is 0.6116092762646357
#The model has worked pretty good

#Make a plot with trending line
np.random.seed(2019)
marathon_preds = marathon_training.assign(
    predictions = marathon_tuned.predict(X_train)
)
marathon_plot = (
    alt.Chart(marathon_preds).mark_circle(opacity = 0.4).encode(
        x = alt.X("max").title("the maximum distance run per week").scale(zero =False),
        y = alt.Y("time_hrs").title("marathon time").scale(zero = False)
    )
    +
    alt.Chart(marathon_preds).mark_line(color="black").encode(
        x = "max",
        y = "predictions"
    )
)
marathon_plot
# %%
