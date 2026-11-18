"""
Resampling, Model Selection and Regularization with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_data
import pandas as pd
from sklearn.datasets import load_diabetes

def load_data():
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target
    return X,y

def describe_data(X, y):
    n,p = X.shape
    features = X.columns.tolist()
    y_mean = round(y.mean(),2)
    return {
        'n':n,
        'p':p,
        'features':features,
        'y_mean':y_mean
    }

# Step 2 - train_test
from sklearn.model_selection import train_test_split

def train_test(X, y, test_size=0.25, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
    return X_train,X_test,y_train,y_test

# Step 3 - validation_set_curve
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def poly_model(degree):
    pipeline = make_pipeline(PolynomialFeatures(degree),LinearRegression())
    return pipeline

def validation_set_curve(X, y, feature, degrees, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X[[feature]],y,test_size=0.5,random_state=random_state)
    res = {}
    for degree in degrees:
        model = poly_model(degree)
        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(preds,y_test)
        res[degree] = round(mse,1)
    return res

def curve_spread(X, y, feature, degrees, seeds):
    results = {}
    for degree in degrees:
        error = []
        for seed in seeds:
            curve = validation_set_curve(X,y,feature,[degree],seed)
            error.append(curve[degree])
        results[degree] = round(max(error)-min(error),1)
    return results

# Step 4 - cv_mse (not yet solved)
# TODO: implement

# Step 5 - cv_spread_by_k (not yet solved)
# TODO: implement

# Step 6 - bootstrap_coefficients (not yet solved)
# TODO: implement

# Step 7 - stepwise_path (not yet solved)
# TODO: implement

# Step 8 - score_path (not yet solved)
# TODO: implement

# Step 9 - one_se_rule (not yet solved)
# TODO: implement

# Step 10 - ridge_path (not yet solved)
# TODO: implement

# Step 11 - cv_curve (not yet solved)
# TODO: implement

# Step 12 - lasso_path (not yet solved)
# TODO: implement

# Step 13 - pcr_model (not yet solved)
# TODO: implement

# Step 14 - pls_model (not yet solved)
# TODO: implement

# Step 15 - fit_all (not yet solved)
# TODO: implement

# Step 16 - test_report (not yet solved)
# TODO: implement

