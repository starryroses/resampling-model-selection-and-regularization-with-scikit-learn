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

# Step 4 - cv_mse
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

def cv_mse(estimator, X, y, k=5, random_state=0):
    kf = KFold(n_splits=k,shuffle=True,random_state=random_state)
    scores = cross_val_score(estimator,X,y,cv=kf,scoring='neg_mean_squared_error')
    mse = -scores
    mean = np.mean(mse)
    se = np.std(mse)/np.sqrt(k)
    return round(mean,2), round(se,2)

def loocv_mse(estimator, X, y):
    loo = LeaveOneOut()
    scores = cross_val_score(estimator,X,y,cv=loo,scoring='neg_mean_squared_error')
    mse = -scores
    mean = np.mean(mse)
    return round(mean,2)

# Step 5 - cv_spread_by_k
import numpy as np

def cv_spread_by_k(estimator, X, y, ks, seeds):
    result = {}
    for k in ks:
        over_seeds = []
        for seed in seeds:
            mean, se = cv_mse(estimator,X,y,k,seed)
            over_seeds.append({'mse':mean,'se':se})
        mean_over_seeds = np.mean([x['mse'] for x in over_seeds])
        std_over_seeds = np.std([x['se'] for x in over_seeds])
        result[k] = (round(mean_over_seeds,1),round(std_over_seeds,1))
    return result

def compare_with_loocv(estimator, X, y, ks, seeds):
    return {
        'loocv':loocv_mse(estimator,X,y),
        'kfold':cv_spread_by_k(estimator,X,y,ks,seeds)
    }

# Step 6 - bootstrap_coefficients
import numpy as np
from sklearn.linear_model import LinearRegression

import numpy as np
from sklearn.linear_model import LinearRegression

def bootstrap_coefficients(X, y, n_boot=200, random_state=0):
    rng = np.random.default_rng(random_state)
    n = len(X)
    p = X.shape[1]
    coefs = np.zeros((n_boot, p))
    for i in range(n_boot):
        indices = rng.integers(0, n, n)
        X_boot = X.iloc[indices]
        y_boot = y.iloc[indices]
        model = LinearRegression()
        model.fit(X_boot, y_boot)
        coefs[i] = model.coef_
    return coefs

def bootstrap_se(coefs):
    std_per_coef = np.std(coefs, axis=0, ddof=1)
    return std_per_coef

import numpy as np

def ols_standard_errors(X, y):
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n, p = X.shape

    # Add intercept column
    A = np.column_stack([np.ones(n), X])

    # Fit OLS coefficients
    beta = np.linalg.lstsq(A, y, rcond=None)[0]

    # Calculate residuals
    residuals = y - A @ beta

    # Estimate error variance
    sigma2 = np.sum(residuals**2) / (n - p - 1)

    # Calculate coefficient covariance matrix
    cov = sigma2 * np.linalg.inv(A.T @ A)

    # Extract standard errors, excluding intercept
    se = np.sqrt(np.diag(cov))[1:]

    return np.round(se, 2)

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

