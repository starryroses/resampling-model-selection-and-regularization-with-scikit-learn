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

# Step 7 - stepwise_path
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

def select_features(X, y, k, direction, cv):
    sfs = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=k,
        direction=direction,
        scoring='neg_mean_squared_error',
        cv=cv,
    )
    sfs.fit(X, y)
    return list(X.columns[sfs.get_support()])

def stepwise_path(X, y, direction, cv):
    p = X.shape[1]
    path = {}
    for k in range(1, p):
        path[k] = select_features(X, y, k, direction, cv)
    path[p] = list(X.columns)   # k = p: nothing to select, use all columns
    return path

# Step 8 - score_path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def score_path(X, y, path, cv):
    sizes, means, ses = [], [], []
    for k in sorted(path):
        scores = -cross_val_score(
            LinearRegression(), X[path[k]], y,
            cv=cv, scoring='neg_mean_squared_error'
        )
        sizes.append(k)
        means.append(round(np.mean(scores), 1))
        ses.append(round(np.std(scores, ddof=1) / np.sqrt(len(scores)), 1))
    return sizes, means, ses

def best_size(sizes, means):
    return sizes[int(np.argmin(means))]

# Step 9 - one_se_rule
import numpy as np

def one_se_rule(values, means, ses, prefer='smaller'):
    means = np.asarray(means, dtype=float)
    ses = np.asarray(ses, dtype=float)
    i_min = int(np.argmin(means))
    threshold = means[i_min] + ses[i_min]
    eligible = [v for v, m in zip(values, means) if m <= threshold]
    return min(eligible) if prefer == 'smaller' else max(eligible)

def choose_subset(X, y, direction, cv):
    path = stepwise_path(X, y, direction, cv)
    sizes, means, ses = score_path(X, y, path, cv)
    size_min = best_size(sizes, means)
    size_1se = one_se_rule(sizes, means, ses, prefer='smaller')
    features_1se = path[size_1se]
    return size_min, size_1se, features_1se

# Step 10 - ridge_path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def ridge_model(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

def ridge_path(X, y, alphas):
    coefs = []
    for a in alphas:
        model = ridge_model(a).fit(X, y)
        coefs.append(model.named_steps['ridge'].coef_)
    return np.array(coefs)

def coef_norms(path):
    return np.round(np.linalg.norm(path, axis=1), 2)

# Step 11 - cv_curve
import numpy as np
from sklearn.model_selection import cross_val_score

def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []
    for value in values:
        model = make_model(value)
        scores = -cross_val_score(model,X,y,cv=cv,scoring='neg_mean_squared_error')        
        means.append(np.mean(scores))
        ses.append(np.std(scores)/ np.sqrt(len(scores)))
    return np.round(means, 1).tolist(), np.round(ses, 1).tolist()

def choose_penalty(make_model, X, y, values, cv):
    means,ses = cv_curve(make_model,X,y,values,cv)
    value_min = values[int(np.argmin(means))]
    value_lse = one_se_rule(values,means,ses,prefer='larger')
    return value_min,value_lse

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

