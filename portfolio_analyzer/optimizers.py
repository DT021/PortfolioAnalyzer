import numpy as np
import pandas as pd


def minimal_variance(data):
    """Optimize portfolio in order to minimize variance."""
    returns_data = data.pct_change().dropna()
    sigma = returns_data.cov().values
    A = 0.5 * sigma
    A = np.hstack((A, -np.ones((sigma.shape[0], 1))))
    A = np.vstack((A, np.ones((1, sigma.shape[0] + 1))))
    A[-1, -1] = 0.0
    B = np.ones((1, A.shape[0]))[0]
    w = np.dot(np.linalg.inv(A), B)
    return pd.DataFrame([w[:-1]], columns=data.columns)


def approximated_max_kelly(data):
    """Find a approximated solution of the portofolio based on kelly criterion."""
    returns_data = data.pct_change().dropna()
    mu = np.mean(returns_data.values, axis=0)
    sigma = returns_data.cov().values
    A = 0.5 * sigma
    w = np.dot(np.linalg.inv(A), mu)
    w /= w.sum()
    return pd.DataFrame([w], columns=data.columns)
