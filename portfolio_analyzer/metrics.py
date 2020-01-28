import numpy as np
import pandas as pd


class MainMetrics:
    """Compute the main metrics for asset."""

    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.benchmark.columns = ["benchmark"]

    def estimate(self, data):
        """Perform the estimation of the metrics for every asset in data."""
        results = {}
        for ticker in data.columns:
            results[ticker] = self.__metrics(data[[ticker]])
        return pd.DataFrame(results)

    def __metrics(self, data):
        main_metrics = {}
        main_metrics["benchmark correlation"] = self.__market_corr(data)
        main_metrics["alpha"] = self.__alpha(data)
        main_metrics["sharpe ratio"] = self.__sharpe_ratio(data)
        return main_metrics

    def __market_corr(self, data):
        return (
            pd.concat([data.pct_change(), self.benchmark.pct_change()], axis=1)
            .corr()
            .values[0, 1]
        )

    def __alpha(self, data):
        data_frequency = (data.index[1] - data.index[0]) / pd.offsets.Day(1)
        year_events = 365 / data_frequency
        average_return = np.exp(
            np.mean(np.log((1 + data.pct_change()).dropna()))
        ).values[0]
        return average_return ** year_events - 1.0

    def __sharpe_ratio(self, data):
        return_data = data.pct_change().dropna()
        mu = np.mean(return_data).values[0]
        std = np.std(return_data).values[0]
        return mu / std
