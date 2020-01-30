import numpy as np


class NaiveBackTest:
    """Performs naive backtest of the portfolio."""

    def __init__(self, tickers_ratio, data):
        self.tickers_ratio = tickers_ratio
        self.tickers = tickers_ratio.keys()
        self.weights = np.array(tickers_ratio.values())
        self.data = data

    def run(self, capital=100.0):
        """Execute the back test."""
        log_data = np.log(
            np.sum(
                (1 + self.data[self.tickers].pct_change()).dropna() * self.weights,
                axis=1,
            )
        )
        historical_series = np.cumsum(log_data)
        portfolio = (np.exp(historical_series) * capital).to_frame()
        portfolio.columns = ["portfolio"]
        return portfolio
