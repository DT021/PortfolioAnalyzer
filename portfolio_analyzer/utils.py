def portfolio2dic(data):
    """Convert the result of portfolio optimization (pandas data frame) in a dictionary ready to feed a backtest."""
    return data.loc[0].to_dict()
