import pandas as pd
from yahoofinancials import YahooFinancials


def yahoo2pandas(tickers, from_date='2000-01-01', to_date='2100-12-31', frequency='weekly'):
    results = {}
    yahoo_financials = YahooFinancials(tickers)
    historical_stock_prices = yahoo_financials.get_historical_price_data(from_date, to_date, frequency)

    for ticker in historical_stock_prices:
        try:
            df = pd.DataFrame(historical_stock_prices[ticker]['prices'])
            df['formatted_date'] = (df['formatted_date']).astype('datetime64[ns]')
            df = df.set_index('formatted_date')
            results[ticker] = df['close'].loc[~df['close'].index.duplicated(keep='first')]
        except:
            continue
    return pd.DataFrame.from_dict(results).dropna()