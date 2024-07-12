"""
Code to get latest datastreams.

Each line indicates the best way to download data streams.
Some are from the Federal Reserve website (FRED).
Others are from DBNomics.
Historical ISM data is copied manually, but new values are updated from DBNomics.

Daily data, where relevant, such as TNX (10-yr treasury rates), are obtained from yfinance.

Need the following modules: fradapi and dbnomics (pip install)

Once we have the data, the goal will be the plug them into our model.
"""

import pandas as pd
import yfinance as yf
from fredapi import Fred
import local
from dbnomics import fetch_series, fetch_series_by_api_link

fred = Fred(api_key= local.fred_api)



#Ten year treasury interest rates.
tnx = yf.download('TNX')

#Stock volatility index
vix = yf.download('VIX')

#ISM purchasing managers index

ism = pd.read_csv('../database/ISM.csv', index_col = 0)
ism.index = pd.to_datetime(ism.index)

#Updating scheme for ISM data.
add_ism = fetch_series('ISM/pmi/pm')
new_vals = add_ism[~(add_ism.period.isin(ism.index))]
new_ism = pd.DataFrame({'Actual':new_vals.value.values}, index = new_vals.period)
ism = pd.concat([new_ism, ism], axis = 0).sort_index()
ism.to_csv('../database/ISM.csv')


#US unemployment data - U3 monthly
unemployment_rate = fred.get_series('UNRATE')


#US weekly initial jobless claims

unemployment_rate = fred.get_series('ICSA')

#Inflation rate CPI, year-on-year

cpi = fred.get_series('CPIAUCSL').pct_change(12)

#University of Michigan consumer sentiment
mcsi = fred.get_series('UMCSENT')
