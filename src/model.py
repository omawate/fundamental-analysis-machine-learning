import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


tickers = ['AMD', 'RDDT']

stock_data = yf.download(tickers, start='2023-04-30', end='2024-04-30')

print(stock_data.head())

daily_returns = stock_data['Adj Close'].pct_change()
daily_returns.plot()
plt.title('Daily Returns')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend(tickers)
plt.show()
