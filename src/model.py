import talib as ta
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import mean_squared_error,r2_score, accuracy_score, confusion_matrix, precision_score, classification_report


"""
Set parameters.
"""

#Stock ticker
tickers = 'SPY'#['AMD', 'RDDT']

#Lookback period for information.
period = 60

#Forward period for prediction.
front_period = 60

#For multi-speed indicators, need a ratio of the period.
aleph = 0.4

#Set up training and validation and testing fractions.
train_frac = 0.7
val_frac = 0.9



stock_data = yf.download(tickers)

print(stock_data.head())

price = stock_data['Adj Close']
daily_returns = stock_data['Adj Close'].pct_change()
daily_returns.plot()
plt.title('Daily Returns')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend(tickers)
plt.show()

#Get technical indicators: make sure they are stationary

#Simple moving average
sma = stock_data['Adj Close'].rolling(window = period).mean()
dsma = sma.pct_change()

#Standard deviation
std = stock_data['Adj Close'].rolling(window = period).std()

#Normalized price
normalized_price = (stock_data['Adj Close'] - sma)/std

#Volume
vol = (stock_data.Volume+1).rolling(window=period).mean().pct_change()
vol_std = vol.rolling(window = period).std()

return_std = stock_data['Adj Close'].pct_change().rolling(window = period).std()

#Standard indiciators
rsi = ta.RSI(sma, timeperiod=int(period*aleph))
macd, macdsignal, macdhist = ta.MACD(sma, fastperiod=int(period*aleph), slowperiod=int(2*period*aleph), signalperiod=int(period*aleph*2/3))
adx = ta.ADX(stock_data.High, stock_data.Low, sma, timeperiod=int(period*aleph))
willr = ta.WILLR(sma,sma,sma, timeperiod=int(period*aleph))

#Develop signal.
future_returns = (stock_data['Adj Close'].pct_change(front_period) > 0).shift(-front_period).dropna()

#Combine everything
df = pd.DataFrame({'signal': future_returns,
				   'normalized_price': normalized_price, 
				   'vol_std': vol_std, 
				   'return_std':return_std, 
				   'rsi': rsi, 
				   'macd': macd, 
				   'macdsignal': macdsignal, 
				   'macdhist': macdhist, 
				   'adx': adx, 
				   'willr': willr, 
				   'dsma': dsma})

#First, reindex so that we have a row for each future return.
#Then forward fill NA values, so that any missing values on daily basis are filled with last known value.
#Finally, drop NA values.
df = df.reindex(future_returns.index).ffill().dropna()

#Set up training and validation. We do not go for testing until we are happy with the model.
y = df['signal']
X = df.drop(columns = 'signal')


train_size = int(train_frac*len(X))
val_size = int(val_frac*len(X))
X_train = X[:train_size]
y_train = y[:train_size]


X_val = X[train_size: val_size]
y_val = y[train_size: val_size]


#Near function to show change in signal
def plotSignalChange(pred, price):
	price = price.reindex(pred.index)
	pred_change = (pred != pred.shift(1))
	pred_buy = pred_change*(pred == True)
	pred_sell = pred_change*(pred == False)

	plt.plot(price.index , price)
	plt.scatter(price.index[pred_buy], price[pred_buy], marker = '^', color = 'green')
	plt.scatter(price.index[pred_sell], price[pred_sell], marker = 'v', color = 'red')
	plt.show()


#Run Model
model = RandomForestClassifier(n_estimators = 800, max_depth = 8)
model.fit(X_train, y_train)

pred_train = pd.Series(model.predict(X_train), index = y_train.index)
print('Training\n')
print(classification_report(y_train, pred_train))

plotSignalChange(pred_train, price)


pred_val = pd.Series(model.predict(X_val), index = y_val.index)
print('Validation\n')
print(classification_report(y_val, pred_val))

plotSignalChange(pred_val, price)



