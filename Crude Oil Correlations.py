import pandas as pd
import numpy as np
import pandas_datareader as web
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sn
import random
from sklearn.linear_model import LinearRegression

# Import Data

symbols = ['CL=F','XLE','DX-Y.NYB','HYG','CADJPY=X','TLT']
symbol_names = ['WTI Crude', 'XLE', 'DXY', 'HYG','CAD/JPY','TLT']
start_date = datetime(2020,7,1)
end_date = datetime(2021,7,1)
assets = web.get_data_yahoo(symbols, start_date, end_date)

# Display full data

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Get Close Price

get_data = pd.DataFrame(assets['Adj Close'].fillna(method ='ffill'))
close_price = get_data.dropna(axis=0)
close_price.columns = symbol_names
first_price = close_price.iloc[1]
change_from_index = close_price.div(first_price).mul(100)
ticker_col = close_price.columns
colors = ['orange','red','blue','gray','green','purple']

print(close_price)
print(close_price.describe())

# 6 graphs of the assets

plt.figure(figsize = (15, 18))
for i in range(len(close_price.columns)):
    plt.subplot(3, 2, i + 1)
    close_price[ticker_col[i]].plot(color = colors[i])
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(symbol_names[i], fontsize = 10, fontweight='bold')
    plt.subplots_adjust(hspace=.8)
plt.savefig('Multiple_line_subplot_oil.pdf')
plt.show()

# Normalized Graph

def normalize_data(close_price):
    min_close = close_price.min()
    max_close = close_price.max()
    x = close_price
    
    y = (x - min_close) / (max_close - min_close)
    
    return y

norm_price = normalize_data(close_price)

plt.figure(figsize=(15,15))
plt.plot(close_price.index,norm_price)
plt.xlabel('Date')
plt.legend(symbol_names)
plt.title('Normalized Prices')
plt.savefig('Normalized_prices_oil.pdf')
plt.show()

# One year returns bar chart

one_yr_returns = ((close_price.iloc[-1].div(first_price)) -1).mul(100)

plt.figure(figsize=(12,12))
plt.bar(symbol_names,one_yr_returns,color=(0.1, 0.1, 0.1, 0.1),  edgecolor='blue')
plt.title('July 2020 - July 2021 Return')
plt.ylabel('Gain (loss)')
plt.savefig('One_yr_returns.pdf')
plt.show()

simple_ror = close_price.pct_change()
expected_return = simple_ror.mean()
expected_return_100=simple_ror.mean().mul(100)

# Expected Return bar chart

plt.figure(figsize=(12,12))
plt.bar(symbol_names,expected_return_100)
plt.title('Expected Return')
plt.savefig('Expected_return_oil.pdf')
plt.show()

sample_std = simple_ror.std()

# Standard Deviation bar chart

plt.figure(figsize=(12,12))
plt.bar(symbol_names,sample_std, color = 'red')
plt.title('Standard Deviation')
plt.savefig('Standard_deviation_oil.pdf')
plt.show()

# Covariance matrix

simple_ror = close_price.pct_change()
covar = simple_ror.cov()
print('Covariance Matrix: \n\n', covar, '\n')

# Correlation matrix of returns

simple_ror = close_price.pct_change()
correl = simple_ror.corr()
print('Correlation Matrix of Returns: \n\n', correl)

plt.figure(figsize=(12,12))
sn.heatmap(correl, vmin=-1, vmax=1,cmap='hot', annot = True)
plt.title('Correlation Matrix of Returns')
plt.savefig('Correlation_heatmap_returns_oil.pdf')
plt.show()

# Correlation matrix of prices

correl = norm_price.corr()
print('Correlation Matrix of Prices: \n\n', correl)

plt.figure(figsize=(12,12))
sn.heatmap(correl, vmin=-1, vmax=1,cmap='hot', annot = True)
plt.title('Correlation Matrix of Prices')
plt.savefig('Correlation_heatmap_prices_oil.pdf')
plt.show()

plt.close('All')

# Linear regression on prices

line_fitter = LinearRegression()

x_xle = close_price['WTI Crude']
y_xle = close_price.XLE

x_xle = x_xle.values.reshape(-1,1)

line_fitter.fit(x_xle,y_xle)
predict_xle = line_fitter.predict(x_xle)

plt.figure(figsize=(12,12))

plt.plot(x_xle,predict_xle)
plt.scatter(x_xle,y_xle)
plt.title('Crude and XLE')
plt.xlabel('Crude')
plt.ylabel('XLE')
plt.savefig('Crude_XLE_regression.pdf')
plt.show()



ax = plt.figure(figsize=(12,20))
ax.suptitle('Regressions using Prices')
for j in range(len(close_price.columns)-1):
    plt.subplot(3, 2, j + 1)
    x = close_price['WTI Crude']
    y = close_price[ticker_col[j+1]]
    x = x.values.reshape(-1,1)
    
    line_fitter.fit(x,y)
    predict_y = line_fitter.predict(x)
    plt.plot(x,predict_y, color = colors[j])
    plt.scatter(x,y, alpha =.4 )

    r2 = line_fitter.score(x, y)
    
    plt.xlabel('WTI Crude')
    plt.ylabel(symbol_names[j+1])
    plt.title('WTI Crude and ' + symbol_names[j+1] + '\n $R^2= %.2f$' % r2 , fontsize = 10, fontweight='bold')
    plt.subplots_adjust(hspace=.8)
plt.savefig('Five_regression_charts_oil.pdf')
plt.show()

# Linear regression on returns

simple_ror = close_price.pct_change()

ax = plt.figure(figsize=(12,20))
ax.suptitle('Regressions using Returns')
for j in range(len(simple_ror.columns)-1):
    plt.subplot(3, 2, j + 1)
    x2 = simple_ror['WTI Crude']
    y2 = simple_ror[ticker_col[j+1]]
    y2 = y2[~np.isnan(y2)]
    x2 = x2[~np.isnan(x2)]
    x2 = x2.values.reshape(-1,1)
    
    line_fitter.fit(x2,y2)
    predict_y2 = line_fitter.predict(x2)
    plt.plot(x2,predict_y2, color = colors[j])
    plt.scatter(x2,y2, alpha =.4 )

    r2 = line_fitter.score(x2, y2)
    
    plt.xlabel('WTI Crude')
    plt.ylabel(symbol_names[j+1])
    plt.title('WTI Crude and ' + symbol_names[j+1] + '\n $R^2= %.2f$' % r2, fontsize = 10, fontweight='bold')
    plt.subplots_adjust(hspace=.8)
plt.savefig('Five_regression_charts_returns_oil.pdf')
plt.show()






































