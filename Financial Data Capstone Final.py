import pandas as pd
import numpy as np
import pandas_datareader as web
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sn
import cvxopt as opt
from cvxopt import blas, solvers
import random

# Import Data

symbols = ['CL=F','XOM','LQD','HYG','CVX','TLT']
symbol_names = ['WTI Crude', 'XOM', 'LQD', 'HYG','CVX','TLT']
start_date = datetime(2020,7,1)
end_date = datetime(2021,7,1)
assets = web.get_data_yahoo(symbols, start_date, end_date)

# Display full data

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Get Close Price

close_price = pd.DataFrame(assets['Adj Close'].fillna(method ='ffill'))
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
plt.savefig('multiple_line_subplot.pdf')
plt.show()

#Normalized Graph

def normalize_data(close_price):
    min_close = close_price.min()
    max_close = close_price.max()
    x = close_price
    
    y = (x - min_close) / (max_close - min_close)
    
    return y

norm_price = normalize_data(close_price)

plt.figure(figsize=(15,15))
plt.plot(assets.index,norm_price)
plt.xlabel('Date')
plt.legend(symbol_names)
plt.title('Normalized Prices')
plt.savefig('Normalized_prices.pdf')
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
plt.savefig('Expected_return.pdf')
plt.show()

sample_std = simple_ror.std()

# Standard Deviation bar chart

plt.figure(figsize=(12,12))
plt.bar(symbol_names,sample_std, color = 'red')
plt.title('Standard Deviation')
plt.savefig('Standard_deviation.pdf')
plt.show()

# Covariance matrix

simple_ror = close_price.pct_change()
covar = simple_ror.cov()
print('Covariance Matrix: \n\n', covar, '\n')

# Correlation matrix

simple_ror = close_price.pct_change()
correl = simple_ror.corr()
print('Correlation Matrix: \n\n', correl)

plt.figure(figsize=(12,12))
sn.heatmap(correl, vmin=-1, vmax=1,cmap='hot', annot = True)
plt.title('Correlation Matrix')
plt.savefig('Correlation_heatmap.pdf')
plt.show()

# Random Portfolios

def return_portfolios(expected_returns, cov_matrix):
    np.random.seed(1)
    port_returns = []
    port_volatility = []
    stock_weights = []
    
    selected = (expected_returns.axes)[0]
    
    num_assets = len(selected) 
    num_portfolios = 5000
    
    for single_portfolio in range(num_portfolios):
        #get stock portfolio weights by dividing random number assigned to each stock with the sum of random numbers
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)
    
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility}
    
    for counter,symbol in enumerate(selected):
        portfolio[symbol +' Weight'] = [Weight[counter] for Weight in stock_weights]
    
    df = pd.DataFrame(portfolio)
    
    column_order = ['Returns', 'Volatility'] + [stock+' Weight' for stock in selected]
    
    df = df[column_order]
   
    return df

# Optimal Portfoilio

def optimal_portfolio(returns):
    n = returns.shape[1]
    returns = np.transpose(returns.values)

    N = 10
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
    
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks



random_portfolios = return_portfolios(expected_return, covar)
#print(random_portfolios)

weights, returns, risks = optimal_portfolio(simple_ror[1:])
#print(weights, returns, risks)

# Efficient Frontier Scatter Plot

random_portfolios.plot.scatter(x='Volatility', y='Returns',figsize =(12,12))

plt.plot(risks,returns , 'g-o')
plt.title("Efficient Frontier")
plt.xlabel("Volatility (Std. Deviation)")
plt.ylabel("Expected Return")
single_asset_std=np.sqrt(np.diagonal(covar))
plt.scatter(single_asset_std,expected_return,marker='X',color='red',s=200)
for i, txt in enumerate(close_price.keys()):
    plt.annotate(txt, (single_asset_std[i], expected_return[i]), size=14, xytext=(10,10), ha='left', textcoords='offset points')
plt.savefig("Efficient_Frontier.pdf")

plt.show()

# Get min/max risk and returns

min_volatility = random_portfolios.Volatility.min()
max_volatility = random_portfolios.Volatility.max()

min_returns = random_portfolios.Returns.min()
max_returns = random_portfolios.Returns.max()

# Min risk

portfolio_min_volatility = [random_portfolios.iloc[[i]] for i in range(5000) \
                            if (random_portfolios.Volatility[i] == min_volatility) \
                            and (random_portfolios.Returns[i] > min_returns)]
print(portfolio_min_volatility)

# Max return

portfolio_max_return = [random_portfolios.iloc[[i]] for i in range(5000) \
                            if (random_portfolios.Volatility[i] > min_volatility) \
                            and (random_portfolios.Returns[i] == max_returns)]

print(portfolio_max_return)
