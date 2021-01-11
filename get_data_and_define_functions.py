import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import math
from advcorr import advcorr

# define function

def mean_sq_er(x, y, ts):
    error = np.array([mean_squared_error(x[i], y[i]) for i in tqdm(range(ts))])
    error = np.round(np.mean(error), 4)
    return error


def plot_examples(stock_input, stock_decoded, number_of_windows):
    fig, axs = plt.subplots(2, number_of_windows, sharex=True, figsize=(15, 10))
    for i in range(number_of_windows):
        axs[0, i].plot(stock_input[i])
        axs[0, i].grid()
        axs[1, i].plot(stock_decoded[i])
        axs[1, i].grid()
        axs[0, i].set_title(f"Realized (top) and decoded (bottom) pair, window {i + 1}")
    plt.show()


def plot_history(history):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    ax0.set_title("Loss on training set")
    ax0.plot(history.history["loss"])
    ax0.grid()
    ax1.set_title("Loss on validation set")
    ax1.plot(history.history["val_loss"])
    ax1.grid()
    plt.show()

# set parameters for our task and download data

# choose stock
tickers = ['AAPL', 'GOOG', 'MSFT', 'ADBE', 'AMZN', 'XLU', 'XLE', 'XOM', 'SNP', 'E']
storage = {}
for ticker in tickers:
    df = yf.download(f"{ticker}", start="2015-02-01", end="2020-03-20")
    # input dimension (recommended to be less at least by 3 than test_samples for plot_examples to work)
    window_length = 20
    # reduction dimension
    encoding_dim = 5
    # number of validation samples as a percentage of total data
    test_samples = math.floor(0.02 * df.shape[0])
    
    
    # transform data
    
    df['log_ret'] = np.log(df['Adj Close']).diff()
    df.rename(columns={'Adj Close': 'price'}, inplace=True)
    df = df[['price', 'log_ret']]
    df.dropna(inplace=True)
    
    scale = MinMaxScaler()
    x_train = np.array([scale.fit_transform(df['log_ret'].values[i - window_length:i].reshape(-1, 1)) for i in
                        tqdm(range(window_length + 1, len(df['log_ret'])))])
    
    x_test = x_train[-test_samples:]
    x_train = x_train[:-test_samples]
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    x_train_simple = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test_simple = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    x_test_deep = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    x_train_deep = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    
    
    # visualise
    
    print("Percentage of test data...", np.round((test_samples / len(x_train)) * 100, 2), "%")
    df.plot(y='price', figsize=(15, 10), grid=True)
    plt.show()
    
    storage[ticker]=df

#printing correlations between chosen stocks
for num,i in enumerate(tickers):
    for j in tickers[num+1:]:
        print(i,j,advcorr(storage[i].price, storage[j].price))