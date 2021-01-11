import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import relativedelta
from sklearn.decomposition import PCA
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model

# SET PARAMETERS
tickers = ['BYD', 'CHH', 'CVEO', 'PLNT', 'RLH', 'HLT', 'H', 'IHG', 'LVS', 'MGM']  # pick stocks
encoding_dim = 5  # choose PCA components and AE bottleneck dimension
intermediate_red_D_AE = 8  # intermediate dimensionality reduction for deep auto-encoder
epochs_FF_AE = 2000  # number of iterations so that loss function converges to some value for feed-forward AE
epochs_D_AE = 2000  # same for deep AE


# DEFINE FUNCTIONS
def plot_examples(stock_input, stock_decoded_pca, stock_decoded_ff,
                  stock_decoded_deep, number_of_windows):
    fig, axs = plt.subplots(2, number_of_windows, sharex=True, figsize=(12, 6))
    # fig.suptitle(f'{decoder_type} method')
    # fig.canvas.set_window_title(f'{decoder_type} method')
    for i in range(number_of_windows):
        axs[0, i].plot(stock_input[i], label='Original')
        axs[0, i].grid()
        axs[0, i].legend()
        axs[1, i].plot(stock_decoded_pca[i], label='PCA')
        axs[1, i].plot(stock_decoded_ff[i], label='Feed-forward AE')
        axs[1, i].plot(stock_decoded_deep[i], label='Deep AE')
        axs[1, i].legend()
        axs[1, i].grid()
        axs[0, i].set_title(f"Example {i + 1}")
    plt.show()


def plot_history(history, AE_type):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    fig.suptitle(f'{AE_type} method')
    ax0.set_title("Loss on training set")
    ax0.plot(history.history["loss"])
    ax0.grid()
    ax1.set_title("Loss on validation set")
    ax1.plot(history.history["val_loss"])
    ax1.grid()
    plt.show()


def plot_dynamics(a, b, c, d, number_of_stocks_to_show, tickers):
    a_r = a.reshape(a.shape[0], a.shape[1])
    fig, axs = plt.subplots(number_of_stocks_to_show, 1, figsize=(10, 10))
    for i in range(number_of_stocks_to_show):
        axs[i].plot(a_r[:, i], label=f'realized dynamics (validation set), {tickers[i]}')
        axs[i].plot(b[:, i], label=f'AE-decoded dynamics, {tickers[i]}')
        axs[i].plot(c[:, i], label=f'AE-decoded dynamics, {tickers[i]}')
        axs[i].plot(d[:, i], label=f'AE-decoded dynamics, {tickers[i]}')
        axs[i].legend()
        axs[i].grid()
    plt.show()


def scatter_plot(original, pca, ff_ae, d_ae, stock_one, stock_two):
    plt.scatter(original[:, stock_one], original[:, stock_two], label='Original dependence')
    plt.scatter(pca[:, stock_one], pca[:, stock_two], label='PCA')
    plt.scatter(ff_ae[:, stock_one], ff_ae[:, stock_two], label='Feed-forward AE')
    plt.scatter(d_ae[:, stock_one], d_ae[:, stock_two], label='Deep AE')
    plt.title('Scatter plots comparison for different methods')
    plt.legend()
    plt.show(figsize=(15, 10))


def prediction_plot(stock_to_predict, tickers):
    zero_signal = x_test_pca.copy()
    zero_signal[:, stock_to_predict] = 0.5

    PCA_recovery = pca.inverse_transform(pca.transform(zero_signal))
    AE_FF_recovery = AE_FF.predict(zero_signal)
    AE_D_recovery = AE_D.predict(zero_signal)

    plt.figure(figsize=(12, 6))
    plt.plot(x_test_pca[:, stock_to_predict], label='Original log-returns of MGM')
    plt.plot(PCA_recovery[:, stock_to_predict], label='PCA recovery')
    plt.plot(AE_FF_recovery[:, stock_to_predict], label='Feed-forward AE recovery')
    plt.plot(AE_D_recovery[:, stock_to_predict], label='Deep AE recovery')
    plt.grid()
    plt.title(f'{tickers[stock_to_predict]} missing data recovery')
    plt.xlabel('Observations')
    plt.ylabel('Scaled log-returns')
    plt.legend()
    plt.show()


# IMPORTING TIME-SERIES OF 10 HOTEL SERVICES STOCKS
dic = {}
for ticker in tickers:
    dic[f'{ticker}'] = yf.download(f"{ticker}")
    dic[f'{ticker}']['log_ret'] = np.log(dic[f'{ticker}'].loc[:, 'Adj Close']).diff()
    dic[f'{ticker}'].dropna(inplace=True)
start_dates = list([dic[f'{ticker}'].index[0]] for ticker in tickers)
end_dates = list([dic[f'{ticker}'].index[-1]] for ticker in tickers)
max_start_date = max(start_dates)[0]
min_end_date = min(end_dates)[0]
end_date_of_training_set = min_end_date - relativedelta(months=2)
start_date_of_test_set = end_date_of_training_set + relativedelta(days=1)

# RESCALING RETURNS TO [0,1] INTERVAL AND PREPARING INPUTS FOR PCA AND AEs
sc = MinMaxScaler()
x_train = np.array([dic[f'{ticker}'].loc[max_start_date:end_date_of_training_set, 'log_ret'] for ticker in tickers])
x_train = np.transpose(x_train)
x_train = np.array([sc.fit_transform(x_train[i].reshape(-1, 1)) for i in range(len(x_train))])
x_test = np.array([dic[f'{ticker}'].loc[start_date_of_test_set:, 'log_ret'] for ticker in tickers])
x_test = np.transpose(x_test)
x_test = np.array([sc.fit_transform(x_test[i].reshape(-1, 1)) for i in range(len(x_test))])

# PCA
x_train_pca = x_train.reshape(x_train.shape[0], x_train.shape[1])
x_test_pca = x_test.reshape(x_test.shape[0], x_test.shape[1])
pca = PCA(encoding_dim).fit(x_train_pca)

print("Principal axes in feature space: ", pca.components_, "\n")
print("The amount of variance explained by each of the selected components.: ", pca.explained_variance_)

z = pca.transform(x_test_pca)
final = pca.inverse_transform(z)

# FEED-FORWARD (SIMPLE) AUTO-ENCODER
input_window = Input(shape=(10,))
bottleneck = Dense(encoding_dim, activation='relu')(input_window)
decoded = Dense(10, activation='sigmoid')(bottleneck)

AE_FF = Model(input_window, decoded)
AE_FF.summary()

AE_FF.compile(optimizer='adam', loss='mean_squared_error')
history = AE_FF.fit(x_train, x_train, epochs=epochs_FF_AE, verbose=1,
                 batch_size=1024, shuffle=True, validation_data=(x_test, x_test))

decoded_stocks = AE_FF.predict(x_test)

# DEEP AUTO-ENCODER
input_window = Input(shape=(10,))

x = Dense(intermediate_red_D_AE, activation='relu')(input_window)
x = BatchNormalization()(x)
encoded = Dense(encoding_dim, activation='relu')(x)

x = Dense(intermediate_red_D_AE, activation='relu')(encoded)
x = BatchNormalization()(x)
decoded = Dense(10, activation='sigmoid')(x)

AE_D = Model(input_window, decoded)

AE_D.summary()
AE_D.compile(optimizer='adam', loss='mean_squared_error')

history = AE_D.fit(x_train, x_train, epochs=epochs_D_AE, batch_size=1024, shuffle=True, validation_data=(x_test, x_test))

decoded_stocks_d = AE_D.predict(x_test)

# PLOT FIGURES
# plot_examples(x_test, final, 3, 'PCA')
# plot_dynamics(x_test, final, 3, tickers, 'PCA')
# plot_history(history, 'Feed-forward auto-encoder')
# plot_examples(x_test, decoded_stocks, 3, 'Feed-forward auto-encoder')
# plot_dynamics(x_test, decoded_stocks, 3, tickers, 'Feed-forward auto-encoder')
# plot_history(history, 'Deep auto-encoder')
# plot_examples(x_test, decoded_stocks_d, 3, 'Deep auto-encoder')
# plot_dynamics(x_test, decoded_stocks_d, 3, tickers, 'Deep auto-encoder')
# scatter_plot(x_test_pca, final, decoded_stocks, decoded_stocks_d, 1, 2)
# PREDICTION
# which_stock_to_predict = 8
# prediction_plot(which_stock_to_predict, tickers)


# Residuals correlation matrix norm comparison
C_true = np.corrcoef(x_test_pca, rowvar=False)
C_pca = np.corrcoef(final, rowvar=False)
C_FF = np.corrcoef(decoded_stocks, rowvar=False)
C_deep = np.corrcoef(decoded_stocks_d, rowvar=False)

# THIS IS THE NORMS OF RESIDUAL CORRELATION MATRICES BETWEEN THE ORIGINAL DATA AND APPROXIMATIONS
# TYPE 'NUC' INSTEAD OF 'FRO' IS YOU WANT TO CHANGE THE NORM TYPE TO THE NUCLEAR ONE
print(np.linalg.norm(C_true - C_pca, 'fro'))
print(np.linalg.norm(C_true - C_FF, 'fro'))
print(np.linalg.norm(C_true - C_deep, 'fro'))
