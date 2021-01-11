from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model

from get_data_and_define_functions import plot_history, plot_examples, mean_sq_er
from get_data_and_define_functions import x_train_deep, x_test_deep, window_length, test_samples, encoding_dim

# set epochs so that MS losses converges to some point (see plots)
epochs = 100
intermediate_red = 2*encoding_dim

input_window = Input(shape=(window_length,))

x = Dense(intermediate_red, activation='relu')(input_window)
x = BatchNormalization()(x)
encoded = Dense(encoding_dim, activation='relu')(x)

x = Dense(intermediate_red, activation='relu')(encoded)
x = BatchNormalization()(x)
decoded = Dense(window_length, activation='sigmoid')(x)

AE = Model(input_window, decoded)

AE.summary()
AE.compile(optimizer='adam', loss='mean_squared_error')

history = AE.fit(x_train_deep, x_train_deep,
                 epochs=epochs,
                 batch_size=1024,
                 shuffle=True,
                 validation_data=(x_test_deep, x_test_deep))

decoded_stocks = AE.predict(x_test_deep)

print("MSE over the validation set...", mean_sq_er(x_test_deep, decoded_stocks, test_samples))

plot_history(history)
plot_examples(x_test_deep, decoded_stocks, 3)
