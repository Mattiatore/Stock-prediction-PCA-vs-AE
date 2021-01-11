from keras.layers import Input, Dense
from keras.models import Model

from get_data_and_define_functions import plot_history, plot_examples, mean_sq_er
from get_data_and_define_functions import x_train_simple, x_test_simple, window_length, test_samples, encoding_dim

# set epochs so that MS losses converges to some point (see plots)
epochs = 200

input_window = Input(shape=(window_length,))
bottleneck = Dense(encoding_dim, activation='relu')(input_window)
decoded = Dense(window_length, activation='sigmoid')(bottleneck)

AE = Model(input_window, decoded)
AE.summary()

AE.compile(optimizer='adam', loss='mean_squared_error')
history = AE.fit(x_train_simple, x_train_simple,
                 epochs=epochs,
                 verbose=1,  # 1 if you want to see progress bar, 2 if not, 0 to suppress output
                 batch_size=1024,
                 shuffle=True,
                 validation_data=(x_test_simple, x_test_simple))

decoded_stocks = AE.predict(x_test_simple)

print("MSE over the validation set...", mean_sq_er(x_test_simple, decoded_stocks, test_samples))

plot_history(history)
plot_examples(x_test_simple, decoded_stocks, 3)
