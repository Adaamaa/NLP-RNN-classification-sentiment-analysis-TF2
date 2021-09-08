
"""
NLP project

Build and train a RNN classifier for sentiment analysis on IMDB dataset.

You are given the output layer with 1 neuron and sigmoid activation function,
create the rest of the network.
"""


import urllib
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_model():
	dataset = tf.keras.datasets.imdb


	vocab_size = 1000
	embedding_dim = 100
	max_length = 120
	trunc_type = "post"

	(X_train, y_train), (X_test, y_test) = dataset.load_data(num_words=vocab_size, maxlen=max_length, )
	X_train = pad_sequences(X_train, maxlen=max_length, padding=trunc_type)
	X_test = pad_sequences(X_test, maxlen=max_length, padding=trunc_type)

	model = tf.keras.models.Sequential([
		tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
		tf.keras.layers.LSTM(units=128),
		tf.keras.layers.Dense(1, activation="sigmoid")
	])

	model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
	model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))

	return model


# save your model in the .h5 format.
# Google expect that to test your solution.
if __name__ == "__main__":
	model = solution_model()
	model.save("nlp.h5")
