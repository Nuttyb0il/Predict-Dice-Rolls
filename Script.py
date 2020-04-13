from random import randint
from numpy import array
from numpy import argmax
from pandas import concat
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

def generate_sequence(length=25):
    return [randint(1, 6) for _ in range(length)]

def one_hot_encode(sequence, n_unique=100):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]

def generate_data():
    sequence = generate_sequence()
    encoded = one_hot_encode(sequence)
    df = DataFrame(encoded)
    df = concat([df.shift(4), df.shift(3), df.shift(2), df.shift(1), df], axis=1)
    values = df.values
    values = values[5:,:]
    X = values.reshape(len(values), 5, 100)
    y = encoded[4:-1,:]
    return X, y

# create model
model = Sequential()
model.add(LSTM(50, batch_input_shape=(5, 5, 100), stateful=True))
model.add(Dense(100, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# for i in range(2000):
#     X, y = generate_data()
#     model.fit(X, y, epochs=1, batch_size=5, verbose=2, shuffle=False)
#     model.reset_states()
# model.save_weights("model_predictor")
# evaluate model on new data
#UN-COMMENT IF YOU WANT TO RESET THE MODEL

model.load_weights("model_predictor")
X, y = generate_data()
yhat = model.predict(X, batch_size=5)

print('Dice rolls : {0}'.format(one_hot_decode(y)))
print('Predicted dices: {0}'.format(one_hot_decode(yhat)))