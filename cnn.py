# %%
import pandas as pd
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

# %%
# import data and preprocessing
(train_x, train_y), (test_x, test_y_classes) = mnist.load_data()

# change the dimension of x to [number of records, length, width, channels]
# change the datatype of x to float32
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1).astype('float32')

# normalization
train_x /= 255
test_x /= 255

# y to one-hot encoding to a 10-D vector
train_y = np_utils.to_categorical(train_y, 10)
test_y = np_utils.to_categorical(test_y_classes, 10)

# %%
# start building model
model = Sequential()

# convolution layer 1
# filter * 16, kernel size = 5 * 5, zero-padding * 1, input image = 28 * 28 * 1, activation function = ReLU
# outputs 28 * 28 image * 16
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same',
          input_shape=(28, 28, 1), activation='relu'))

# pooling layer 1
# max pooling with pool size = 2 * 2, stride = 2
# outputs 14 * 14 image * 16
model.add(MaxPooling2D(pool_size=(2, 2)))

# convolution layer 2
# filter * 36, kernel size = 5 * 5, zero-padding * 1, activation function = ReLU
# outputs 14 * 14 image * 36
model.add(Conv2D(filters=36, kernel_size=(5, 5),
          padding='same', activation='relu'))

# pooling layer 2
# max pooling with pool size = 2 * 2
# outputs 7 * 7 image * 36
model.add(MaxPooling2D(pool_size=(2, 2)))

# dropout layer
# set 25% of weights to 0 to avoid overfitting
model.add(Dropout(0.25))

# full-connected layers

# flatten layer
# 1764 neurons
# outputs vectors of dimension = 7 * 7 * 36 = 1764
model.add(Flatten())

# hidden layer
# 128 neurons, activation function = ReLU
model.add(Dense(128, activation='relu'))

# dropout layer
# set 50% of weights to 0 to avoid overfitting
model.add(Dropout(0.5))

# output layer
# 10 neurons, activation function = softmax
model.add(Dense(10, activation='softmax'))

# print model structure
print(model.summary())

# %%
# complie model
# loss function: categorical cross-entropy (for training)
# metrics: accuracy (as pomparision index)
# optimizer: Adam
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# train model and get training result
train_history = model.fit(train_x, train_y, epochs=20,
                          batch_size=100, verbose=2)

# plot training loss curve
loss = train_history.history["loss"]
plt.plot(loss)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# %%
# perdiction
result = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', result[0])
print('Test accuracy:', result[1])

predict_y = model.predict(test_x)
predict_y_classes = np.argmax(predict_y, axis=1)

# get confusion matrix
pd.crosstab(test_y_classes, predict_y_classes, rownames=[
            'True Class'], colnames=['Perdicted Class'])
