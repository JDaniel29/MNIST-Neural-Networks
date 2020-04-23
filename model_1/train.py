import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
import matplotlib.pyplot as plt

# Read in our test and training datasets
train_data_raw = np.loadtxt("mnist_train.csv", skiprows=1, delimiter=',')
num_images = train_data_raw.shape[0]

X_train = train_data_raw[:, 1:].reshape(num_images, 28, 28, 1)
Y_train = train_data_raw[:, 0]

# Create our Model
model = Sequential()
model.add(Conv2D(12, kernel_size = (3, 3), activation = "relu", input_shape=(28, 28, 1)))
model.add(Conv2D(24, kernel_size = (3, 3), activation = "relu"))
model.add(Conv2D(48, kernel_size = (3, 3), activation = "relu"))
model.add(Conv2D(96, kernel_size = (3, 3), activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size = 100, epochs = 4, validation_split = 0.2)

model.save('saved_models/model')
printf("Saved Model!")
