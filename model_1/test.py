import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
import matplotlib.pyplot as plt

test_data_raw = np.loadtxt("../mnist_test.csv", skiprows=1, delimiter=',')
num_images = test_data_raw.shape[0]

X_test = test_data_raw[:, 1:].reshape(num_images, 28, 28, 1)
Y_test = test_data_raw[:, 0]

model = keras.models.load_model("../saved_models/model_1")
model.summary()

predictions = model.predict(X_test)
misfires = []

for image in range(len(X_test)):
    actual_label = Y_test[image]
    predicted_label = np.argmax(predictions[image])

    if actual_label != predicted_label:
        misfires.append(image)


print("Number of Misfires: ", len(misfires))
print("Overall Accuracy: ", model.evaluate(X_test, Y_test, verbose=0)[1])

for misfire in misfires:
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.title("Actual: " + str(int(Y_test[misfire])) + "\nPrediction: " + str(np.argmax(predictions[misfire])))
    plt.imshow(X_test[misfire, :, :, 0])
    plt.subplot(1, 2, 2)
    plt.title("Confidence: " + str(predictions[misfire][np.argmax(predictions[misfire])]))
    plt.xticks(range(10))
    plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], predictions[misfire])
    plt.savefig("misfires/" + str(int(Y_test[misfire])) + "-" + str(np.argmax(predictions[misfire])) + " - " + str(misfire) + ".png")
    plt.close()

print("MISFIRES HAVE BEEN SAVED!")
