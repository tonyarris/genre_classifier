import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import json
import random

# CNN - https://www.youtube.com/watch?v=dOG-HxpbMSw


DATASET_PATH = "./data/data.json"


def load_data(dataset_path):
    with open(dataset_path, 'r')as fp:
        data = json.load(fp)

    # convert list into numpy arrays
    X = np.array(data['mfcc'])
    y = np.array(data['labels'])

    return X, y


def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data(DATASET_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # TensorFlow expects a 3d array for each sample, need an extra dimension of 1
    X_train = X_train[..., np.newaxis]  # 4d array -> {num samples, 130,  13, 1}
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def predict(model, X, y):
    # model.predict expects a 4d array, so we need to add dimension 0
    X = X[np.newaxis, ...]

    # prediction = [ [0, 1, 2, ..., 9] ]
    prediction = model.predict(X)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)  # 1d array indicating the index
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))


def build_model(input_shape):
    # create model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(
        keras.layers.BatchNormalization())  # normalises the activations in the layer that get presented to the next layer, speeding up training

    # 2nd conv layer
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(
        keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(64, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(
        keras.layers.BatchNormalization())

    # flatten output and feed to dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output softmax layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def saveModel(model):
    # save the model to disk
    model.save('classifier_model')


if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25,
                                                                                    0.2)  # training and validation size

    # build the CNN
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # compile the network
    optimizer = keras.optimizers.Nadam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    # train the CNN
    model.fit(X_train, y_train,
              validation_data=(X_validation, y_validation),
              batch_size=32,
              epochs=50)

    # evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # make a prediction on a sample
    rnd = random.randint(0,100)
    X = X_test[rnd]
    y = y_test[rnd]
    predict(model, X, y)

    # save model
    saveModel(model)