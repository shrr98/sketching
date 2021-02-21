import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Concatenate, Input

class DoodleSDQModel(Model):
    def __init__(self, global_input_size, local_input_size, output_size):
        super(DoodleSDQModel, self).__init__()
        self.global_input_size = global_input_size
        self.local_input_size = local_input_size
        self.output_size = output_size

        self.model_global = self._create_global_CNN()
        self.model_local = self._create_local_CNN()
        self.dense = self._create_dense()

    def _create_global_CNN(self):
        model = Sequential()
        model.add(Input(shape=self.global_input_size))
        model.add(Conv2D(32, (8,8), stride=(4,4), activation="relu"))
        model.add(Conv2D(64, (4,4), stride=(2,2), activation="relu"))
        model.add(Conv2D(64, (3,3), stride=(1,1), activation="relu"))
        return model

    def _create_local_CNN(self):
        model = Sequential()
        model.add(Input(shape=self.local_input_size))
        model.add(Conv2D(128, (11,11), activation="relu"))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def _create_dense(self):
        model = Sequential()
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.output_size, activation="softmax"))
        return model

    def call(self, inputs):
