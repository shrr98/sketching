from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from collections import deque
import numpy as np
import random
import time
import tensorflow as tf
from tqdm import tqdm
import os
from PIL import Image
import cv2
from environment.drawing_env import DrawingEnvironment

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


######################################################################
##                         MOD TENSORBOARD                          ##
######################################################################
# Modified Tensorboard
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._log_write_dir = "logs/"
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()




######################################################################
##                              DQN AGENT                           ##
######################################################################
class DQNAgent:
    def __init__(self):
        # main model : this is what we train every step
        self.model = self.create_model()

        # target model : this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # how we keep the agent consistent taking step with the model over time -> prediction consistency
        # this will save REPLAY_MEMORY_SIZE steps
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")\

        self.target_update_counter = 0


    def create_model(self):
        model_pretrained = tf.keras.models.load_model(model_path)
        # model = tf.keras.models.load_model(model_path)

        x = model_pretrained.layers[-2].output

        outputs = Activation("linear", name="QValue")(x)
        model = Model(inputs=model_pretrained.inputs, outputs=outputs)
        
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        
        return model


    def update_replay_memory(self, transition):
        #transition is observation space, reward, action space (?)
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0] # * : unpacking shape

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # after taking action
        new_current_states = np.array([transition[3] for transition in minibatch])/255

        future_qs_list = self.target_model.predict(new_current_states)

        X = [] # images taken from the game
        y = [] # action that we decide to take

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch): # this is why new current_states is on index 3
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, \
            shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        
        #updating to determine if we want to update targer_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


