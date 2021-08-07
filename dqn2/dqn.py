from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import time
import tensorflow as tf
from tqdm import tqdm
import os
import cv2

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 20_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 512  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'

ACTION_SPACE_SIZE=242


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
    def __init__(self, model_path):
        # main model : this is what we train every step
        self.model = self._load_pretrained_dqn_model(model_path, action_space=ACTION_SPACE_SIZE)

        # target model : this is what we .predict against every step
        self.target_model = self._load_pretrained_dqn_model(model_path, action_space=ACTION_SPACE_SIZE)

        # how we keep the agent consistent taking step with the model over time -> prediction consistency
        # this will save REPLAY_MEMORY_SIZE steps
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")\

        self.target_update_counter = 0


    # def create_model(self):
    #     model = Sequential()
    #     model.add(Conv2D(256, (3,3), input_shape=env.OBSERVATION_SPACE_VALUES))
    #     model.add(Activation("relu"))
    #     model.add(MaxPooling2D(2,2))
    #     model.add(Dropout(0.2))

    #     model.add(Conv2D(256, (3,3)))
    #     model.add(Activation("relu"))
    #     model.add(MaxPooling2D(2,2))
    #     model.add(Dropout(0.2))

    #     model.add(Flatten())
    #     model.add(Dense(64))

    #     model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))        
    #     model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

    #     return model

    
    @staticmethod
    def _load_pretrained_dqn_model(model_path, action_space):
        model_pretrained = tf.keras.models.load_model(model_path)
        # model = tf.keras.models.load_model(model_path)
        # for layer in model_pretrained.layers[:-5]:
        #     layer.trainable = False

        x = model_pretrained.layers[-2].output

        outputs = Activation("linear", name="QValue")(x)
        model = Model(inputs=model_pretrained.inputs, outputs=outputs)
        
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        
        return model

    def update_replay_memory(self, transition):
        #transition is observation space, reward, action space (?)
        self.replay_memory.append(transition)

    def sample_gameplay_batch(self):
        """
        Samples a batch of gameplay experiences for training purposes.

        :return: a list of gameplay experiences
        """
        batch_size = min(MINIBATCH_SIZE, len(self.replay_memory))
        sampled_gameplay_batch = random.sample(self.replay_memory, batch_size)

        state_global_patches = []
        state_local_patches = []
        next_state_global_patches = []
        next_state_local_patches = []
        action_batch = []
        reward_batch = []
        done_batch = []

        for gameplay_experience in sampled_gameplay_batch:
            state_global_patches.append(gameplay_experience[0][0])
            state_local_patches.append(gameplay_experience[0][1])
            next_state_global_patches.append(gameplay_experience[1][0])
            next_state_local_patches.append(gameplay_experience[1][1])
            reward_batch.append(gameplay_experience[2])
            action_batch.append(gameplay_experience[3])
            done_batch.append(gameplay_experience[4])
        
        # STATE BATCH
        global_patches = np.stack(state_global_patches, axis=0).squeeze()
        local_patches = np.stack(state_local_patches, axis=0).squeeze()
        state_batch = [global_patches, local_patches]


        # NEXT STATE BATCH
        global_patches = np.stack(next_state_global_patches, axis=0).squeeze()
        local_patches = np.stack(next_state_local_patches, axis=0).squeeze()
        next_state_batch = [global_patches, local_patches]

        return (state_batch, next_state_batch, np.array(action_batch),
                np.array(reward_batch), np.array(done_batch))

    def get_qs(self, state):
        return self.model.predict(state)[0] # * : unpacking shape

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        state_batch, next_state_batch, action_batch, reward_batch, done_batch  = self.sample_gameplay_batch()

        current_qs_list = self.model.predict(state_batch)

        future_qs_list = self.target_model.predict(next_state_batch)

        max_future_qs = np.amax(future_qs_list, axis=1)

        for i in range(action_batch.shape[0]):
            target_q_val = reward_batch[i] / 100.0
            if not done_batch[i]:
                target_q_val += 0.997 * max_future_qs[i] # discount
            current_qs_list[i][action_batch[i]] = target_q_val

        self.model.fit(x=state_batch, y= current_qs_list, batch_size=MINIBATCH_SIZE, verbose=0, \
            shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        
        #updating to determine if we want to update targer_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


