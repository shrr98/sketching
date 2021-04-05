import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.layers import Dense, Activation

from tensorflow.python.keras.optimizers import Adam
import time
from dqn.modifiedtb import ModifiedTensorBoard

class DQNAgent:
    """
    DQN Agent : the agent that explores the game and
    should eventually learn how to play the game.
    """

    def __init__(self, model_path=None):
        self.ACTION_SPACE_SIZE = 242
        self.MIN_EPSILON = 0.02
        self.EPSILON = 0.5
        self.MODEL_NAME = "dqn_4000_from3000_datasets"
        self.q_net = self._load_pretrained_dqn_model(model_path, action_space=self.ACTION_SPACE_SIZE)
        self.target_q_net = self._load_pretrained_dqn_model(model_path, action_space=self.ACTION_SPACE_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.MODEL_NAME}-{int(time.time())}", profile_batch=0)

    @staticmethod
    def _build_dqn_model():
        """
        Builds a deep neutal net which predict the Q values for all possible
        actions given a state. The input should have the shape of the state, and
        the output should have the same shape as the action space since we want
        1 Q value per possible action.
        
        :return: Q Network
        """
        q_net = Sequential()
        q_net.add(Dense(64, input_dim=4, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        q_net.add(Dense(2, activation='linear', kernel_initializer='he_uniform'))

        q_net.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=["accuracy"])

        return q_net

    @staticmethod
    def _load_pretrained_dqn_model(model_path, action_space):
        model_pretrained = tf.keras.models.load_model(model_path)
        # model = tf.keras.models.load_model(model_path)
        # for layer in model_pretrained.layers[:-5]:
        #     layer.trainable = False

        x = model_pretrained.layers[-2].output

        outputs = Activation("linear", name="QValue")(x)
        model = Model(inputs=model_pretrained.inputs, outputs=outputs)
        
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
        
        return model

    def random_policy(self, state):
        """
        Outputs a random action

        :param state: not used
        :return: action
        """
        return np.random.randint(0,self.ACTION_SPACE_SIZE)

    def collect_policy(self, state):
        """
        Similar to policy but with some randomness to encourage exploration.

        :param state: the game state
        :return: action
        """
        if np.random.random() < self.EPSILON: # epsilon
            return self.random_policy(state)
        return self.policy(state)


    def policy(self, state):
        """
        Takes a state from the game environment and returns
        an action that has the highest Q value and should be taken
        as the next step.

        :param state: the current game environment state
        :return: an action
        """
        # state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.q_net(state)
        action = np.argmax(action_q.numpy()[0], axis=0)
        return action

    def update_target_network(self):
        """
        Updates the current target_q_net with the q_net which brings all the
        training in the q_net to the target_q_net.
        
        :return: None
        """
        self.target_q_net.set_weights(self.q_net.get_weights())

    def rollback_network(self):
        """
        Updates the current target_q_net with the q_net which brings all the
        training in the q_net to the target_q_net.
        
        :return: None
        """
        self.q_net.set_weights(self.target_q_net.get_weights())

    def update_epsilon(self):
        self.EPSILON *= 0.75

    
    def train(self, batch):
        """
        Trains the underlying network with a batch of gameplay experiences to
        help it better predict the Q values.

        :param batch: a batch of gameplay experiences
        :return: training loss
        """
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = batch
        current_q = self.q_net(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch).numpy()
        max_next_q = np.amax(next_q, axis=1)

        for i in range(action_batch.shape[0]):
            target_q_val = reward_batch[i]
            if not done_batch[i]:
                target_q_val += 0.997 * max_next_q[i] # discount
            target_q[i][action_batch[i]] = target_q_val

        training_history = self.q_net.fit(x=state_batch, y= target_q, verbose=0, callbacks=[self.tensorboard])
        loss = training_history.history['loss']

        return loss

