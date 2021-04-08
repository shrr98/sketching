from environment.drawing_env import DrawingEnvironment
from dqn2.dqn import DQNAgent
from tqdm import tqdm
import numpy as np
import random
import tensorflow as tf
import os

DISCOUNT = 0.99
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
MODEL_NAME = '2x256'
MIN_REWARD = -1000  # For model save

# Environment settings
EPISODES = 20

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = False

#------MAIN--------------
env = DrawingEnvironment('datasets/')

# For stats
ep_rewards = [-1000]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

agent = DQNAgent('model/0405_newest4.h5')

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit = "episode"):
    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1    
    current_state = env.reset()

    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, 242)

        new_state, reward, done = env.step(action)

        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        agent.update_replay_memory((current_state, new_state, reward, action, done))
        agent.train(done, step)
        
        current_state = new_state
        step+=1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
