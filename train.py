
"""
Training loop
This module trains the DQN agent by trial and error. In this module the DQN
agent will play the game episode by episode, store the gameplay experiences
and then use the saved gameplay experiences to train the underlying model.
"""
from environment.drawing_env import DrawingEnvironment
from dqn.agent import DQNAgent
from dqn.replaybuffer import ReplayBuffer
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

def evaluate_training_result(env, agent):
    """
    Evaluates the performance of the current DQN agent by using it to play a
    few episodes of the game and then calculates the average reward it gets.
    The higher the average reward is the better the DQN agent performs.
    :param env: the game environment
    :param agent: the DQN agent
    :return: average reward across episodes
    """
    total_reward = 0.0
    episodes_to_play = 10
    for i in range(episodes_to_play):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    average_reward = total_reward / episodes_to_play
    return average_reward


def collect_gameplay_experiences(env, agent, buffer):
    """
    Collects gameplay experiences by playing env with the instructions
    produced by agent and stores the gameplay experiences in buffer.
    :param env: the game environment
    :param agent: the DQN agent
    :param buffer: the replay buffer
    :return: None
    """
    state = env.reset()
    done = False
    while not done:
        action = agent.collect_policy(state)
        next_state, reward, done = env.step(action)
        # if done:
        #     reward = 500
        buffer.store_gameplay_experience(state, next_state,
                                         reward, action, done)
        state = next_state

def plot(losses, rewards):

    eps = range(len(losses))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(eps, rewards, label='Avg Rewards')
    plt.title('Average Rewards')

    plt.subplot(1, 2, 2)
    plt.plot(eps, losses, label='Loss')
    plt.title('Loss')
    plt.show()

def train_model(max_episodes=50000):
    """
    Trains a DQN agent to play the CartPole game by trial and error
    :return: None
    """
    AGGREGATE_STATS_EVERY = 20
    agent = DQNAgent(model_path="models/dqn_3000_from2000_datasets_1617518990___182.56max_-424.49avg_-715.88min.h5")
    buffer = ReplayBuffer()
    env = DrawingEnvironment("datasets/")
    loss = []
    avg_reward = []
    for _ in range(100):
        collect_gameplay_experiences(env, agent, buffer)
    for episode_cnt in tqdm(range(1, max_episodes+1), ascii=True, unit = "episode"):
        collect_gameplay_experiences(env, agent, buffer)
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = loss + agent.train(gameplay_experience_batch)
        avg_reward.append(evaluate_training_result(env, agent))
        print('---Episode {0}/{1} ep_reward={2} '
              'loss={3}'.format(episode_cnt, max_episodes,
                                   avg_reward[-1], loss[-1]))
        
        if episode_cnt % 5 == 0:
            if avg_reward[-1]>-700:
                agent.update_target_network()
            else:
                agent.rollback_network()

        if episode_cnt % AGGREGATE_STATS_EVERY == 0:
            # agent.update_epsilon()
            
            average_reward = sum(avg_reward[-AGGREGATE_STATS_EVERY:])/len(avg_reward[-AGGREGATE_STATS_EVERY:])
            min_reward = min(avg_reward[-AGGREGATE_STATS_EVERY:])
            max_reward = max(avg_reward[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= -1000:
                agent.q_net.save(f'models/{agent.MODEL_NAME}_{int(time.time())}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min.h5')

    plot(loss, avg_reward)

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

train_model(max_episodes=1000)