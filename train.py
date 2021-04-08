
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

def evaluate_training_result(env, agent, show=False, episodes=10):
    """
    Evaluates the performance of the current DQN agent by using it to play a
    few episodes of the game and then calculates the average reward it gets.
    The higher the average reward is the better the DQN agent performs.
    :param env: the game environment
    :param agent: the DQN agent
    :return: average reward across episodes
    """
    rewards = []
    episodes_to_play = episodes
    for i in range(episodes_to_play):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
            if show:
                env.show()
        rewards.append(episode_reward)
    if show:
        env.close_show()
    average_reward = np.mean(rewards)
    return average_reward


def collect_gameplay_experiences(env, agent, buffer, ontrain=True):
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
    observations = []
    total_reward = 0
    last_rewards = [0,0,0,0,0]
    last_positions = [(-1,-1) for _ in range(5)]
    while not done:
        action = agent.policy(state)
        # israndom = np.mean(last_rewards) <= -5
        # # if the pen is stuck
        # if israndom:
        #     action = env.get_random_action()
        #     # print("random: {} -> {}".format(last_rewards, action))
        #     last_rewards = [0,0,0,0,0]

        next_state, reward, done = env.step(action)
        
        ## Stuck by position
        curr_position = env.drawer.get_pen_position()
        if (reward == -5 and curr_position in last_positions) or \
            (reward<=0 and len([x for x in last_positions if x==curr_position])>=2):
            action = env.get_random_action()
            # last_positions = [(-1,-1) for _ in range(5)]
            state = next_state
            next_state, reward, done = env.step(action)


        last_positions.pop(0)
        last_positions.append(curr_position)

        # last_rewards.pop(0)
        # last_rewards.append(reward)
        total_reward += reward
        observations.append((state, next_state, reward, action, done))
        # if not israndom:
        # buffer.store_gameplay_experience(state, next_state,
        #                                 reward, action, done)
        state = next_state
        # env.show()
        # print("action: {} | reward: {}".format(action, reward))

    print("Train reward : {}".format(total_reward), end = ' | ')
    if ontrain and total_reward <= - 500:
        return
    
    for state, next_state, reward, action, done in observations:
        buffer.store_gameplay_experience(state, next_state,
                                         reward, action, done)
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
    REDUCE_EPSILON_EVERY = 50
    SHOW_EVERY = 50
    SHOW_RENDER = True

    agent = DQNAgent(model_path="model/0405_newest4.h5")
    buffer = ReplayBuffer()
    env = DrawingEnvironment("datasets/")
    loss = []
    avg_rewards = []
    MAX_AVG_REWARD = -2000
    for _ in range(100):
        collect_gameplay_experiences(env, agent, buffer, ontrain=False)
    # return
    agent.EPSILON = 0.0
    for episode_cnt in tqdm(range(1, max_episodes+1), ascii=True, unit = "episode"):
        agent.tensorboard.step = episode_cnt
        collect_gameplay_experiences(env, agent, buffer)
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = loss + agent.train(gameplay_experience_batch)
        avg_reward = evaluate_training_result(env, agent)
        avg_rewards.append(avg_reward)
        print('---Episode {0}/{1} ep_reward={2} '
              'loss={3}'.format(episode_cnt, max_episodes,
                                   avg_rewards[-1], loss[-1]))
        
        # if episode_cnt % AGGREGATE_STATS_EVERY == 0:
        #     if avg_rewards[-1]>-1000:
        #         agent.update_target_network()
        #     else:
        #         agent.rollback_network()
        
        if episode_cnt % REDUCE_EPSILON_EVERY == 0:
            agent.update_epsilon()

        if episode_cnt % AGGREGATE_STATS_EVERY == 0:            
            average_reward = sum(avg_rewards[-AGGREGATE_STATS_EVERY:])/len(avg_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(avg_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(avg_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward)

            agent.update_target_network()

            # if average_reward >= MAX_AVG_REWARD:    
            #     MAX_AVG_REWARD = average_reward
            #     agent.update_target_network()
            # else:
            #     agent.rollback_network()
            # Save model, but only when min reward is greater or equal a set value
            # if min_reward >= -500:
            agent.q_net.save(f'models/{agent.MODEL_NAME}_{int(time.time())}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min.h5')
        
        if SHOW_RENDER and episode_cnt % SHOW_EVERY==0:
            r = evaluate_training_result(env, agent, show=True, episodes=1)
            print("\nShowing... Reward: {}\n".format(r))
    plot(loss, avg_rewards)

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

train_model(max_episodes=1000)