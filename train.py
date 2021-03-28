
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
        if done:
            reward = -1.0
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
    agent = DQNAgent(model_path="model/2403-minstrokes4-maxstrokes64-jumping-0_01-whiteonblack-242-6-maxpool.h5")
    buffer = ReplayBuffer()
    env = DrawingEnvironment("examples/")
    loss = []
    avg_reward = []
    for _ in range(100):
        collect_gameplay_experiences(env, agent, buffer)
    for episode_cnt in range(max_episodes):
        print("---Episode {}".format(episode_cnt))
        collect_gameplay_experiences(env, agent, buffer)
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = loss + agent.train(gameplay_experience_batch)
        avg_reward.append(evaluate_training_result(env, agent))
        print('Episode {0}/{1} and so far the performance is {2} and '
              'loss is {3}'.format(episode_cnt, max_episodes,
                                   avg_reward[-1], loss[-1]))
        if episode_cnt % 20 == 0:
            agent.update_target_network()
    print('No bug lol!!!')
    plot(loss, avg_reward)


train_model(max_episodes=40)