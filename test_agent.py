from dqn.agent import DQNAgent

if __name__ == "__main__":
    agent = DQNAgent("model/2403-minstrokes4-maxstrokes64-jumping-0_01-whiteonblack-242-6-maxpool.h5")
    agent.q_net.summary()