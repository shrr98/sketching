import numpy as np
from drawing_env import DrawingEnvironment

class RandomStrokeGenerator:
    def __init__(self):
        print("init")

    def step(self, action):
        return [], 0, False

if __name__=="__main__":
    rsg = RandomStrokeGenerator()
    total_reward = 0
    for i in range(50):
        action = np.random.randint(0, 242)
        new_observation, reward, done = rsg.step(action=action)
        total_reward += reward
        # print(f"Step {i+1} {reward} \t {total_reward}")
        images = rsg.show()
         