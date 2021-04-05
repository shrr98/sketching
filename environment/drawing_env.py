import cv2
import numpy as np
from utils.utils import pixel_similarity, crop_image, position_to_action
import math
import os
from environment.drawer import Drawer
from environment.reference import Reference 

class DrawingEnvironment:

    SHOW_RENDER = True
    MAX_STEP    = 200
    GOAL_SIMILARITY_THRESH = .01

    PENALTY_STEP = -5 # coba coba

    def __init__(self, datadir='datasets/'):
        self.reference = Reference() # Reference Image to be drawn
        self.drawer = Drawer()
        self.reference_paths = self._get_all_references(datadir)
        self.reset()

    def reset(self):
        """
        Reset the RL Environment
        """
        index = np.random.randint(0, len(self.reference_paths))
        self.reference.load_canvas(self.reference_paths[index]) # placeholder
        self.episode_step = 0
        self.drawer.reset()

        self.last_similarity = pixel_similarity(self.reference.get_canvas(), self.drawer.get_canvas())

        observation = self.get_observation()
        return observation

    def _get_all_references(self, datadir):
        paths = []
        for f in os.listdir(datadir):
            paths.append(os.path.join(datadir, f))
        return paths

    def get_observation(self):
        observation_global = np.expand_dims(np.stack(
            (
                self.drawer.get_canvas(), self.reference.get_canvas(),
                self.drawer.get_distance_map(), self.drawer.get_color_map()
            ), 
            axis=2
        ), axis=0)

        observation_local = np.expand_dims(np.stack(
            (
                self.drawer.get_patch(),
                self.reference.get_patch(self.drawer.get_pen_position())
            ),
            axis=2
        ), axis=0)

        return [observation_global, observation_local]


    def step(self, action):
        """
        Executes a step by doing action.

        :param action: (int) action to be taken
        :return: (tuple) new observation, reward, game state
        """
        self.episode_step += 1
        action_taken = self.drawer.do_action(action)
        done = False

        new_observation = self.get_observation()

        similarity = pixel_similarity(self.reference.get_canvas(), self.drawer.get_canvas())
        # print("similarity: {} {}".format(similarity, similarity/255**2))

        reward = self.calculate_reward(similarity, action_taken)

        self.last_similarity = similarity

        if self.episode_step >= self.MAX_STEP or similarity/(255*255) <= self.GOAL_SIMILARITY_THRESH:
            done = True
            if similarity/(255*255) <= self.GOAL_SIMILARITY_THRESH:
                reward = 100

        return new_observation, reward, done
        

    def calculate_reward(self, current_similarity, step_taken):
        """
        """
        reward_pixel =  self.last_similarity - current_similarity
        reward = reward_pixel
        
        if step_taken["pen_state"]==0 or ((abs(step_taken["x"]) < 5 and abs(step_taken["y"]) < 5)) or reward < 5:
            reward += self.PENALTY_STEP 

        return reward
            
    def show(self):
        """
        Shows the current state.

        :return: None
        """
        if self.SHOW_RENDER:
            images = np.vstack(
                        (np.hstack((self.reference.get_canvas(), self.drawer.get_canvas())),
                        np.hstack((self.drawer.get_distance_map(), self.drawer.get_color_map())))
            )
            cv2.imshow('Current State', images)
            cv2.waitKey(1)
            return images
