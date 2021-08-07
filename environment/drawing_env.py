import utils
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
    GOAL_SIMILARITY_THRESH = .05

    PENALTY_STEP = -1

    def __init__(self, datadir='datasets/'):
        self.reference = Reference() # Reference Image to be drawn
        self.drawer = Drawer()
        self.reference_paths = self._get_all_references(datadir)
        self.index=-1
        self.reset()

    def set_reference(self, path):
        self.reference_paths = [path]

    def reset(self):
        """
        Reset the RL Environment
        """
        # self.index = np.random.randint(0, len(self.reference_paths))
        self.index += 1
        if self.index == len(self.reference_paths): 
            self.index = 0
            # np.random.shuffle(self.reference_paths)
        self.reference.load_canvas(self.reference_paths[self.index]) # placeholder
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

        if self.episode_step >= self.MAX_STEP or similarity <= self.GOAL_SIMILARITY_THRESH:
            done = True
            # if similarity/(84*84) <= self.GOAL_SIMILARITY_THRESH:
            #     reward = 100

        return new_observation, reward, done
        

    def calculate_reward(self, current_similarity, step_taken):
        """
        """
        reward_pixel =  self.last_similarity - current_similarity
        reward = reward_pixel
        
        if step_taken["x"]==0 and step_taken["y"]==0 and reward<=0:
            reward = -10
        elif step_taken["pen_state"]==0 or ((abs(step_taken["x"]) < 5 and abs(step_taken["y"]) < 5)):
            reward += self.PENALTY_STEP 
        # if step_taken["pen_state"]==0 or abs(reward_pixel)<1:
        #     reward += self.PENALTY_STEP 
        # elif ((abs(step_taken["x"]) < 5 and abs(step_taken["y"]) < 5)):
        #     reward += (5 - max(step_taken["x"], step_taken["y"]))/10 * self.PENALTY_STEP
        return reward
            
    def show(self):
        """
        Shows the current state.

        :return: None
        """
        if self.SHOW_RENDER:
            canvas = np.array(self.drawer.get_canvas()*255, dtype=np.uint8)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

            ref = np.array(self.reference.get_canvas()*255, dtype=np.uint8)
            ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
            
            pen_pos = self.drawer.get_pen_position()

            canvas = cv2.circle(canvas, pen_pos, 1, (0,0,255), 1)
            ref = cv2.circle(ref, pen_pos, 1, (0,0,255), 1)

            distance_map = np.array(self.drawer.get_distance_map()*255, dtype=np.uint8)
            distance_map = cv2.cvtColor(distance_map, cv2.COLOR_GRAY2BGR)
            
            color_map = np.array(self.drawer.get_color_map()*255, dtype=np.uint8)
            color_map = cv2.cvtColor(color_map, cv2.COLOR_GRAY2BGR)

            images = np.vstack(
                        (np.hstack((ref, canvas)),
                        np.hstack((distance_map, color_map)))
            )

            images = cv2.resize(images, (3*84*2, 3*84*2))
            
            cv2.imshow('Current State', images)
            cv2.waitKey(1)
            return images

    def close_show(self):
        cv2.destroyAllWindows()

    def closest_node(self, node, nodes):
        nodes = np.asarray(nodes)
        deltas = abs(nodes - node)
        manhattan_dist = np.max(deltas, axis=1)
        # dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argmin(manhattan_dist)

    def get_actions_to_nearest_stroke(self):
        diff = self.reference.get_canvas() - self.drawer.get_canvas()
        ys, xs = np.where(diff>0.9)

        if ys.shape[0] == 0:
            actions = [self.get_random_action(pen_state=0)]
            return actions

        points = np.array(
            [p for p in zip(xs, ys)], dtype=np.int8
        )

        p = self.drawer.get_pen_position()
        pos = np.array(self.drawer.get_pen_position(), dtype=np.int8)

        idx = self.closest_node(pos, points)
        
        target_point = points[idx]
        x_jump, y_jump = target_point - pos

        # if(del_x > 5): del_x = 5
        # elif(del_x < -5): del_x = -5

        # if(del_y > 5): del_y = 5
        # elif(del_y < -5): del_y = -5
        pen_step = 5
        x_direction = 1 if x_jump >=0 else -1
        y_direction = 1 if y_jump >= 0 else -1
        
        if x_jump==0 and y_jump==0: # if not jumping
            return [self.get_random_action()]

        x_steps = [x_direction*pen_step for _ in range(x_direction*pen_step, x_jump+x_direction, x_direction*pen_step)]
        if x_jump % pen_step :
            if x_direction==1:
                x_steps.append(x_jump%pen_step)
            else:
                x_steps.append(x_jump%pen_step - pen_step)
        
        y_steps = [y_direction*pen_step for _ in range(y_direction*pen_step, y_jump+y_direction, y_direction*pen_step)]
        if y_jump % pen_step :
            if y_direction==1:
                y_steps.append(y_jump%pen_step)
            else:
                y_steps.append(y_jump%pen_step - pen_step)

        moves = []
        
        if len(x_steps) >= len(y_steps):
            moves = [(x,y) for x, y in zip(x_steps[:len(y_steps)], y_steps)]
            for x in x_steps[len(y_steps):] :
                moves.append((x,0))
        elif len(x_steps) < len(y_steps):
            moves = [(x,y) for x, y in zip(x_steps, y_steps[:len(y_steps)])]
            for y in y_steps[len(x_steps):]:
                moves.append((0,y))

        actions = [position_to_action(m, 0) for m in moves]
    
        # print(target_point, pos, [x_jump, y_jump], moves, actions)

        # diff_rgb = np.zeros((84,84,3))

        # for i in range(ys.shape[0]):
        #     cv2.circle(diff_rgb, (xs[i], ys[i]), 0, (255,0,0))

        # diff = np.hstack((self.reference.get_canvas(), self.drawer.get_canvas(), diff))
        # cv2.imshow("a", diff)
        # cv2.imshow("ab", diff_rgb)
        # cv2.waitKey(500)

        return actions


    def get_random_action(self, pen_state=None):
        pen_position = self.drawer.get_pen_position()
        pen_state = np.random.randint(0, 2) if pen_state==None else pen_state
        x_left = min(5, pen_position[0])
        x_right = max(83-5, pen_position[0])

        y_top = min(5, pen_position[1])
        y_bot = max(83-5, pen_position[1])

        x = np.random.randint(-x_left, 84-x_right)
        y = np.random.randint(-y_top, 84-y_bot)

        return position_to_action((x,y), pen_state, 11)

    def get_ref_image(self):
        return self.reference_paths[self.index]

    def get_ref_number(self):
        return len(self.reference_paths)
