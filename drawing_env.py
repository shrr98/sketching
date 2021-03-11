import cv2
import numpy as np
from utils import pixel_similarity, crop_image, position_to_action
import math

class Drawer:
    """
    Class implementasi Drawer untuk DoodleSQN.

    Image (canvas, distance_map, dan color_map) dinormalisasi
    sehingga nilai per piksel antara 0-1.
    """

    CANVAS_SIZE     = 84
    PATCH_SIZE      = 11
    CANVAS_IMG_SIZE = (CANVAS_SIZE, CANVAS_SIZE)
    PATCH_IMG_SIZE  = (PATCH_SIZE, PATCH_SIZE)

    PEN_STATE_SIZE  = 2
    PATCH_SPACE_SIZE = PATCH_SIZE**2
    ACTION_SPACE_SIZE = PEN_STATE_SIZE * PATCH_SIZE * PATCH_SIZE


    def __init__(self):
        self.canvas = None          # canvas untuk menggambar
        self.distance_map = None    # distance map untuk posisi pena
        self.color_map = None       # colormap untuk keadaan pena 
        self.pen_position = (0,0)   # posisi pena pada kanvas
        self.pen_state = 0          # kondisi pena (0: up, 1: down)

    def set_canvas(self, canvas):
        self.canvas = np.copy(canvas)

    def set_pen_position(self, pen_position):
        self.pen_position = pen_position
        

    def do_action(self, action):
        """
        Executes action the Agent decided.

        :param action: (int) action to be executed
        :return: Coordinate in patch the pen to be placed
        """
        index = action % self.PATCH_SPACE_SIZE

        # expected x and y target by moving to index
        x = index % self.PATCH_SIZE -5
        y = index // self.PATCH_SIZE -5

        # the real x and y target (due to canvas boundary) 
        target = (
                min(self.CANVAS_SIZE-1, max(0, (self.pen_position[0]) + x)),
                min(self.CANVAS_SIZE-1, max(0, (self.pen_position[1]) + y))
            )
        x_target = target[0] - self.pen_position[0] 
        y_target = target[1] - self.pen_position[1] 

        self.pen_state = action // self.PATCH_SPACE_SIZE
        
        # determine the executed action
        action_real = position_to_action((x_target, y_target), self.pen_state, self.PATCH_SIZE)
        
        if not self.pen_state: # 1. Pena down
            self.draw_stroke(target)
            
        else: # 2. Pena up
            self.move_pen(target)

        return {
            "pen_state" : self.pen_state, 
            "x" : x,
            "y" : y,
            "x_real" : x_target,
            "y_real" : y_target,
            'action_real' : action_real  
        }
        
        
    def draw_stroke(self, target):
        cv2.line(self.canvas, self.pen_position, target, 1, 1)
        self.move_pen(target)


    def move_pen(self, target):
        self.pen_position = target

    def reset(self):
        """
        Reset the environment.
        1. Clear the canvas
        2. Initialize pen position randomly
        3. Set pen state to 0 (up)
        4. Calculate distance map
        5. Calculate color map
        """
        self.canvas = np.full(self.CANVAS_IMG_SIZE, 0, dtype=np.float)
        
        pen_position_init = (np.random.randint(0,self.CANVAS_SIZE), np.random.randint(0,self.CANVAS_SIZE))
        self.move_pen(pen_position_init)
        self.pen_state = 0

        self.color_map = np.full(self.CANVAS_IMG_SIZE, 1, dtype=np.float)


    def func_l2_distance(self, x, y):
        """
        Calculate L2 distance map.

        :param x: coordinate x of the pixel
        :param y: coordinate y of the pixel

        :return: float
        """
        return np.sqrt((x-self.pen_position[1])**2 + (y-self.pen_position[0])**2) / self.CANVAS_SIZE

    def get_canvas(self):
        """
        Get current canvas.

        :return: Mat
        """
        return np.copy(self.canvas)

    def get_patch(self, pen_position=None):
        """
        Get the current patch image
        
        :param pen_position: absolute pen position as the center of the patch.
        :return: Mat
        """

        if not pen_position:
            pen_position = self.pen_position
        # calculate bounding box
        top     = pen_position[1] - self.PATCH_SIZE//2
        bottom  = pen_position[1] + self.PATCH_SIZE//2 + 1
        left    = pen_position[0] - self.PATCH_SIZE//2
        right   = pen_position[0] + self.PATCH_SIZE//2 + 1 

        crop = crop_image(self.canvas, (top, left, bottom, right))

        return np.copy(crop)

    def get_pen_position(self):
        """
        Get the current pen position.

        :return: list
        """
        return self.pen_position


    def get_distance_map(self):
        """
        Get current distance map.

        :return: Mat
        """
        self.distance_map = np.fromfunction(self.func_l2_distance, self.CANVAS_IMG_SIZE)
        # self.distance_map[self.distance_map > 1] = 1
        return np.copy(self.distance_map)

    def get_color_map(self):
        """
        Get current color map.

        :return: Mat
        """
        self.color_map[:] = 0 if self.pen_state else 1
        return np.copy(self.color_map)

class DrawingEnvironment:

    SHOW_RENDER = True
    MAX_STEP    = 100
    GOAL_SIMILARITY_THRESH = .05

    PENALTY_STEP = -1 # coba coba

    def __init__(self):
        self.reference = None # Reference Image to be drawn
        self.drawer = Drawer()
        self.reset()

    def reset(self):
        """
        Reset the RL Environment
        """
        self.reference = np.zeros(self.drawer.CANVAS_IMG_SIZE, dtype=np.uint8) # placeholder
        self.episode_step = 0
        self.last_similarity = 0

        self.drawer.reset()


    def step(self, action):
        """
        Executes a step by doing action.

        :param action: (int) action to be taken
        :return: (tuple) new observation, reward, game state
        """
        self.episode_step += 1
        step_taken = self.drawer.do_action(action)

        new_observation = (
            self.reference, self.drawer.get_canvas(),
            self.drawer.get_distance_map(), self.drawer.get_color_map(),
            self.drawer.get_patch()
        )

        similarity = pixel_similarity(self.reference, self.drawer.get_canvas())
        print(similarity)

        reward = self.calculate_reward(similarity, step_taken)

        if self.episode_step >= self.MAX_STEP or similarity/(255**2) <= self.GOAL_SIMILARITY_THRESH:
            done = True

        return np.copy(new_observation), reward, done
        

    def calculate_reward(self, current_similarity, step_taken):
        """
        """
        reward_pixel = current_similarity - self.last_similarity
        reward = reward_pixel
        
        if step_taken["pen_state"] and (abs(step_taken["x"]) < 5 and abs(step_taken["y"]) < 5):
            reward += self.PENALTY_STEP 

        return reward
            
    def show(self):
        """
        Shows the current state.

        :return: None
        """
        if self.SHOW_RENDER:
            images = np.vstack(
                        (np.hstack((self.reference, self.drawer.get_canvas())),
                        np.hstack((self.drawer.get_distance_map(), self.drawer.get_color_map())))
            )
            cv2.imshow('Current State', images)
            cv2.waitKey(0)
            return images


if __name__ == "__main__":
    env = DrawingEnvironment()
    total_reward = 0
    SAVE_VIDEO = True
    # if SAVE_VIDEO:
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     out = cv2.VideoWriter('output.avi',fourcc, 20.0, (84*2,84*2))
    for i in range(20):
        action = np.random.randint(0, 242)
        new_observation, reward, done = env.step(action=action)
        total_reward += reward
        # print(f"Step {i+1} {reward} \t {total_reward}")
        images = env.show()
        if SAVE_VIDEO:
            images = (np.vstack(
                        (np.hstack((new_observation[0], new_observation[1])),
                        np.hstack((new_observation[2], new_observation[3])))
            )*255).astype("uint8")
            images = cv2.cvtColor(images, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Video", images)
            cv2.imshow("patch", new_observation[4])
            # out.write(images)
        
    # out.release()