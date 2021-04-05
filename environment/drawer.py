import cv2
import numpy as np
from utils.utils import pixel_similarity, crop_image, position_to_action
import math
from environment.reference import Reference

class Drawer(Reference):
    """
    Class implementasi Drawer untuk DoodleSQN.

    Image (canvas, distance_map, dan color_map) dinormalisasi
    sehingga nilai per piksel antara 0-1.
    """

    def __init__(self):
        self.canvas = None          # canvas untuk menggambar
        self.distance_map = None    # distance map untuk posisi pena
        self.color_map = None       # colormap untuk keadaan pena 
        self.pen_position = (0,0)   # posisi pena pada kanvas
        self.pen_state = 0          # kondisi pena (0: up, 1: down)

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
        
        if self.pen_state: # 1. Pena down
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
        
        self.pen_position = (np.random.randint(0,self.CANVAS_SIZE), np.random.randint(0,self.CANVAS_SIZE))
        self.move_pen(self.pen_position)
        self.pen_state = 0

        self.color_map = np.full(self.CANVAS_IMG_SIZE, 0, dtype=np.float)


    def func_l2_distance(self, x, y):
        """
        Calculate L2 distance map.

        :param x: coordinate x of the pixel
        :param y: coordinate y of the pixel

        :return: float
        """
        return np.sqrt((x-self.pen_position[1])**2 + (y-self.pen_position[0])**2) / self.CANVAS_SIZE


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
        self.color_map[:] = self.pen_state
        return np.copy(self.color_map)
