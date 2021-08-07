import cv2
import numpy as np
from utils.utils import pixel_similarity, crop_image, position_to_action
import math

class Reference(object):
    """
    Class implementasi Reference Image untuk DoodleSQN.

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
        self.pen_position = (0,0)

    def load_canvas(self, filename):
        """
        Load image from filename and set to canvas.
        """
        img = cv2.imread(filename)
        img = cv2.resize(img, (84,84))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img = cv2.threshold(img, 120, 255, 0)
        img = 1-(img / 255)
        # img = img / 255
        self.canvas = np.array(img, dtype=np.float)

    def set_canvas(self, canvas):
        self.canvas = np.copy(canvas)

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