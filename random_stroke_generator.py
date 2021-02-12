import numpy as np
from tensorflow import keras
from drawing_env import Drawer
import math
import cv2

class RandomStrokeGenerator(keras.utils.Sequence):
    """
    [?] 
    """
    MAX_STROKES = 64

    def __init__(self, batch_size, num_data):
        self.batch_size = batch_size
        self.num_data = num_data

    def __len__(self):
        return math.ceil(self.num_data / self.batch_size)

    def __getitem__(self, idx):
        drawer = Drawer()
        drawer.reset()
        # skip_strokes = np.random.randint(1,5, self.batch_size)
        actions = np.random.randint(0, drawer.ACTION_SPACE_SIZE, 5*self.batch_size+1)
        # next_checkpt = skip_strokes[0]
        color_maps = []
        distance_maps = []
        canvases = []
        patches = []
        y = []
        for i in range(actions.shape[0]):
            action_done = drawer.do_action(actions[i])
            if i%5==4:
                color_maps.append(drawer.get_color_map())
                distance_maps.append(drawer.get_distance_map())
                canvases.append(drawer.get_canvas())
                patches.append(drawer.get_patch())
                y.append(actions[i+1])
        ref = drawer.get_canvas()
        return np.array([
                [ref, can, dis, col, pat] for can,dis,col, pat in zip (canvases, distance_maps, color_maps, patches)
            ]), np.array(y)

    
    def on_epoch_end(self):
        self.num_strokes = 0


if __name__ == "__main__":
    gen = RandomStrokeGenerator(batch_size=4,num_data=8)

    for i in range(5):
        X, y = gen.__getitem__(0)
        images = np.hstack((X[0][0], X[0][1], X[0][2], X[0][3], cv2.resize(X[0][4], (84,84))))
        for j in range(1,4):
            im1 = np.hstack((X[j][0], X[j][1], X[j][2], X[j][3], cv2.resize(X[j][4], (84,84))))
            images = np.vstack((images, im1))

        cv2.imshow("images", images)
        cv2.waitKey(0)