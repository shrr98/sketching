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

    def __init__(self, batch_size, num_data, max_strokes = 16):
        self.batch_size = batch_size
        self.num_data = num_data
        self.max_strokes = max_strokes
        self.generate()

    def __len__(self):
        return math.ceil(self.num_data / self.batch_size)

    def __getitem__(self, idx):
        # drawer = Drawer()
        # drawer.reset()
        # # skip_strokes = np.random.randint(1,5, self.batch_size)
        # actions = np.random.randint(0, drawer.ACTION_SPACE_SIZE, self.batch_size+1)
        # # next_checkpt = skip_strokes[0]
        # color_maps = []
        # distance_maps = []
        # canvases = []
        # patches = []
        # refs = []
        # y = []
        # for i in range(actions.shape[0]):
        #     color_maps.append(drawer.get_color_map())
        #     distance_maps.append(drawer.get_distance_map())
        #     canvases.append(drawer.get_canvas())
        #     patches.append(drawer.get_patch())
        #     y.append(actions[i])
        #     action_done = drawer.do_action(actions[i])
        #     refs.append(drawer.get_canvas())
        # X = []

        # for can, ref, dis, col in zip(canvases, refs, distance_maps, color_maps):
        #     # print(ref.shape, can.shape, dis.shape, col, shape, pat,shape)
        #     x = np.stack( (ref, can, dis, col), axis=2)
        #     # x = ref
        #     X.append(x)
        # # print("getitem hampir return")

        # # Shuffle the batch
        # indices = np.arange(self.batch_size)
        # np.random.shuffle(indices)
        # X = np.array(X, dtype=np.float)[indices]
        # patches = np.array(patches, dtype=np.float)[indices]
        # y = np.array(y, dtype=np.int)[indices]
        # return [np.squeeze(X), np.squeeze(patches)], np.squeeze(y)
        batch_start = idx * self.batch_size
        batch_end = min( batch_start + self.batch_size, self.num_data)
        return [self.X[batch_start:batch_end,:,:,:], self.patches[batch_start:batch_end,:,:]], self.y[batch_start:batch_end]
    
    def generate(self):
        drawer = Drawer()
        drawer.reset()
        actions = np.random.randint(0, drawer.ACTION_SPACE_SIZE/2, self.num_data)
        color_maps = []
        distance_maps = []
        canvases = []
        patches = []
        refs = []
        y = []
        num_strokes = 0
        for i in range(actions.shape[0]):
            num_strokes += 1
            color_maps.append(drawer.get_color_map())
            distance_maps.append(drawer.get_distance_map())
            canvases.append(drawer.get_canvas())
            patches.append(drawer.get_patch())
            y.append(actions[i])
            action_done = drawer.do_action(actions[i])
            refs.append(drawer.get_canvas())
            if num_strokes == self.max_strokes:
                drawer.reset()
                num_strokes = 0
        X = []

        for can, ref, dis, col in zip(canvases, refs, distance_maps, color_maps):
            # print(ref.shape, can.shape, dis.shape, col, shape, pat,shape)
            x = np.stack( (ref, can, dis, col), axis=2)
            # x = ref
            X.append(x)
        # print("getitem hampir return")

        # Shuffle the batch
        indices = np.arange(self.num_data)
        np.random.shuffle(indices)
        X = np.array(X, dtype=np.float)[indices]
        patches = np.array(patches, dtype=np.float)[indices]
        y = np.array(y, dtype=np.int)[indices]

        self.X =  np.squeeze(X)
        self.patches = np.squeeze(patches)
        self.y = np.squeeze(y)
    
    def on_epoch_end(self):
        """
        still not used
        """
        self.num_strokes = 0
        indices = np.arange(self.num_data)
        np.random.shuffle(indices)
        X = np.array(self.X, dtype=np.float)[indices]
        patches = np.array(self.patches, dtype=np.float)[indices]
        y = np.array(self.y, dtype=np.int)[indices]

        self.X =  np.squeeze(X)
        self.patches = np.squeeze(patches)
        self.y = np.squeeze(y)


if __name__ == "__main__":
    gen = RandomStrokeGenerator(batch_size=8,num_data=80, max_strokes=8)

    for i in range(1,10):
        [X, x], y = gen.__getitem__(i)
        print(X.shape)
        print(x.shape)
        images = np.hstack((X[0][:,:,0], X[0][:,:,1], X[0][:,:,2], X[0][:,:,3], cv2.resize(x[0], (84,84))))
        for j in range(1,4):
            im1 = np.hstack((X[j][:,:,0], X[j][:,:,1], X[j][:,:,2], X[j][:,:,3], cv2.resize(x[j], (84,84))))
            images = np.vstack((images, im1))

        cv2.imshow("images", images)
        cv2.waitKey(0)