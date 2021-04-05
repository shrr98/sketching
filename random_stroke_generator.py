import numpy as np
from tensorflow import keras
from environment.drawer import Drawer
import math
import cv2
from utils.utils import position_to_action

class RandomStrokeGenerator(keras.utils.Sequence):
    """
    [?] 
    """
    MAX_STROKES = 64

    def __init__(self, batch_size, num_data, min_strokes=8, max_strokes = 16, jumping_rate=0, max_jumping_step=41):
        self.batch_size = batch_size
        self.num_data = num_data
        self.MIN_STROKES = min_strokes
        self.MAX_STROKES = max_strokes
        self.curr_step = 8      # curriculum step
        self.curr_epoch = 3     # curriculum epoch
        self.min_strokes = min_strokes
        self.max_strokes = min_strokes + self.curr_step
        self.epoch = 0

        # params for pen jumping
        self.jumping_rate = jumping_rate
        self.max_jumping_step = max_jumping_step

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.num_data / self.batch_size)

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min( batch_start + self.batch_size, self.num_data)
        return [self.X[batch_start:batch_end,:,:,:], self.patches[batch_start:batch_end,:,:,:]], self.y[batch_start:batch_end]

    def get_pen_jumping(self, pen_position, canvas_size, pen_step):
        '''
        Jump the pen to a random position.
        '''
    
        max_x = self.max_jumping_step
        min_x = -self.max_jumping_step
        max_y = self.max_jumping_step
        min_y = -self.max_jumping_step

        if pen_position[0] + max_x >= canvas_size:
            max_x = (canvas_size-1) - pen_position[0]
        elif pen_position[0] + min_x < 0:
            min_x = -pen_position[0]


        if pen_position[1] + max_y >= canvas_size:
            max_y = (canvas_size-1) - pen_position[1]
        elif pen_position[1] + min_y < 0:
            min_y = -pen_position[1]

        x_jump = np.random.randint(min_x, max_x, 1)[0]
        y_jump = np.random.randint(min_y, max_y, 1)[0]
        
        x_direction = 1 if x_jump >=0 else -1
        y_direction = 1 if y_jump >= 0 else -1
        
        if x_jump==0 and y_jump==0: # if not jumping
            return None

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

        return moves

    
    def generate(self):
        drawer = Drawer()
        drawer.reset()

        actions = np.random.randint(0, drawer.ACTION_SPACE_SIZE, self.num_data)
        
        color_maps = []
        distance_maps = []
        canvases = []
        canv_patches = []
        ref_patches = []
        ref = None
        y = []
        X = []
        patches = []
        pen_positions = []
        num_strokes = 0
        max_strokes = np.random.randint(self.min_strokes, self.max_strokes, 1)[0]
        for i in range(actions.shape[0]):
            if num_strokes < max_strokes-1 and np.random.random(1)[0] <= self.jumping_rate:
                jumping_steps = self.get_pen_jumping(drawer.get_pen_position(), drawer.CANVAS_SIZE, drawer.PATCH_SIZE//2)
                if jumping_steps is not None:   # if jumping
                    # print("jump")
                    for jump in jumping_steps:
                        num_strokes += 1
                        jump_action = position_to_action(jump, 0, drawer.PATCH_SIZE)
                        color_maps.append(drawer.get_color_map())
                        distance_maps.append(drawer.get_distance_map())
                        pen_positions.append(drawer.get_pen_position())
                        canvases.append(drawer.get_canvas())
                        canv_patches.append(drawer.get_patch())
                        action_done = drawer.do_action(jump_action)
                        y.append(action_done['action_real'])
                        if num_strokes == max_strokes-1:
                            break

            num_strokes += 1
            color_maps.append(drawer.get_color_map())
            distance_maps.append(drawer.get_distance_map())
            pen_positions.append(drawer.get_pen_position())
            canvases.append(drawer.get_canvas())
            canv_patches.append(drawer.get_patch())
            action_done = drawer.do_action(actions[i])
            y.append(action_done["action_real"])
            

            if num_strokes == max_strokes or i == self.num_data-1:
                # proceess the ref
                ref = drawer.get_canvas()
                for pos in pen_positions:
                    #get patches based on pen position every step.
                    ref_patches.append(drawer.get_patch(pos))
                
                for can, dis, col, cp, rp in zip(canvases, distance_maps, color_maps, canv_patches, ref_patches):
                    x = np.stack( (can, ref, dis, col), axis=2)
                    X.append(x)

                    p = np.stack( (cp, rp), axis=2)
                    patches.append(p)
                
                drawer.reset()  # Reset drawer state

                # clear all buffers
                pen_positions.clear()
                canvases.clear()
                distance_maps.clear()
                color_maps.clear()
                canv_patches.clear()
                ref_patches.clear()

                # Reset strokes
                num_strokes = 0
                max_strokes = np.random.randint(self.min_strokes, self.max_strokes, 1)[0]


        # Shuffle the dataset
        indices = np.arange(self.num_data)
        np.random.shuffle(indices)
        X = np.array(X, dtype=np.float)[indices]
        patches = np.array(patches, dtype=np.float)[indices]
        y = np.array(y, dtype=np.int)[indices]

        # Squeeze the dims
        self.X =  np.squeeze(X)
        self.patches = np.squeeze(patches)
        self.y = np.squeeze(y)
    
    def on_epoch_end(self):
        """
        Generate new data at every end of epoch.
        """
        self.epoch += 1
        if self.epoch % self.curr_epoch == 0 and self.max_strokes < self.MAX_STROKES:
            self.min_strokes += self.curr_step
            self.max_strokes += self.curr_step

        self.generate()



if __name__ == "__main__":
    gen = RandomStrokeGenerator(batch_size=32,num_data=256, max_strokes=256)

    # moves = gen.randomize_pen_jumping((80,80), 84, 5)

    print(len(gen))

    for i in range(0,len(gen)):
        [X, x], y = gen.__getitem__(i)
        print(X.shape)
        print(x.shape)
        images = np.hstack((X[0][:,:,0], X[0][:,:,1], X[0][:,:,2], X[0][:,:,3], cv2.resize(x[0][:,:,0], (84,84)), cv2.resize(x[0][:,:,1], (84,84))))
        for j in range(1,8):
            im1 = np.hstack((X[j][:,:,0], X[j][:,:,1], X[j][:,:,2], X[j][:,:,3], cv2.resize(x[j][:,:,0], (84,84)), cv2.resize(x[j][:,:,1], (84,84))))
            images = np.vstack((images, im1))

        cv2.imshow("images", images)
        cv2.waitKey(0)