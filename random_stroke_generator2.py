import numpy as np
from tensorflow import keras
from environment.drawer import Drawer
import math
import cv2
from utils.utils import position_to_action, action_to_position

class RandomStrokeGenerator(keras.utils.Sequence):
    """
    [?] 
    """
    MAX_STROKES = 64
    CANVAS_SIZE = 84

    def __init__(self, batch_size, num_data, min_strokes=8, max_strokes = 16, filename='out.txt'):
        self.batch_size = batch_size
        self.num_data = num_data
        self.epoch = 0

        with open(filename, 'r') as f:
            lines = f.readlines()
        
        self.lines = lines[:-1] # ignore the last empty line
        self.num_lines = len(self.lines)

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.num_data / self.batch_size)

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min( batch_start + self.batch_size, self.num_data)
        return [self.X[batch_start:batch_end,:,:,:], self.patches[batch_start:batch_end,:,:,:]], self.y[batch_start:batch_end]


    def get_random_action(self, pen_position, pen_state=None):
        pen_state = np.random.randint(0, 2) if pen_state==None else pen_state
        x_left = min(5, pen_position[0])
        x_right = max(83-5, pen_position[0])

        y_top = min(5, pen_position[1])
        y_bot = max(83-5, pen_position[1])

        x = np.random.randint(-x_left, 84-x_right)
        y = np.random.randint(-y_top, 84-y_bot)

        return position_to_action((x,y), pen_state, 11)

    def parse(self, line):
        '''
        Parse the line to reconstruct the demonstrated strokes
        :return init_pos: initial position of the pen to start drawing
        :return actions: sequence of actions recorded 
        '''
        data = line.strip('\n').split(';')
        boundary = tuple(int(x) for x in data[0].split(','))
        init_pos = boundary[:2]
        top, left, bottom, right = boundary[2:] # top, left, bottom, right
        init_pos_todraw = (
            np.random.randint(init_pos[0] - left, self.CANVAS_SIZE - (right-init_pos[0])),
            np.random.randint(init_pos[1] - top, self.CANVAS_SIZE - (bottom-init_pos[1]))
        )

        actions = []
        for d in data[1:-1]:
            d = d.split(',')
            stroke = (int(d[0]), int(d[1]))
            pen = int(d[2])
            action = position_to_action(stroke, pen)
            actions.append(action)

        return init_pos_todraw, actions


    def move_sequence(self, current, target):
        '''
        Generate sequence of movements from current to target
        on pen up.
        '''
        moves = []
        pen_step = 5

        x_jump = target[0] - current[0]
        y_jump = target[1] - current[1]

        if x_jump==0 and y_jump==0: # if not jumping
            return []

        x_direction = 1 if x_jump >=0 else -1
        y_direction = 1 if y_jump >= 0 else -1

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

        
        # pushing the moves
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
        drawer.pen_state = 1

        ref_drawer = Drawer()
        ref_drawer.reset()
        
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

        while num_strokes < self.num_data:
            num_examples = np.random.randint(1, 2)
            lines = np.random.choice(self.lines, num_examples)

            # reset all states
            drawer.reset()  # Reset drawer state
            drawer.pen_state = 1
            ref_drawer.reset() # reset ref drawer

            # clear all buffers
            pen_positions.clear()
            canvases.clear()
            distance_maps.clear()
            color_maps.clear()
            canv_patches.clear()
            ref_patches.clear()

            for line in lines:
                init_pos, actions = self.parse(line)
                current_pos = drawer.get_pen_position()

                # move to init_pos
                movements = self.move_sequence(current_pos, init_pos)

                for m in movements:
                    a = position_to_action(m, 0, drawer.PATCH_SIZE)

                    # save current state of drawer
                    color_maps.append(drawer.get_color_map())
                    distance_maps.append(drawer.get_distance_map())
                    pen_positions.append(drawer.get_pen_position())
                    canvases.append(drawer.get_canvas())
                    canv_patches.append(drawer.get_patch())

                    action_done = drawer.do_action(a)
                    y.append(action_done["action_real"])

                # set reference drawer state to drawer's
                ref_drawer.set_pen_position(init_pos)

                for a in actions:
                    # save current state of drawer
                    color_maps.append(drawer.get_color_map())
                    distance_maps.append(drawer.get_distance_map())
                    pen_positions.append(drawer.get_pen_position())
                    canvases.append(drawer.get_canvas())
                    canv_patches.append(drawer.get_patch())

                    action_done = drawer.do_action(a)
                    y.append(action_done["action_real"])

                    ref_drawer.do_action(a)

            
                num_strokes += len(movements) + len(actions)

            # proceess the ref
            ref = ref_drawer.get_canvas()
            for pos in pen_positions:
                #get patches based on pen position every step.
                ref_patches.append(drawer.get_patch(pos))
            
            for can, dis, col, cp, rp in zip(canvases, distance_maps, color_maps, canv_patches, ref_patches):
                x = np.stack( (can, ref, dis, col), axis=2)
                X.append(x)

                p = np.stack( (cp, rp), axis=2)
                patches.append(p)
            
        
        # Shuffle the dataset
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        X = np.array(X, dtype=np.float)[indices]
        patches = np.array(patches, dtype=np.float)[indices]
        y = np.array(y, dtype=np.int)[indices]
        # print(y[y<121].shape, y[y>=121].shape)

        # Squeeze the dims
        self.X =  np.squeeze(X)
        self.patches = np.squeeze(patches)
        self.y = np.squeeze(y)

    def on_epoch_end(self):
        """
        Generate new data at every end of epoch.
        """

        self.generate()

        self.epoch += 1


if __name__ == "__main__":
    gen = RandomStrokeGenerator(batch_size=16,
                                num_data=484, 
                                min_strokes=64, 
                                max_strokes=64,
                                filename='valid.txt'
                            )

    # moves = gen.randomize_pen_jumping((80,80), 84, 5)

    print(len(gen))

    for i in range(0,len(gen)):
        [X, x], y = gen.__getitem__(i)
        print(X.shape)
        images = np.hstack((X[0][:,:,0], X[0][:,:,1], X[0][:,:,2], X[0][:,:,3], cv2.resize(x[0][:,:,0], (84,84)), cv2.resize(x[0][:,:,1], (84,84))))
        for j in range(1,X.shape[0]):
            im1 = np.hstack((X[j][:,:,0], X[j][:,:,1], X[j][:,:,2], X[j][:,:,3], cv2.resize(x[j][:,:,0], (84,84)), cv2.resize(x[j][:,:,1], (84,84))))
            images = np.vstack((images, im1))

        cv2.imshow("images", images)
        # im = np.array(255*X[0][:,:,1]).astype(np.uint8)
        # cv2.imwrite("random_images/{}.png".format(i), im)
        cv2.waitKey(0)

    # key, val = np.unique(y, return_counts=True, return_index=True)
    # print("\n".join(a for a in zip(key, val)))
        