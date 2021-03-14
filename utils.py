import numpy as np

def pixel_similarity(reference, canvas):
    """
    Calculate pixel similarity given 2 images.

    :param reference: reference image
    :param canvas: imitation image
    :return: similarity value (float)
    """
    L = reference.shape[0]
    diff_square = np.subtract(reference, canvas)**2
    similarity = np.sum(diff_square) / L**2

    return similarity

def crop_image(img, bb):
    """
    :param img: image to be cropped
    :param bb: (top, left, bottom, right)
    :return: cropped image
    """
    top, left, bottom, right = bb
    row, col = img.shape
    target_row, target_col = (bottom-top, right-left)
    crop = np.full((bottom-top, right-left), 0, dtype=np.float)

    target_top, top = (0, top) if top>=0 else (-top, 0)
    target_left, left = (0, left) if left>=0 else (-left, 0)
    target_bottom, bottom = (target_row, bottom) if bottom<row else (target_row-(bottom-row), row)
    target_right, right = (target_col, right) if right<col else (target_col-(right-col), col)

    crop[target_top:target_bottom, target_left:target_right] = img[top:bottom, left:right]

    return crop

def position_to_action(position, pen_state, patch_size):
    """
    Convert a position step to action.
    :param position: position step (x,y)
    :param pen_state: the pen state (0: up, 1: down)
    :param patch_size: patch size, this also can be used to determine number of possible actions.
    :return: (int) action
    """
    x, y = position
    action = (x + 5) + (y + 5) * patch_size
    if pen_state:
        action += patch_size**2
    return action

if __name__=="__main__":
    ref = np.random.randint(0,255, (84,84))
    can = np.zeros((84,84))

    print(pixel_similarity(ref, can))