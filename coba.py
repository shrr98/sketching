import numpy as np
import cv2

def crop_image(img, bb):
    """
    :param bb: (top, left, bottom, right)
    :return: cropped image
    """
    top, left, bottom, right = bb
    row, col = img.shape
    target_row, target_col = (bottom-top, right-left)
    crop = np.full((bottom-top, right-left), 255, dtype=np.uint8)

    target_top, top = (0, top) if top>=0 else (-top, 0)
    target_left, left = (0, left) if left>=0 else (-left, 0)
    target_bottom, bottom = (target_row, bottom) if bottom<row else (target_row-(bottom-row), row)
    target_right, right = (target_col, right) if right<col else (target_col-(right-col), col)

    print(target_top, target_left, target_bottom, target_right)
    print(top, left, bottom, right)
    crop[target_top:target_bottom, target_left:target_right] = img[top:bottom, left:right]

    return crop


image = np.random.randint(0, 255, (84,84), dtype=np.uint8)

x = crop_image(image, (-5, -2, 2, 8))
# x = crop_image(image, (80, 83, 85, 88))

cv2.imshow("image",image)
cv2.imshow("x",x)
cv2.waitKey(0)