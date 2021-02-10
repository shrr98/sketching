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


if __name__=="__main__":
    ref = np.random.randint(0,255, (84,84))
    can = np.zeros((84,84))

    print(pixel_similarity(ref, can))