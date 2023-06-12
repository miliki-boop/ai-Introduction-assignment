import numpy as np
import random

def poison(y_train,poison_num):
    feature_idxs = np.random.choice(60000, poison_num, replace=False)
    for idx in feature_idxs:
        random_number = random.randint(0, 9)
        y_train[idx] = random_number
    return y_train


def poison_images(images, poison_percentage):
    num_images, num_pixels = images.shape
    num_pixels_to_poison = int(num_pixels * poison_percentage)
    poisoned_images = np.copy(images)

    for i in range(num_images):
        pixels_to_poison = np.random.choice(num_pixels, num_pixels_to_poison, replace=False)
        poisoned_images[i, pixels_to_poison] = np.random.randint(0, 256, num_pixels_to_poison)

    return poisoned_images