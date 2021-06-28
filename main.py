
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from style_image.inversion import optimize_latent_codes


def load_image(img_path, *args, **kwargs):
    return cv2.imread(img_path, *args, **kwargs)


def write_image(img_path, im, *args, **kwargs):
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    cv2.imwrite(img_path, im, *args, **kwargs)


def show_img(img):
    cv2.imshow(img)


def blur_image(im, k):
    return cv2.filter2D(im, -1, k)



class Config(object):
    def __init__(self):
        self.imgs_dir = "./data"
        self.reconstructions_dir = "./out/reconstructions"
        self.latents_dir = "./out/latents"
        self.optimizer = 'adam'
        self.input_img_size = (1024, 1024)
        self.perceptual_img_size =(256, 256)
        self.learning_rate = 1e-3
        self.total_iterations = 1000
        self.cache_dir = "./cache"

def create_blur_kernels():
    filters_shape = (30,30)
    k1 = np.zeros(filters_shape)
    k1[((k1.shape[0]-1)//2)] = 1
    k1 /= k1.sum()
    k2 = k1.T
    k3 = np.eye(filters_shape[0])
    k3 /= k3.sum()
    k4 = np.fliplr(k3)

if __name__ == '__main__':
    optimize_latent_codes(Config())