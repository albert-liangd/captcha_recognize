#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-12-04 15:05
# @Author  : Albert liang
# @Email   : ld602199512@gmail.com
# @File    : data_helper.py

from PIL import Image
import numpy as np
import random
import os


current_dir = os.path.abspath(".")
base = os.path.join(current_dir, "data/")

# initialize global variable
num = [str(n) for n in range(10)]    # transform int number to str
alphabet = [chr(i) for i in range(97, 123)]  # lower alphabet
Alphabet = [s.upper() for s in alphabet]      # upper alphabet
total_ele = num + alphabet + Alphabet
ele2index = {e: i for i, e in enumerate(total_ele)}

max_captcha = 4  # the max length of the words
char_set_len = len(total_ele)

# Image  info
img_height = 40
img_width = 120


# transform photography to numpy array
def read_all_pic(fp):
    img = Image.open(fp)
    pic_array = np.array(img)
    return pic_array


def get_data():
    all_image = os.listdir(base)
    name = random.choice([name[:-4] for name in all_image])
    image = Image.open(base + name + ".png")
    image = np.array(image)
    image = transform_pic(image)
    return image, name


def load_train_data():
    pass


# transform photograph to grey-scale map
def transform_pic(img):
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    return img


# transform label to one-hot vector
def text2vec(name):
    vector = np.zeros(max_captcha * len(total_ele))
    for index, char in enumerate(name):
        idx = index * char_set_len + total_ele.index(char)
        vector[idx] = 1
    return vector


# transform vector to label
def vec2text(vector):
    vector = vector.reshape([max_captcha, -1])
    text = ''
    for item in vector:
        index = np.argmax(item)
        text += total_ele[index]
    return text


# data iteration
def data_iter(batch_size=64):
    batch_x = np.zeros([batch_size, img_height * img_width])
    batch_y = np.zeros([batch_size, max_captcha * char_set_len])
    for i in range(batch_size):
        image, label = get_data()
        batch_x[i, :] = (image.flatten()) / 255
        batch_y[i, :] = text2vec(label)
    return batch_x.reshape([-1, 40, 120, 1]), batch_y