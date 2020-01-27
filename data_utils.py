import cv2, os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf


class DataSet():

    def __init__(self, rgb_path, thermal_path, label_path, data_type, img_size=512):
        self.rgb_path = np.array(rgb_path)
        self.thermal_path = np.array(thermal_path)
        self.label_path = np.array(label_path)
        self.batch_count = 0
        self.epoch_count = 0
        self.data_type = data_type
        self.img_size = img_size
        self.input_shape = (img_size, img_size)

    def num_examples(self):
        return self.label_path.shape[0]


    def next_batch(self, batch_size):
        start = self.batch_count * batch_size
        end = start + batch_size
        self.batch_count += 1

        if end > self.label_path.shape[0]:
            self.batch_count = 0
            random_index = np.random.permutation(self.label_path.shape[0])
            if self.data_type == 'rgbt':
                self.rgb_path = self.rgb_path[random_index]
                self.thermal_path = self.thermal_path[random_index]
            elif self.data_type == 'thermal':
                self.thermal_path = self.thermal_path[random_index]
            else:
                self.rgb_path = self.rgb_path[random_index]
            self.label_path = self.label_path[random_index]
            self.epoch_count += 1
            start = self.batch_count * batch_size
            end = start + batch_size
            self.batch_count += 1

        if self.data_type == 'rgbt':
            image_batch, label_batch = self.read_path(
                self.rgb_path[start:end], self.thermal_path[start:end], 
                self.label_path[start:end])
        elif self.data_type == 'thermal':
            image_batch, label_batch = self.read_path(
                None, self.thermal_path[start:end], 
                self.label_path[start:end])
        else:
            image_batch, label_batch = self.read_path(
                self.rgb_path[start:end], None,
                self.label_path[start:end])
        
        return image_batch, label_batch

    def read_path(self, rgb_path, thermal_path, label_path):
        x = []
        y = []
        for i in range(label_path.shape[0]):
            if self.data_type == 'thermal':
                thermal_img = cv2.resize(cv2.imread(thermal_path[i], 0), self.input_shape)\
                    .reshape((self.img_size, self.img_size, 1))
                x.append(thermal_img)
            elif self.data_type == 'rgbt':
                rgb_img = cv2.resize(cv2.imread(rgb_path[i], 1), self.input_shape)
                thermal_img = cv2.resize(cv2.imread(thermal_path[i], 0), self.input_shape) \
                    .reshape((self.img_size, self.img_size, 1))
                rgbt_img = np.concatenate((rgb_img, thermal_img), axis=2)
                x.append(rgbt_img)
            else:
                rgb_img = cv2.resize(cv2.imread(rgb_path[i], 1), self.input_shape)
                x.append(rgb_img)
            y.append(cv2.resize(np.array(Image.open(label_path[i])), self.input_shape))

        return np.array(x), np.array(y)
