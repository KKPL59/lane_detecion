import torch as T
from torch import nn
from torch import optim
from torch import flatten
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
import json
import pandas as pd
import pickle
import os
import random
import shutil



class DataGenerator():
    def __init__(self, batch_size, width, height):
        self.batch_size = batch_size
        self.width = width
        self.height = height

    def get_lists(self, label_path):
        paths_list = []
        lanes_list = []

        with open(label_path, "r") as f:
            for i in f:
                dict_json = json.loads(i)
                lanes_list.append(T.tensor(dict_json["lanes"]))
                paths_list.append(dict_json["raw_file"])

        return paths_list, lanes_list

    def get_data(self, paths_list_, label_list):
        image_list = []
        tensor_list = []
        list_y = [240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430,
                  440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630,
                  640, 650, 660, 670, 680, 690, 700, 710]

        paths_list = [random.choice(paths_list_) for _ in range(self.batch_size)]
        labels_index = [paths_list_.index(x) for x in paths_list]

        # labels_index = [random.choice(range(len(paths_list_))) for _ in range(self.batch_size)]
        # paths_list = [paths_list_[x] for x in labels_index]

        for idx, file in enumerate(paths_list):
            print(idx)
            for img in range(1, 21):
                img_path = rf"C:\Users\lukas\PycharmProjects\line_detection\datasets\archive (10)\TUSimple\train_set\{file[:-6]}{img}.jpg"
                numpy_image = cv2.imread(img_path)
                numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)
                numpy_image = cv2.resize(numpy_image, (self.width, self.height))
                # cv2.imshow("test1", numpy_image)
                # cv2.waitKey(0)
                # numpy_image = numpy_image / 255. ###################################################################################
                image_list.append(T.tensor(numpy_image))

            tensor_zeros = T.zeros((20, self.height, self.width))
            for index, i in enumerate(image_list):
                #         print(i.shape)
                tensor_zeros[index] = i

            tensor_list.append(tensor_zeros)
            image_list = []

            tensor_zeros = T.zeros((self.batch_size, 20, self.height, self.width))
            for index, i in enumerate(tensor_list):
                tensor_zeros[index] = i

        labels_tensor = T.full((self.batch_size, 5, 48), -2)
        for index1, i in enumerate(labels_index):
            label = label_list[i]
            for index, j in enumerate(label):
                labels_tensor[index1][index] = j[:48]

        image = np.zeros((self.batch_size, 720, 1280), dtype="uint8")
        image_resized = np.zeros((self.batch_size, 128, 256), dtype="uint8")

        for batch in range(self.batch_size):
            for lane in range(5):
                points = []
                for label_x, label_y in zip(labels_tensor[batch][lane], list_y):
                    if label_x == -2:
                        continue

                    points.append([label_x.item(), label_y])

                points = np.array(points)
                points = points.reshape((-1, 1, 2))
                image[batch] = cv2.polylines(image[batch], [points], False, (255, 255, 255), 15)
                # image_resized[batch] = cv2.resize(image[batch], (320, 176)) -------------------
                image_resized[batch] = cv2.resize(image[batch], (256, 128))
                # cv2.imshow("image", image_resized[batch])
                # cv2.waitKey(0)
                # print(np.count_nonzero(image_resized[batch]))

            # cv2.imshow("img", image_resized[batch])
            # cv2.waitKey(0)
            # print(np.count_nonzero(image_resized[batch]))
        print(tensor_zeros[0].shape)
        # return tensor_zeros[:, :], T.tensor(image_resized, dtype=T.float)
        return tensor_zeros[:, 15:], T.tensor(image_resized, dtype=T.float)


#
# width, height = 427, 240
# data_gen = DataGenerator(3, width, height)
#
# paths_list1, lanes_list1 = data_gen.get_lists(
#     r"C:\Users\lukas\PycharmProjects\line_detection\datasets\archive (10)\TUSimple\train_set\label_data_0313.json")
# paths_list2, lanes_list2 = data_gen.get_lists(
#     r"C:\Users\lukas\PycharmProjects\line_detection\datasets\archive (10)\TUSimple\train_set\label_data_0531.json")
# paths_list3, lanes_list3 = data_gen.get_lists(
#     r"C:\Users\lukas\PycharmProjects\line_detection\datasets\archive (10)\TUSimple\train_set\label_data_0601.json")
#
# paths_list1.extend(paths_list2)
# paths_list1.extend(paths_list3)
#
# lanes_list1.extend(lanes_list2)
# lanes_list1.extend(lanes_list3)
#
# data, target = data_gen.get_data(paths_list1, lanes_list1)
# print(data.shape)
# print(target.shape)
