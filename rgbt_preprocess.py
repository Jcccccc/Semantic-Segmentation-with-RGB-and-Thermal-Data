import os
import csv
import cv2
import json
import random
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


img_width, img_height = 3600, 2700
crop_width, crop_height = 1500, 1500
#resize_width, resize_height = 512, 512
origin_count, train_count, test_count = 0, 0, 0
thermal_sum = 0
CROP_NUM_PER_IMAGE = 8

test_image_list = ['20181210_100253_338_R',
                   '20181210_101156_717_R',
                   '20181210_093820_224_R',
                   '20181210_092807_710_R',
                   '20181210_104820_397_R',
                   '20181210_103932_201_R',
                   '20190112_103653_644_R',
                   '20190112_091456_748_R',
                   '20190112_102919_191_R',
                   '20190112_093301_457_R']

csv_files = ['./dataset/Cityzone/convertannotationsNorth1.csv',
             './dataset/Cityzone/convertannotationsSouth1.csv',
             './dataset/Cityzone/convertannotationsEast1.csv',
             './dataset/Cityzone/convertannotationsWest1.csv',
             './dataset/CampusNorth_zone1/convertannotationsNorth.csv', 
             './dataset/CampusNorth_zone1/convertannotationsSE.csv', 
             './dataset/CampusNorth_zone1/convertannotationsSW.csv']
rgb_dir = ['./dataset/Cityzone/CityZone1_North/RGB',
            './dataset/Cityzone/CityZone1_South/RGB',
            './dataset/Cityzone/CityZone1_East/RGB',
            './dataset/Cityzone/CityZone1_West/RGB',
            './dataset/CampusNorth_zone1/North_RGB',
            './dataset/CampusNorth_zone1/SE_RGB',
            './dataset/CampusNorth_zone1/SW_RGB']
thermal_dir = ['./dataset/Cityzone/CityZone1_North/thermal',
            './dataset/Cityzone/CityZone1_South/thermal',
            './dataset/Cityzone/CityZone1_East/thermal',
            './dataset/Cityzone/CityZone1_West/thermal',
            './dataset/CampusNorth_zone1/North_thermal',
            './dataset/CampusNorth_zone1/SE_thermal',
            './dataset/CampusNorth_zone1/SW_thermal']
output_dir = './rgbt_balance_v2'
rgb_paths, thermal_paths, label_paths = [], [], []
test_rgb_paths, test_thermal_paths, test_label_paths = [], [], []


def voc_colormap(N=256):
    bitget = lambda val, idx: ((val & (1 << idx)) != 0)
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3

        cmap[i, :] = [r, g, b]
    return cmap


def checked(cropped, low=0.2, high=0.9):
    if not cropped:
        return False
    crop_arr = np.asarray(cropped, dtype=np.uint8)
    non_zero = np.count_nonzero(crop_arr)
    total = np.prod(crop_arr.shape)
    if non_zero > total*low and non_zero < total*high:
        return True
    else:
        return False


def crop_and_save(img_name, img, img_rgb, img_thermal, crop_num=CROP_NUM_PER_IMAGE):
    global train_count
    global test_count
    colormap = voc_colormap()
    if img_name in test_image_list:
        frag = 'test'
    else:
        frag = 'train'
    for i in range(crop_num):
        tries = 0
        cropped_label = None
        while not checked(cropped_label) and tries < 10:
            top = random.randint(0, img_height-crop_height)
            left = random.randint(0, img_width-crop_width)
            right, bottom = left+crop_width, top+crop_height
            cropped_label = img.crop((left, top, right, bottom))
            tries += 1
        if tries == 10:
            continue

        cropped_label.putpalette((colormap).astype(np.uint8).flatten())
        cropped_rgb = img_rgb.crop((left, top, right, bottom))
        cropped_thermal = img_thermal.crop((left, top, right, bottom))
        label_unique = np.unique(np.array(cropped_label))
        if 3 in label_unique or 5 in label_unique:
            labels, thermals, rgbs = [], [], []
            for j in range(4):
                labels.append(cropped_label.rotate(90*j))
                thermals.append(cropped_thermal.rotate(90*j))
                rgbs.append(cropped_rgb.rotate(90*j))
            labels.append(cropped_label.transpose(Image.FLIP_LEFT_RIGHT))
            thermals.append(cropped_thermal.transpose(Image.FLIP_LEFT_RIGHT))
            rgbs.append(cropped_rgb.transpose(Image.FLIP_LEFT_RIGHT))
            labels.append(cropped_label.transpose(Image.FLIP_TOP_BOTTOM))
            thermals.append(cropped_thermal.transpose(Image.FLIP_TOP_BOTTOM))
            rgbs.append(cropped_rgb.transpose(Image.FLIP_TOP_BOTTOM))
            for j in range(6):
                rgb_path = os.path.join(output_dir, frag, 'rgb',
                                        img_name+'_{}{}.jpg'.format(i, j))
                thermal_path = os.path.join(output_dir, frag, 'thermal',
                                            img_name+'_{}{}.jpg'.format(i, j))
                label_path = os.path.join(output_dir, frag, 'label', 
                                          img_name+'_{}{}.png'.format(i, j))
                cropped_label.save(label_path)
                cropped_rgb.save(rgb_path)
                cropped_thermal.save(thermal_path)
                
                if frag == 'train':
                    for k in range(10):
                        rgb_paths.append(rgb_path)
                        thermal_paths.append(thermal_path)
                        label_paths.append(label_path)
                    train_count += 1
                else:
                    test_rgb_paths.append(rgb_path)
                    test_thermal_paths.append(thermal_path)
                    test_label_paths.append(label_path)
                    test_count += 1
                print('Saved '+label_path)
        else:
            rgb_path = os.path.join(output_dir, frag, 'rgb',
                                    img_name+'_{}.jpg'.format(i))
            thermal_path = os.path.join(output_dir, frag, 'thermal',
                                        img_name+'_{}.jpg'.format(i))
            label_path = os.path.join(output_dir, frag, 'label', 
                                      img_name+'_{}.png'.format(i))
            cropped_label.save(label_path)
            cropped_rgb.save(rgb_path)
            cropped_thermal.save(thermal_path)
            if frag == 'train':
                rgb_paths.append(rgb_path)
                thermal_paths.append(thermal_path)
                label_paths.append(label_path)
                train_count += 1
            else:
                test_rgb_paths.append(rgb_path)
                test_thermal_paths.append(thermal_path)
                test_label_paths.append(label_path)
                test_count += 1
            print('Saved '+label_path)
        


for k, csv_file in enumerate(csv_files):
    img = None
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        curr_img_name = ''
        for i, line in enumerate(reader):
            if i == 0 or len(line) < 6:
                continue
            img_name = line[0].split('.')[0]
            polygon = json.loads(line[5])
            x_pos, y_pos = polygon['all_points_x'], polygon['all_points_y']
            points = [(x_pos[i], y_pos[i]) for i in range(len(x_pos))]
            label = json.loads(line[6])['Objects']
            if img_name != curr_img_name:
                if img:
                    thermal_sum += np.mean(np.array(img_thermal))
                    crop_and_save(img_name, img, img_rgb, img_thermal)
                curr_img_name = img_name
                origin_count += 1
                img = Image.new('P', (img_width, img_height))
                img_rgb = Image.open(os.path.join(rgb_dir[k], img_name+'.JPG'))
                img_thermal = Image.open(os.path.join(thermal_dir[k], img_name+'.JPG'))
                draw = ImageDraw.Draw(img)
            draw.polygon((points), fill=int(label))
        crop_and_save(img_name, img, img_rgb, img_thermal)
        thermal_sum += np.mean(np.array(img_thermal))

df_train = pd.DataFrame({'rgb': rgb_paths, 'thermal': thermal_paths, 
                         'label': label_paths})
df_train.to_csv(output_dir+'/path_list.csv', index=False)
df_test = pd.DataFrame({'rgb': test_rgb_paths, 'thermal': test_thermal_paths, 
                        'label': test_label_paths})
df_test.to_csv(output_dir+'/test_list.csv', index=False)
print('Original image number: ', origin_count)
print('Train image number: ', train_count)
print('Test image number: ', test_count)
print('Thermal mean: ', thermal_sum/(origin_count))
