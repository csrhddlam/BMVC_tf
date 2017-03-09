import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import os
import scipy.misc


def get_data(sp_list, total_back_number, patch_size, channel, thres):
    cover_size = 1
    feature_length = channel * patch_size * patch_size

    train_names = []
    train_instances = []
    train_anno = []
    with open("./list/car_train.txt") as f:
        for line in f:
            if line == '\n':
                continue
            split = line.split(' ')
            train_names.append(split[0])
            train_instances.append(int(split[1][0]) - 1)
            train_anno.append(sio.loadmat('./gt/' + train_names[-1] + '.mat')['anno'])

    import h5py

    f = h5py.File('./features/res_info_train.mat')
    res_info = f['res_info']

    total_data_number = 0

    for i in range(len(train_names)):  # image
        # print('image', i)
        semantic_parts = train_anno[i][train_instances[i]][1]
        for j in sp_list:  # semantic part
            instances = semantic_parts[j][0]
            total_data_number += len(instances)

    total_data = np.zeros((feature_length, total_data_number), dtype=np.float32)
    total_label = np.zeros((1, total_data_number), dtype=np.int32)
    total_list = [[] for i in range(39)]
    total_index = 0

    back_data = np.zeros((feature_length, total_back_number), dtype=np.float32)
    back_index = 0
    # fig = plt.figure()

    for i in range(len(train_names)):  # image
        # print('image', i)
        group = f[res_info[i][0]]
        dist = np.array(group['layer_feature_dist'])
        feat = np.array(group['layer_feature_ori'])
        img = np.array(group['img'])
        # padded_img =np.lib.pad(img, ((0,), (100,), (100,)), 'constant', constant_values=(0,))
        [_, width, height] = dist.shape
        if thres < 0:
            binary = (feat - 55) / 110
        elif thres == 0:
            binary = (dist - 1) / 2
        else:
            binary = np.float32(dist < thres)
        binary = np.float32(binary)
        binary_mask = np.zeros((1, width, height))
        padded = np.lib.pad(binary, ((0,), (patch_size//2,), (patch_size//2,)), 'constant', constant_values=(0,))
        semantic_parts = train_anno[i][train_instances[i]][1]
        for j in sp_list:  # semantic part
            dir_name = 'figures/sp'+str(j+1)
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            instances = semantic_parts[j][0]
            for k in range(len(instances)):
                instance = instances[k]
                xy = (instance[0:2] + instance[2:4]) / 2
                # cropped_img = padded_img[:, xy[0] - 50 + 100: xy[0] + 50 + 100, xy[1] - 50 + 100: xy[1] + 50 + 100]
                # permuted = np.transpose(cropped_img)
                # img_name = 'figures/sp'+str(j+1)+'/'+str(i)+'_'+str(k)+'.jpg'
                # scipy.misc.imsave(img_name, permuted)
                pool_xy = np.round((xy - 8.5) / 16)
                if pool_xy[0] < 0:
                    print(xy[0])
                    pool_xy[0] = 0
                if pool_xy[0] > width - 1:
                    # print(xy[0], pool_xy[0], width - 1, i, j, k, 0)
                    pool_xy[0] = width - 1
                if pool_xy[1] < 0:
                    print(xy[1])
                    pool_xy[1] = 0
                if pool_xy[1] > height - 1:
                    # print(xy[1], pool_xy[1], height - 1, i, j, k, 1)
                    pool_xy[1] = height - 1

                binary_mask[0, max(pool_xy[0] - cover_size // 2, 0):min(pool_xy[0] + 1 + cover_size // 2, width),
                               max(pool_xy[1] - cover_size // 2, 0):min(pool_xy[1] + 1 + cover_size // 2, height)] = 1

                total_data[:, total_index] = np.reshape(
                    padded[:, pool_xy[0]:pool_xy[0] + patch_size, pool_xy[1]:pool_xy[1] + patch_size],
                    (feature_length, ))
                total_label[:, total_index] = j + 1
                total_list[j].append(total_index)
                total_index += 1
        if back_index >= total_back_number:
            continue
        for j in range(width):
            for k in range(height):
                if binary_mask[0, j, k] == 0 and back_index < total_back_number and random.random() < 0.015:
                    print(i, j, k)
                    back_data[:, back_index] = np.reshape(
                        padded[:, j:j + patch_size, k:k + patch_size],
                        (feature_length, ))
                    back_index += 1

    return [total_data, total_label, total_list, back_data]