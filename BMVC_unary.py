import numpy as np
from data_loader import get_data
import datetime
import scipy.io as sio
import pickle

visible = 8624
hidden = 1
patch_size = 7
total_sp = 40
total_vc = 176
back_samples = 10000
[data, label, index_list, back_data] = get_data(range(39), back_samples, patch_size, total_vc, 0.65)

mat21 = pickle.load(open("mat21_U_clusters_hy.pickle", "rb"))
mat22 = pickle.load(open("mat22_U_clusters_hy.pickle", "rb"))
mat2122 = pickle.load(open("mat21_22_U_clusters_hy.pickle", "rb"))

counting = np.zeros((total_sp, 176, patch_size, patch_size))
counting_number = np.zeros((total_sp, ))
'''-------------------------------------------------------------'''
for j in range(39):# semantic part
    temp_index_list = index_list[j]
    sum_data = np.sum(data[:, temp_index_list], axis=1)
    counting[j, :, :, :] += sum_data.reshape((total_vc, patch_size, patch_size))
    counting_number[j] += len(temp_index_list)

sum_data = np.sum(back_data, axis=1)
counting[39, :, :, :] = sum_data.reshape((total_vc, patch_size, patch_size))
counting_number[39] += back_samples

# for j in [40, 41, 42, 43]:
#     if j + 1 == 41:
#         temp_index_list = index_list[1 - 1] + index_list[23 - 1]
#     elif j + 1 == 42:
#         temp_index_list = index_list[2 - 1] + index_list[3 - 1]
#     elif j + 1 == 43:
#         temp_index_list = index_list[22 - 1] + index_list[23 - 1]
#     elif j + 1 == 44:
#         temp_index_list = index_list[38 - 1] + index_list[39 - 1]
#     sum_data = np.sum(data[:, temp_index_list], axis=1)
#     counting[j, :, :, :] += sum_data.reshape((total_vc, patch_size, patch_size))
#     counting_number[j] += len(temp_index_list)
#
# for j in [44]:
#     temp_index_list = [index_list[21][i] for i in mat21[j - 44]]
#     sum_data = np.sum(data[:, temp_index_list], axis=1)
#     counting[j, :, :, :] += sum_data.reshape((total_vc, patch_size, patch_size))
#     counting_number[j] += len(temp_index_list)
# for j in [45]:
#     temp_index_list = [index_list[22][i] for i in mat22[j - 45]]
#     sum_data = np.sum(data[:, temp_index_list], axis=1)
#     counting[j, :, :, :] += sum_data.reshape((total_vc, patch_size, patch_size))
#     counting_number[j] += len(temp_index_list)
# for j in [46, 47]:
#     index_21_22 = index_list[22 - 1] + index_list[23 - 1]
#     temp_index_list = [index_21_22[i] for i in mat2122[j - 46]]
#     sum_data = np.sum(data[:, temp_index_list], axis=1)
#     counting[j, :, :, :] += sum_data.reshape((total_vc, patch_size, patch_size))
#     counting_number[j] += len(temp_index_list)
'''-------------------------------------------------------------'''

prob = np.divide(counting, np.reshape(counting_number, (total_sp, 1, 1, 1))) + 0.0000001
weight = np.log(np.divide(prob, 1 - prob))
logZ = np.log(np.divide(1, 1 - prob))
# ZZ = np.log(np.exp(weight) + 1)
prior = np.log(counting_number)
sio.savemat('model.mat', {'weight': weight, 'prior': prior, 'logZ': logZ})