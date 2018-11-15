# File: local-variance-map.py
# Author: Zeng Ruizi, (Rey)

import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as pp
from operator import itemgetter
import timeit

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_k_first(k, f, s):
    return np.int((k-(f-s))/s) * (f-s) + k # proof by induction

def get_k_last(k, f, s):
    return np.int(k/s) * (f-s) + k # proof by induction

def get_num_steps(n, f, s):
    return np.floor((n-f)/s) + 1

def get_max_num_appearances(f, s):
    return np.ceil(f/s).astype(np.int)

"""
computes the indices that the current index k will be mapped to
when an array of element n is convolved with a filter of size f and with stride s
"""
def compute_expanded_indices(k, n, f, s):
    assert s > 0, "stride (s) must be greater than 0"
    assert f > 0, "window size (f) must be greater than 0"
    assert s < f, "stride (s) must be smaller than filter size (f), unsupported atm"
    assert f < n, "filter size (f) must be smaller than image dimension (n), unsupported atm"

    if s < f:
        k_last = np.int(k/s) * (f-s) + k # proof by induction
        array_of_indices_sets = np.full((get_max_num_appearances(f, s)), np.int(get_num_steps(n, f, s) * f), dtype=np.int)
        step_size = f - s
        for l in range(get_max_num_appearances(f, s)):
            idx = k_last - step_size*l
            if idx >= np.int(get_num_steps(n, f, s) * f):
                continue
            if idx < get_k_first(k, f, s):
                break
            array_of_indices_sets[len(array_of_indices_sets) - l - 1] = idx
            if idx < f:
                break

    return array_of_indices_sets

m = 608
n = 744
f = 3
s = 1

file_path = ""
im = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
im = im[:m, :n]

print(im.shape)

# x_val = np.arange(n*n)
x_val = im

x = tf.constant(x_val, tf.float32, (1, m, n, 1), 'x')

x_tiled = tf.tile(x, (1, 1, 1, get_max_num_appearances(f, s)**2))

x_tiled_shape = x_tiled.shape.as_list()

assert len(x_tiled_shape) == 4, "wrong shape"

indices_shape = np.array(x_tiled_shape)
indices_shape = np.append(indices_shape, len(x_tiled_shape))

indices = np.zeros(indices_shape, np.int)

j_indices = [compute_expanded_indices(j, x.shape[1].value, f, s) for j in range(x.shape[1])]
k_indices = [compute_expanded_indices(k, x.shape[2].value, f, s) for k in range(x.shape[2])]
for i in range(x.shape[0].value):
    for j in range(x.shape[1].value):
        curr_j_indices = j_indices[j]
        for k in range(x.shape[2].value):
            curr_k_indices = k_indices[k]
            total_length = len(j_indices[j]) * len(k_indices[k])
            for l in range(x.shape[3].value):
                new_indices = np.transpose([np.repeat(i, total_length), np.repeat(curr_j_indices, len(curr_k_indices)), np.tile(curr_k_indices, len(curr_j_indices)), np.repeat(l, total_length)]) # 4d cartesian product
                indices[i][j][k][l*total_length:(l+1)*total_length] = new_indices

x_expanded = tf.scatter_nd(indices, x_tiled, shape=tf.constant([1, np.int(get_num_steps(m, f, s) * f + 1), np.int(get_num_steps(n, f, s) * f + 1), 1], dtype=tf.int64), name='x_expanded')
x_expanded = x_expanded[:, :-1, :-1, :] # removing aggregated values at the end
x_expanded = tf.cast(x_expanded, tf.float32)

avg_filt = tf.fill((f, f, 1, 1), 1/f**2, name="avg_filt")

x_avg = tf.nn.conv2d(x_expanded, avg_filt, [1, f , f, 1], 'VALID')
x_avg_tiled = tf.tile(x_avg, tf.constant([1, 1, 1, f*f]))

x_avg_tiled_shape = x_avg_tiled.shape.as_list()
indices_shape = np.append(x_avg_tiled_shape, len(x_avg_tiled_shape))

indices = np.zeros(indices_shape, np.int)

for i in range(x_avg_tiled.shape[0].value):
    for j in range(x_avg_tiled.shape[1].value):
        for k in range(x_avg_tiled.shape[2].value):
            for l in range(x_avg_tiled.shape[3].value):
                new_j = np.int(l/f) + j*f
                new_k = l%f + k*f
                new_idx = [0, new_j, new_k, 0]
                indices[i][j][k][l] = new_idx

x_avg_tiled_expanded = tf.scatter_nd(indices, x_avg_tiled, shape=tf.constant(x_expanded.shape.as_list(), dtype=tf.int64), name='x_avg_tiled_expanded')

x_squared_mean_difference = tf.square(x_expanded-x_avg_tiled_expanded, name="sqrd_diff")

sum_filt = tf.fill((f, f, 1, 1), 1., name="sum_filt")

x_variance = tf.nn.conv2d(x_squared_mean_difference, sum_filt, [1, f , f, 1], 'VALID')

# x_variance_grad = tf.gradients(x_variance, [x])

x_variance_normlaized = (x_variance - tf.reduce_min(x_variance))/(tf.reduce_max(x_variance) - tf.reduce_min(x_variance))

with tf.Session() as sess:
    x_variance_out = sess.run(x_variance_normlaized)
    # x_variance_grad_out = sess.run(x_variance_grad)

with open("x_variance.txt", "wb") as f:
    np.savetxt(f, np.squeeze(x_variance_out*255), fmt='%.4f', delimiter=", ")

cv2.imwrite("x_variance_out.png", np.squeeze(x_variance_out*255))

