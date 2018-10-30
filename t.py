# coding:utf-8
from __future__ import print_function
import tensorflow as tf

batch_size = 6
FRAMES_PER_SAMPLE = 100
NEFF = 129
EMBEDDING_D = 40

embedding_v = tf.get_variable('embedding_v', shape=(batch_size, FRAMES_PER_SAMPLE*NEFF, EMBEDDING_D))
Y_v = tf.get_variable('Y_v', shape=(batch_size, FRAMES_PER_SAMPLE*NEFF, 2))

VVT = tf.matmul(embedding_v, tf.transpose(embedding_v, [0,2,1])) # (batch_size, FRAMES_PER_SAMPLE*NEFF, FRAMES_PER_SAMPLE*NEFF)
YYT = tf.matmul(Y_v, tf.transpose(Y_v, [0,2,1])) # (batch_size, FRAMES_PER_SAMPLE*NEFF, FRAMES_PER_SAMPLE*NEFF)

VTV = tf.matmul()
