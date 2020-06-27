# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 22:49:58 2020

@author: KGU2BAN
"""

import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import tensorflow_datasets as tfds

#%%
ds = tf.data.Dataset.from_tensor_slices([1,2,4,6]).map(lambda x: x*x)
ds = ds.enumerate()

for i in ds:
    print('enumeratte: ', i[0].numpy(),'squares',  i[1].numpy())

gen = tfds.as_numpy(ds)
it = iter(gen)
print(next(it))

#%%

ds = tf.data.TextLineDataset(['text.txt', 'text2.txt'])
for i in ds:
    print(i.numpy())

#%%
files = tf.data.Dataset.list_files('*.txt')
ds = tf.data.TextLineDataset(files)
for i in files:
    print(i.numpy())
for i in ds:
    print(i.numpy())

#%%
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.enumerate(start=5)
for element in dataset: 
  print(element)
it = iter(dataset)
next(it)

#%%
def transform_dataset(ds):
    return ds.map(lambda x: x**3)
    
ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
ds = ds.apply(transform_dataset)

for i in ds:
    print(i.numpy())

#%%
dataset = tf.data.Dataset.from_tensor_slices({'a': ([1, 2], [3, 4]), 
                                              'b': [5, 6]})
j = 0
for i in dataset:
    if j == 0:
        print(i)
    j = 1

#%%
ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8])
ds = ds.batch(3, True)

for i in ds:
    print(i.numpy())

#%%
ds = tf.data.Dataset.from_tensor_slices(list(range(20))).batch(3)
ds.cache('caches')
for i in ds:
    print(i.numpy())

#%%

ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
ds_temp = tf.data.Dataset.from_tensor_slices([5, 6, 7])
ds = ds.concatenate(ds_temp).filter(lambda x: x>3)

for i in ds:
    print(i.numpy())

#%%
import itertools

def iteration():
    for i in range(100):
        yield i

it = iteration()
#it = itertools.cycle(it)

#for i in range(1000):
#    print(next(it))

ds = tf.data.Dataset.from_generator(iteration, 'float32')
for i in ds:
    print(i.numpy())

    
#def data_generator_func(): #some arbit generator function
#    yield ((story, question), answer) # ([ndarray, ndarray], ndarray)
#ds = tf.data.Dataset.from_generator(data_generator_func,
#                                              ((tf.int64, tf.int64), tf.int64),
#                                              ((tf.TensorShape([12,6]), tf.TensorShape([6])), tf.TensorShape([1]))  )

#%%
ds = tf.data.Dataset.from_tensor_slices(list(range(10)))
ds = ds.window(4, 1, 1)

for i in ds:
    print([j.numpy() for j in list(i)])

#%%
ds3 = tf.random.uniform((4, 10))
print(ds3[0])

#%%
dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

for i in dataset2:
    a = i

#%%
a = iter(np.ones((5, 10)))
next(a)

#%%
dataset = tf.data.Dataset.range(8) 
dataset = dataset.batch(3) 
[i.numpy() for i in dataset]
#%%

a = iter(([1, 2], [3, 4]))
next(a)

#%%

mylist = [1, 2, 3]
mygen = itertools.cycle(mylist)

a = iter([1, 2, 3, 4])
#%%
dataset = tf.data.Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
for i in dataset:
    print(i.numpy())

dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

for i in dataset:
    print(i.numpy())

nested = ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
window_dataset = tf.data.Dataset.from_tensor_slices(nested).window(2)
for ds in window_dataset:
    pass
