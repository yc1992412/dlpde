# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:38:28 2020
@author: Cheng Yuan
This code is for learning tf2 
"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import yc_tools as support

#%% making data

pi = np.pi

x = np.arange(0,1,0.01)
y = np.arange(0,1,0.01)

X,Y = np.meshgrid(x,y)
Z = np.cos(pi*X)*np.cos(pi*Y)

fig, ax = plt.subplots()
ax.pcolor(X,Y,Z,cmap='rainbow')
# plt.show()
plt.savefig('exact_solution.png')



#%% model

# method1: keras with self-defined train loop

inner = keras.Input(shape=(2,))
boundary = keras.Input(shape=(2,))
f = keras.Input(shape=(1,))

H1 = layers.Dense(50, activation='relu')
H2 = layers.Dense(100, activation='relu')
H3 = layers.Dense(100, activation='relu')
H4 = layers.Dense(50, activation='relu')
H5 = layers.Dense(1, activation='linear', name = 'u')

with tf.GradientTape() as tape:
    tape.watch(inner)
    h1 = H1(inner)
    h2 = H2(h1)
    h3 = H3(h2)
    h4 = H4(h3)
    u = H5(h4)

du = tape.gradient(u,inner)
ux = du[:,:1]
uy = du[:,1:]

energy = (ux**2+uy**2)/2 + u*u/2-f*u
integral = (K.mean(u))

model = keras.Model(inputs = [inner, f], outputs = [energy,integral])
model.summary()

batch_size = 10000
history = []
steps = 1000

optimizer = keras.optimizers.Adam(1e-4)

def my_loss(y_pred):
    loss_in = K.mean(y_pred[0])
    loss_bd = K.abs(y_pred[1])
    loss = K.mean(y_pred[0]) + 0
    return loss, loss_in, 0

@tf.function
def train_step(batch_data):
    with tf.GradientTape() as tape:
        [energy,integral] = model(batch_data, training=True)
        loss,loss_in,loss_bd = my_loss([energy,integral])    
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss, loss_in, loss_bd

for step in range(steps):
    [bd_data, in_data, f_data] = support.sampling(batch_size)
    batch_data = [in_data, f_data]
    loss,loss_in,loss_bd = train_step(batch_data)    

    if step%50 ==0:
        print('step = ', step,
              'loss_in = %.4f' %(float(loss_in)))
        history.append([step, float(loss), float(loss_bd)])