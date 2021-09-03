#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 00:19:05 2021

@author: tural
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

img_h = 128
img_w = 128
img_c = 3


file_loc = "Fish_Dataset/Fish_Dataset/training_full/"


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  file_loc,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_h, img_w),
  batch_size=32)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  file_loc,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_h, img_w),
  batch_size=32)




# Augmentation layer help us to make additional training..
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_h, 
                                                              img_w,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)


model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(9)
])
  
  
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])




epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

model.save("img_class.h5")


plt.figure(figsize=(8,5))
plt.plot(history.history['loss'],color='green',linewidth=3,label="Loss")
plt.plot(history.history['val_loss'],color='green',linewidth=1,linestyle="--",label="Val Loss")
plt.title("Model Loss",fontsize=18)
plt.xlabel("Epochs",fontsize=18)
plt.ylabel("Loss",fontsize=18)
plt.rcParams.update({'font.size': 20})
plt.legend(loc=3, prop={'size': 20})


plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'],color='green',linewidth=3,label="Accurancy")
plt.plot(history.history['val_accuracy'],color='green',linewidth=1,linestyle="--",label="Val Accurancy")
plt.title("Model Accurancy",fontsize=18)
plt.xlabel("Epochs",fontsize=18)
plt.ylabel("Accurancy",fontsize=18)
plt.rcParams.update({'font.size': 20})
plt.legend(loc=4, prop={'size': 20})