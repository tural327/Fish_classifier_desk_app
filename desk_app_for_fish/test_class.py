from keras.models import load_model
import tensorflow as tf
import numpy as np
model = load_model("img_class.h5")

def find_class(picture):
    tensor = tf.keras.preprocessing.image.load_img(
        picture, grayscale=False, color_mode='rgb', target_size=(128, 128),
        interpolation='nearest'
        )
    input_arr = tf.keras.preprocessing.image.img_to_array(tensor)

    tesnor_for_pred = tf.reshape(input_arr, [1, 128, 128, 3])

    res = model.predict(tesnor_for_pred)

    result = np.argmax(res)
    image_class = ["Black Sea Sprat","Gilt-Head Bream","Hourse Mackerel","Red Mullet","Red Sea Bream","Sea Bass","Shrimp","Striped Red Mullet","Trout"]

    return image_class[result]