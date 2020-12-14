import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import layers

def preprocess(ds, shape, shuffle=False, augment=False):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    if augment:
        ds = ds.map(lambda x, y: (crop_hue_brightness(x, shape), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (flip_rotation(x), y), num_parallel_calls=AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(1000)
    
    ds = ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)

    #return ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def crop_hue_brightness(image, shape):
    image = tf.image.random_crop(image, size=shape)
    image = tf.image.random_hue(image, max_delta=0.25)
    image = tf.image.random_brightness(image, max_delta=0.25)
    return image

flip_rotation = ks.models.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])

normalize = layers.experimental.preprocessing.Rescaling(1./255)