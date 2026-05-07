
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def make_datasets(x_train, y_train, x_val, y_val, batch_size):

    H_l, W_l = 7, 7

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, (y_train, np.zeros((len(x_train), H_l, W_l, 1), dtype=np.float32)))
    )

    val_ds = tf.data.Dataset.from_tensor_slices(
        (x_val, (y_val, np.zeros((len(x_val), H_l, W_l, 1), dtype=np.float32)))
    )

    aug = tf.keras.Sequential([
        layers.RandomRotation(0.055),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1,0.1),
        layers.RandomFlip("horizontal")
    ])

    preprocess = tf.keras.applications.efficientnet.preprocess_input

    train_ds = train_ds.map(lambda x,y: (aug(x, training=True), y))
    train_ds = train_ds.map(lambda x,y: (preprocess(x), y))
    val_ds   = val_ds.map(lambda x,y: (preprocess(x), y))

    train_ds = train_ds.shuffle(512).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

