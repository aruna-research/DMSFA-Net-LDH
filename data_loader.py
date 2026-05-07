
import os
import cv2
import numpy as np
import tensorflow as tf

def load_dataset(data_dir, input_shape):
    images, labels = [], []

    class_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    class_map = {name: i for i, name in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)

        for f in os.listdir(cls_dir):
            img = cv2.imread(os.path.join(cls_dir, f))
            if img is None:
                continue

            img = cv2.resize(img, input_shape[:2]) / 255.0
            images.append(img)
            labels.append(class_map[cls])

    images = np.array(images, dtype="float32")
    labels = tf.keras.utils.to_categorical(labels, len(class_names))

    return images, labels, class_names

