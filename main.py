import tensorflow as tf
import tensorflow.keras.backend as K
print('TensorFlow version is:', tf.__version__)

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, Dense, Add, Flatten, BatchNormalization, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D, Concatenate, Activation
from tensorflow.keras import Model, optimizers

import os, glob, numpy as np, time, shutil, pickle, matplotlib.pyplot as plt, cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# image data preparation
path = './chest_xray_images/'
normal_images = list()
for img_name in glob.glob(pathname=path+'normal/*'):
    img = load_img(path=img_name, color_mode='grayscale')
    img = img_to_array(img=img, data_format='channels_last')
    normal_images.append(img)

normal_images = np.array(normal_images)
print('number of normal chest xrays:', len(normal_images))

covid_images = list()
for img_name in glob.glob(pathname=path+'/covid19/*'):
    img = load_img(path=img_name, color_mode='grayscale')
    img = img_to_array(img=img, data_format='channels_last')
    covid_images.append(img)

covid_images = np.array(covid_images)
print('number of covid19 chest xrays:', len(covid_images))

normal_labels = [0 for _ in range(len(normal_images))]
covid_labels = [1 for _ in range(len(covid_images))]

X = np.concatenate((covid_images, normal_images))
y = np.array(covid_labels + normal_labels)

X = np.array([cv2.resize(image, dsize=(400, 400), interpolation=cv2.INTER_CUBIC) for image in X])
X = np.array([np.expand_dims(a=image, axis=-1) for image in X])
X = np.array([X[idx] / 255. for idx in range(len(X))])

print('number of total dataset images:', len(X))
print('number of total dataset labels:', len(y))
print('dataset shape:', X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=18)

print('show a random image from the dataset:')
rnd_idx = np.random.choice(a=len(X), size=None)
plt.imshow(X=X[rnd_idx].squeeze(), cmap='gray')
plt.show()
