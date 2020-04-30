from keras.preprocessing.image import load_img, img_to_array
import cv2.cv2 as cv2, numpy as np
from keras.applications import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras import Model


IMG_PATH = ['./chest_xray_images/normal/15268.jpg', '0']
IMG_SHAPE = (320, 320, 3)

test_img = load_img(path=IMG_PATH[0], color_mode='grayscale')
test_img = img_to_array(img=test_img, data_format='channels_last')
test_img = cv2.resize(test_img, dsize=IMG_SHAPE[:2], interpolation=cv2.INTER_NEAREST)
test_img = np.expand_dims(test_img, axis=-1)
test_img = test_img.astype(np.uint8)
test_img = test_img / 255.
test_img = np.concatenate((test_img, test_img, test_img), axis=-1)
print('external image(s) shape:', test_img.shape)

backbone = DenseNet121(include_top=False, weights=None, input_shape=(320, 320, 3))
backbone_out = backbone.output
gap = GlobalAveragePooling2D(name='pooling_layer')(backbone_out)
output = Dense(units=14, activation='sigmoid', name='output_layer')(gap)
predictor = Model(inputs=backbone.input, outputs=output)
print(predictor.summary())
predictor.load_weights('C:/Users/Arman/Desktop/Covid19-Detection/checkpoints/CheXNet/CheXNet_v0.3.0.h5')
print(predictor.predict(np.expand_dims(test_img, axis=0)))

