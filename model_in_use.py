# from main import build_model, get_last_weights
from keras.layers import Input, Conv2D, Dense, Flatten
from keras import Model
from keras.preprocessing.image import load_img, img_to_array
from vis.visualization import visualize_cam, overlay
import numpy as np, glob, cv2, matplotlib.pyplot as plt, matplotlib.cm as cm


def get_weights(folder):
    """
    find last saved weights file and its epoch number
    :param folder: string
    :return: string
    """
    num_epochs = list()
    for weights_file in glob.glob(folder + '/**.hdf5'):
        num_epoch = int(weights_file[weights_file.find('=')+1:weights_file.find('_')])
        num_epochs.append((num_epoch, weights_file))

    last_file = max(num_epochs)[1]
    print('last saved file:', last_file)
    return last_file


def build_model(input_shape):
    a0 = Input(shape=input_shape, name='input_layer')
    a1 = Conv2D(filters=32, kernel_size=5, strides=2, padding='valid', activation='relu', name='conv_layer1')(a0)
    a2 = Conv2D(filters=32, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv_layer2')(a1)
    a3 = Conv2D(filters=32, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv_layer3')(a2)
    a4 = Conv2D(filters=32, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv_layer4')(a3)
    a5 = Conv2D(filters=32, kernel_size=3, strides=2, padding='valid', activation='relu', name='conv_layer5')(a4)
    a6 = Flatten(name='flatten_layer')(a5)
    a7 = Dense(units=10, activation='relu', name='fc_layer1')(a6)
    a8 = Dense(units=10, activation='relu', name='fc_layer2')(a7)
    a9 = Dense(units=1, activation='sigmoid', name='output_layer')(a8)
    return Model(inputs=a0, outputs=a9, name='binary_classifier')


IMG_PATH = ['./chest_xray_images/covid19/016.jpeg', 1]

test_img = load_img(path=IMG_PATH[0], color_mode='grayscale')
test_img = img_to_array(img=test_img, data_format='channels_last')
test_img = cv2.resize(test_img, dsize=(400, 400), interpolation=cv2.INTER_NEAREST)
test_img = test_img / 255.
test_img = np.expand_dims(test_img, axis=-1)
print('external image(s) shape:', test_img.shape)

predictor = build_model(input_shape=(400, 400, 1))
predictor.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
predictor.load_weights(get_weights('./checkpoints'))

grads = visualize_cam(predictor, layer_idx=-1, filter_indices=None, seed_input=test_img, backprop_modifier=None,
                      grad_modifier=None, penultimate_layer_idx=None)
jet_heatmap = np.uint8(cm.jet(grads)[:, :, :, 0] * 255)

# test_img_3ch = np.concatenate((test_img, test_img, test_img), axis=-1)
# plt.imshow(overlay(jet_heatmap, test_img_3ch, alpha=0.4))
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(jet_heatmap)
plt.axis('off')
plt.title('pred=%.2f' % predictor.predict(np.expand_dims(test_img, axis=0)))
plt.subplot(1, 2, 2)
plt.imshow(test_img.squeeze(), cmap='gray')
plt.axis('off')
plt.title('label=%d' % IMG_PATH[1])
plt.show()
# fig.savefig(fname='./?.png')

