import os, glob, numpy as np, shutil, matplotlib.pyplot as plt, cv2
import tensorflow as tf, keras
import keras.callbacks as cb

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.layers import Input, Conv2D, Dense, Add, Flatten, BatchNormalization, Dropout
# from keras.layers import GlobalAveragePooling2D, MaxPooling2D, Concatenate, Activation
from keras import Model, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def data_preparation(path):
    normal_images = list()
    for img_name in glob.glob(pathname=path + '/normal/*'):
        img = load_img(path=img_name, color_mode='grayscale')
        img = img_to_array(img=img, data_format='channels_last')
        normal_images.append(img)

    normal_images = np.array(normal_images)
    print('number of normal chest xrays:', len(normal_images))

    covid_images = list()
    for img_name in glob.glob(pathname=path + '/covid19/*'):
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
    rnd_idx = np.random.choice(a=len(X), size=None)
    plt.imshow(X=X[rnd_idx].squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(label='a random image from the dataset')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)
    return X_train, X_test, y_train, y_test


def get_last_weights(folder):
    """
    find last saved weights file and its epoch number
    :param folder: string
    :return: int, string
    """
    num_epochs = list()
    for weights_file in glob.glob(folder + '/**.hdf5'):
        num_epoch = int(weights_file[weights_file.find('=')+1:weights_file.rfind('_')])
        num_epochs.append((num_epoch, weights_file))

    last_epoch = max(num_epochs)[0]
    print('last saved epoch:', last_epoch)
    last_file = max(num_epochs)[1]
    print('last saved file:', last_file)
    return last_epoch, last_file


def delete_other_weights(folder, last_file):
    """
    delete all weights files saved before, except the last one which is the best
    :param folder: string
    :param last_file: string
    :return None
    """

    for weights_file in os.listdir(folder):
        if weights_file.endswith('.hdf5'):
            file_path = os.path.join(folder, weights_file)
            if file_path != last_file:
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('failed to delete %s. because of: %s' % (file_path, e))
            else:
                pass
    print('deleted all weights files saved before, except the last one.')


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


print('TensorFlow version is:', tf.__version__)
print('Keras version is:', keras.__version__)


"""image data preparation"""
X_train, X_test, y_train, y_test = data_preparation(path='./chest_xray_images/')


"""data augmentation using keras"""
augmenter = ImageDataGenerator(rotation_range=90, horizontal_flip=True, vertical_flip=True, rescale=None)
test_augmenter = ImageDataGenerator(rescale=None)


"""model callbacks"""
checkpoint = cb.ModelCheckpoint(filepath='./checkpoints/base_model/v1.1/eps={epoch:03d}_valAcc={val_accuracy:.4f}.hdf5',
                                monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
cb_list = [checkpoint]


"""create the classifier model"""
classifier = build_model(input_shape=X_train[0].shape)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(classifier.summary())
print('number of network layers:', len(classifier.layers))


"""model training and learning curves"""
training = classifier.fit(augmenter.flow(x=X_train, y=y_train, batch_size=32), callbacks=cb_list, epochs=100,
                          verbose=1, validation_data=test_augmenter.flow(x=X_test, y=y_test))

fig = plt.figure()
plt.plot(training.history['loss'], color='r', label='training_loss')
plt.plot(training.history['val_loss'], color='g', label='validation_loss')
plt.legend()
plt.show()
fig.savefig('./checkpoints/base_model/v1.1/training_history.png')


"""best results on the test-set"""
weights_folder = './checkpoints/base_model/v1.1'
_, best_weights = get_last_weights(weights_folder)
acc = float(best_weights[best_weights.rfind('=')+1:best_weights.rfind('.')])
print('best validation accuracy:', acc)
classifier.load_weights(best_weights)
delete_other_weights(folder=weights_folder, last_file=best_weights)

y_pred = classifier.predict_generator(generator=test_augmenter.flow(x=X_test, batch_size=1, shuffle=False), steps=len(X_test))
print('number of test-set images:', len(y_test))
print(y_test)
y_pred = np.round(np.reshape(a=y_pred, newshape=(1, -1)), decimals=2)[0]
print(y_pred)
y_pred_rnd = np.round(np.reshape(a=y_pred, newshape=(1, -1)))[0]
cm = confusion_matrix(y_true=y_test, y_pred=y_pred_rnd)
print(cm)
print('accuracy:', (cm[0][0] + cm[1][1])/np.sum(cm))


"""save the model to a json file"""
model_json = classifier.to_json()
with open("./checkpoints/base_model/v1.1/base_model.json", "w") as json_file:
    json_file.write(model_json)

