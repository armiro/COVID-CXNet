import os, glob, numpy as np, shutil, matplotlib.pyplot as plt, cv2
# import tensorflow as tf
import keras.callbacks as cb
from keras.applications import ResNet50, DenseNet121
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.layers import Dense, Dropout
# from keras.layers import GlobalAveragePooling2D, MaxPooling2D
from keras import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def data_preparation(path, image_size):
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

    X = np.array([cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_CUBIC) for image in X])
    X = np.array([np.expand_dims(a=image, axis=-1) for image in X])
    X = np.array([X[idx] / 255. for idx in range(len(X))])
    X = np.array(([np.concatenate((image, image, image), axis=-1) for image in X]))

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


"""image data preparation"""
X_train, X_test, y_train, y_test = data_preparation(path='./chest_xray_images/', image_size=(224, 224))


"""data augmentation using keras"""
augmenter = ImageDataGenerator(rotation_range=90, horizontal_flip=True, vertical_flip=True, rescale=None)
test_augmenter = ImageDataGenerator(rescale=None)


checkpoint = cb.ModelCheckpoint(filepath='./checkpoints/densenet121/eps={epoch:03d}_valAcc={val_accuracy:.4f}.hdf5',
                                monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
cb_list = [checkpoint]


"""create the classifier from pretrained models"""
base_model = DenseNet121(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
base_model_out = base_model.output
out = Dense(units=1, activation='sigmoid', name='output_layer')(base_model_out)
classifier = Model(inputs=base_model.input, outputs=out)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# classifier.summary()
print('number of pretrained network layers:', len(classifier.layers))


"""fine-tuning"""
fine_tuning = classifier.fit(augmenter.flow(x=X_train, y=y_train, batch_size=16), callbacks=cb_list, epochs=20,
                             verbose=1, validation_data=test_augmenter.flow(x=X_test, y=y_test))

fig = plt.figure()
plt.plot(fine_tuning.history['loss'], color='r', label='training_loss')
plt.plot(fine_tuning.history['val_loss'], color='g', label='validation_loss')
plt.legend()
plt.show()
fig.savefig('./checkpoints/densenet121/fine_tuning.png')


"""save the model to a json file"""
model_json = classifier.to_json()
with open("./checkpoints/densenet121/densenet121.json", "w") as json_file:
    json_file.write(model_json)


"""classification report on the test-set"""
y_pred = classifier.predict_generator(generator=test_augmenter.flow(x=X_test, batch_size=1, shuffle=False), steps=len(X_test))
print('number of test-set images:', len(y_test))
print(y_test)
y_pred = np.round(np.reshape(a=y_pred, newshape=(1, -1)), decimals=2)[0]
print(y_pred)
y_pred_rnd = np.round(np.reshape(a=y_pred, newshape=(1, -1)))[0]
cm = confusion_matrix(y_true=y_test, y_pred=y_pred_rnd)
print(cm)
print('accuracy:', (cm[0][0] + cm[1][1])/np.sum(cm))
