import glob, numpy as np, matplotlib.pyplot as plt, cv2.cv2 as cv2, seaborn as sns
import keras.callbacks as cb
from keras.applications import ResNet50, DenseNet121, NASNetMobile, Xception
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from BEASF import BEASF


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
    print('number of covid chest xrays:', len(covid_images))

    normal_labels = [0 for _ in range(len(normal_images))]
    covid_labels = [1 for _ in range(len(covid_images))]

    X = np.concatenate((covid_images, normal_images))
    y = np.array(covid_labels + normal_labels)

    X = np.array([cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_CUBIC) for image in X])
    X = np.array([np.expand_dims(a=image, axis=-1) for image in X])
    X = X.astype(dtype=np.uint8)

    # apply image enhancements and concat with the original image
    X_beasf = np.array([BEASF(image=image, gamma=1.5) for image in X])
    X_clahe = np.array([cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image) for image in X])
    X_clahe = np.array([np.expand_dims(a=image, axis=-1) for image in X_clahe])
    X = np.concatenate((X, X_beasf, X_clahe), axis=-1)

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


"""image data preparation"""
X_train, X_test, y_train, y_test = data_preparation(path='./chest_xray_images/', image_size=(96, 96))


"""data augmentation using keras"""
augmenter = ImageDataGenerator(rotation_range=90, horizontal_flip=True, vertical_flip=True, rescale=None)


"""model callbacks"""
checkpoint = cb.ModelCheckpoint(filepath='./checkpoints/densenet121/eps={epoch:03d}_valLoss={val_loss:.4f}.hdf5',
                                monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
cb_list = [checkpoint]


"""create the classifier from pretrained models"""
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(96, 96, 3))
base_model_out = base_model.output
# flatten = Flatten(name='flatten_layer')(base_model_out)
pooling = GlobalAveragePooling2D(name='pooling_layer')(base_model_out)
fc1 = Dense(units=10, activation='sigmoid', name='fc1_layer')(pooling)
drop1 = Dropout(rate=0.2, name='dropout1_layer')(fc1)
fc2 = Dense(units=20, activation='sigmoid', name='fc2_layer')(drop1)
drop2 = Dropout(rate=0.2, name='dropout2_layer')(fc2)
out = Dense(units=1, activation='sigmoid', name='output_layer')(drop2)
classifier = Model(inputs=base_model.input, outputs=out)
for layer in classifier.layers[:-6]:
    layer.trainable = False
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.summary()
print('number of pretrained network layers:', len(classifier.layers))


"""fine-tuning"""
fine_tuning = classifier.fit(augmenter.flow(x=X_train, y=y_train, batch_size=32), callbacks=cb_list, epochs=10,
                             verbose=1, validation_data=(X_test, y_test))

fig = plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(fine_tuning.history['loss'], color='r', label='training_loss')
plt.plot(fine_tuning.history['val_loss'], color='g', label='validation_loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(fine_tuning.history['accuracy'], color='r', label='training_accuracy')
plt.plot(fine_tuning.history['val_accuracy'], color='g', label='validation_accuracy')
plt.legend()
plt.show()
# fig.savefig('./checkpoints/densenet121/fine_tuning.png')


"""save the model to a json file"""
# model_json = classifier.to_json()
# with open("./checkpoints/densenet121/densenet121.json", "w") as json_file:
#     json_file.write(model_json)


"""classification reports"""
y_pred = classifier.predict(X_test)
print('number of test-set images:', len(y_test))
print(y_test)
y_pred = np.round(np.reshape(a=y_pred, newshape=(1, -1)), decimals=2)[0]
print(y_pred)
y_pred_rnd = np.round(np.reshape(a=y_pred, newshape=(1, -1)))[0]
cm = confusion_matrix(y_true=y_test, y_pred=y_pred_rnd)
print('confusion matrix:')
print(cm)
print('test-set accuracy:', (cm[0][0] + cm[1][1])/np.sum(cm))

print('classification report:')
print(classification_report(y_true=y_test, y_pred=y_pred_rnd,
                            target_names=['normal', 'covid']))

fig1 = plt.figure()
sns.heatmap(data=cm, cmap='Blues', annot=True, annot_kws={'size': 14}, fmt='d',
            vmin=0, vmax=len(y_test)/2.)
plt.title('annotated heatmap for confusion matrix')
plt.show()
# fig1.savefig('./checkpoints/densenet121/cm_heatmap.png')


fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred, pos_label=None)
roc_auc = auc(x=fpr, y=tpr)
fig2 = plt.figure()
plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# fig2.savefig('./checkpoints/densenet121/roc.png')
