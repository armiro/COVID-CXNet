# from main import build_model, get_last_weights
from keras.models import model_from_json, load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np, glob, cv2.cv2 as cv2, matplotlib.pyplot as plt, copy
from skimage.segmentation import mark_boundaries
from visualization_tools import generate_explanation, generate_heatmap
from BEASF import BEASF
from keras.applications import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras import Model


def get_weights(folder):
    """
    find last saved weights file and its epoch number
    :param folder: string
    :return: string
    """
    num_epochs = list()
    for weights_file in glob.glob(folder + '/**.hdf5'):
        num_epoch = int(weights_file[weights_file.find('=')+1:weights_file.rfind('_')])
        num_epochs.append((num_epoch, weights_file))

    last_file = max(num_epochs)[1]
    print('last saved file:', last_file)
    return last_file


IMG_PATH = ['./chest_xray_images/covid19/178.jpeg', '1']
IMG_SHAPE = (300, 300, 3)

test_img = load_img(path=IMG_PATH[0], color_mode='grayscale')
test_img = img_to_array(img=test_img, data_format='channels_last')
test_img = cv2.resize(test_img, dsize=IMG_SHAPE[:2], interpolation=cv2.INTER_NEAREST)
test_img = np.expand_dims(test_img, axis=-1)
test_img = test_img.astype(np.uint8)
ref_img = copy.deepcopy(x=test_img)

if IMG_SHAPE[-1] == 3:
    test_img_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(test_img)
    test_img_clahe = np.expand_dims(a=test_img_clahe, axis=-1)
    test_img_beasf = BEASF(image=test_img, gamma=1.5)
    test_img = np.concatenate((test_img, test_img_beasf, test_img_clahe), axis=-1)
else:
    pass

test_img = test_img / 255.
print('external image(s) shape:', test_img.shape)

# load model as a json file and load weights from .hdf5 file
json_file = open(file='./checkpoints/base_model/v_free/base_model.json', mode='r')
model_json = json_file.read()
json_file.close()
predictor = model_from_json(model_json)
predictor.load_weights(get_weights('./checkpoints/base_model/v_free'))

# backbone = DenseNet121(include_top=False, weights=None, input_shape=(320, 320, 3))
# backbone_out = backbone.output
# gap = GlobalAveragePooling2D(name='pooling_layer')(backbone_out)
# output = Dense(units=14, activation='sigmoid', name='output_layer')(gap)
# predictor = Model(inputs=backbone.input, outputs=output)
# print(predictor.summary())
# predictor.load_weights('C:/Users/Arman/Desktop/Covid19-Detection/checkpoints/CheXNet/CheXNet_v0.3.0.h5')


# load_model if the model is saved as a single .h5 file
# predictor = load_model('./checkpoints/CheXNet/CheXNet_Keras_0.3.0.h5')
# predictor.summary()

jet_heatmap = generate_heatmap(model=predictor, input_image=test_img)
# temp_img = np.concatenate((ref_img, ref_img, ref_img), axis=-1)
# overlay_heatmap = cv2.addWeighted(src1=np.uint8(temp_img * 255), alpha=0.5, src2=jet_heatmap, beta=0.5, gamma=0)
# print(predictor.predict(np.expand_dims(test_img, axis=0)))

fig1 = plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(jet_heatmap)
plt.axis('off')
plt.title('pred=%.4f' % predictor.predict(np.expand_dims(test_img, axis=0)))
plt.subplot(1, 2, 2)
plt.imshow(ref_img.squeeze(), cmap='gray')
plt.axis('off')
plt.title('label=%s' % IMG_PATH[1])
plt.show()
# fig1.savefig(fname='./?.png')

temp, mask = generate_explanation(model=predictor, input_image=test_img)

fig2 = plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.axis('off')
plt.title('pred=%.4f' % predictor.predict(np.expand_dims(test_img, axis=0)))
plt.subplot(1, 2, 2)
plt.imshow(ref_img.squeeze(), cmap='gray')
plt.axis('off')
plt.title('label=%s' % IMG_PATH[1])
plt.show()
# fig1.savefig(fname='./?.png')

