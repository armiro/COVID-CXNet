# from main import build_model, get_last_weights
from keras.models import model_from_json, load_model
from keras.preprocessing.image import load_img, img_to_array
from vis.visualization import visualize_cam
import numpy as np, glob, cv2, matplotlib.pyplot as plt, matplotlib.cm as cm


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


IMG_PATH = ['./chest_xray_images/covid19/020.jpg', 1]

test_img = load_img(path=IMG_PATH[0], color_mode='grayscale')
test_img = img_to_array(img=test_img, data_format='channels_last')
test_img = cv2.resize(test_img, dsize=(400, 400), interpolation=cv2.INTER_NEAREST)
test_img = test_img / 255.
test_img = np.expand_dims(test_img, axis=-1)
print('external image(s) shape:', test_img.shape)

# load model as a json file and load weights from .hdf5 file
# json_file = open(file='./checkpoints/base_model/base_model.json', mode='r')
# model_json = json_file.read()
# json_file.close()
# predictor = model_from_json(model_json)
# predictor.load_weights(get_weights('./checkpoints/base_model'))

# load_model if the model is saved as a single .h5 file
predictor = load_model('./checkpoints/CheXNet/CheXNet_Keras_0.3.0.h5')
predictor.summary()

# grads = visualize_cam(predictor, layer_idx=-1, filter_indices=None, seed_input=test_img, backprop_modifier=None,
#                       grad_modifier=None, penultimate_layer_idx=None)
# jet_heatmap = np.uint8(cm.jet(grads)[:, :, :, 0] * 255)
#
# # test_img_3ch = np.concatenate((test_img, test_img, test_img), axis=-1)
# # out = cv2.addWeighted(src1=np.uint8(test_img_3ch * 255), alpha=0.8, src2=jet_heatmap, beta=0.2, gamma=0)
#
# fig = plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(jet_heatmap)
# plt.axis('off')
# plt.title('pred=%.2f' % predictor.predict(np.expand_dims(test_img, axis=0)))
# plt.subplot(1, 2, 2)
# plt.imshow(test_img.squeeze(), cmap='gray')
# plt.axis('off')
# plt.title('label=%d' % IMG_PATH[1])
# plt.show()
# # fig.savefig(fname='./?.png')

