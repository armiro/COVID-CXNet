# test file
from keras.preprocessing.image import load_img, img_to_array
import cv2.cv2 as cv2, numpy as np, matplotlib.pyplot as plt
from skimage.filters import rank
from skimage.morphology import disk
from BEASF import BEASF

IMG_PATH = ['./chest_xray_images/covid19/188.jpg', '1']
IMG_SHAPE = (500, 500, 1)

test_img = load_img(path=IMG_PATH[0], color_mode='grayscale')
test_img = img_to_array(img=test_img, data_format='channels_last')
test_img = cv2.resize(test_img, dsize=IMG_SHAPE[:2], interpolation=cv2.INTER_NEAREST)
test_img = test_img.astype(np.uint8)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
test_img_clahe = clahe.apply(test_img)
# test_img_ahe = rank.equalize(image=test_img, selem=disk(150))
# test_img_equ = cv2.equalizeHist(test_img)
#
# fig = plt.figure()
# plt.subplot(2, 2, 1)
# plt.imshow(test_img, cmap='gray')
# plt.axis('off')
# plt.title('org')
# plt.subplot(2, 2, 2)
# plt.imshow(test_img_equ, cmap='gray')
# plt.axis('off')
# plt.title('HE')
# plt.subplot(2, 2, 3)
# plt.imshow(test_img_ahe, cmap='gray')
# plt.axis('off')
# plt.title('AHE')
# plt.subplot(2, 2, 4)
# plt.imshow(test_img_clahe, cmap='gray')
# plt.axis('off')
# plt.title('CLAHE')
# plt.show()
# # fig.savefig(fname='./hist_equal14.png')

fig = plt.figure()
for i in range(0, 6):
    plt.subplot(2, 3, i+1)
    if i == 0:
        plt.imshow(X=test_img, cmap='gray')
        plt.title('org img')
    elif i == 5:
        plt.imshow(X=test_img_clahe, cmap='gray')
        plt.title('CLAHE')
    else:
        plt.imshow(X=BEASF(image=test_img, gamma=(i/2)), cmap='gray')
        plt.title('BEASF (gamma=%.1f)' % (i / 2))
    plt.axis('off')
plt.show()
# fig.savefig(fname='./beasf6.png')
