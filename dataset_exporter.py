from tensorflow.keras.preprocessing.image import load_img, img_to_array
import glob, numpy as np, matplotlib.pyplot as plt, cv2.cv2 as cv2


def collect_images_from(path, exclude_early_stages=True, exclude_pediatrics=True):
    normal_images = list()
    for img_name in glob.glob(pathname=path + '/normal/*'):
        img = load_img(path=img_name, color_mode='grayscale')
        img = img_to_array(img=img, data_format='channels_last')
        normal_images.append(img)

    normal_images = np.array(normal_images, dtype=object)
    normal_images = np.random.choice(normal_images, size=3674, replace=False)
    print('num normal CXRs:', len(normal_images))

    cap_images = list()
    for img_name in glob.glob(pathname=path + '/cap/*'):
        img = load_img(path=img_name, color_mode='grayscale')
        img = img_to_array(img=img, data_format='channels_last')
        cap_images.append(img)

    cap_images = np.array(cap_images, dtype=object)
    cap_images = np.random.choice(cap_images, size=3500, replace=False)
    print('num community-acquired pneumonia CXRs:', len(cap_images))

    covid_images = list()
    num_early_stage_images = 0
    num_pediatric_images = 0
    for img_name in glob.glob(pathname=path + '/covid19/*'):
        early_stage = False
        pediatric = False
        img_num = img_name[img_name.rfind('\\') + 1:img_name.rfind('.')]
        if ('-' in img_num) & exclude_early_stages:
            num_early_stage_images += 1
            early_stage = True
        if ('p' in img_num) & exclude_pediatrics:
            num_pediatric_images += 1
            pediatric = True

        if early_stage | pediatric:
            continue
        else:
            img = load_img(path=img_name, color_mode='grayscale')
            img = img_to_array(img=img, data_format='channels_last')
            covid_images.append(img)

    print("num total early stage images:", num_early_stage_images)
    print("num total pediatric images:", num_pediatric_images)
    covid_images = np.array(covid_images, dtype=object)
    print('num collected covid CXRs:', len(covid_images))

    normal_labels = [0 for _ in range(len(normal_images))]
    cap_labels = [1 for _ in range(len(cap_images))]
    covid_labels = [2 for _ in range(len(covid_images))]

    X = np.concatenate((covid_images, cap_images, normal_images))
    y = np.array(covid_labels + cap_labels + normal_labels)
    return X, y


def resize_images_of(X):
    X = np.array([cv2.resize(image, dsize=(320, 320), interpolation=cv2.INTER_CUBIC) for image in X])
    X = np.array([np.expand_dims(a=image, axis=-1) for image in X])
    X = X.astype(dtype=np.uint8)
    return X


def show_random_image_from(X):
    rnd_idx = np.random.choice(a=len(X), size=None)
    plt.imshow(X=X[rnd_idx].squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(label='a random image from the dataset')
    plt.show()


def save_dataset(data, labels):
    print('number of total dataset images:', len(data))
    print('number of total dataset labels:', len(labels))
    print('dataset shape:', data.shape)
    print("export images as npy file? (y/n)")
    if input() == 'y':
        np.save('./cxr_samples.npy', arr=data)
        np.save('./cxr_labels.npy', arr=labels)
    else:
        print('dataset exportation aborted.')


def main():
    data_path = './chest_xray_images/'
    X, y = collect_images_from(path=data_path, exclude_early_stages=False, exclude_pediatrics=False)
    X = resize_images_of(X=X)
    show_random_image_from(X=X)
    save_dataset(data=X, labels=y)


if __name__ == '__main__':
    main()
