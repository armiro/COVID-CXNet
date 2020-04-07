import numpy as np
import copy


def subhist(image_pdf, minimum, maximum, normalize):
    minimum = np.int32(minimum)
    maximum = np.int32(maximum+1)
    hi = np.zeros(shape=image_pdf.shape)
    total = 0
    for idx in range(minimum, maximum):
        total += image_pdf[idx]
        hi[idx] = image_pdf[idx]
    if normalize:
        for idx in range(minimum, maximum):
            hi[idx] /= total
    return hi


def BEASF(image, gamma):
    """
    Computes the BEASF algorithm; built using Numpy package
    A python implementation of the original MATLAB code:
    The algorithm is introduced by XXX, in the research paper: XXX
    :param image: numpy.ndarray
    :param gamma: float [0, 1]
    :return: numpy.ndarray
    """
    m = int(np.mean(image, dtype=np.int32))
    h = np.histogram(image, bins=256)[0] / 2500
    h_lower = subhist(image_pdf=h, minimum=0, maximum=m, normalize=True)
    h_upper = subhist(image_pdf=h, minimum=m, maximum=255, normalize=True)

    # cdf_lower = norm.cdf(h_lower)
    cdf_lower = np.zeros(shape=h_lower.shape)
    cdf_lower[0] = h_lower[0]
    for idx in range(1, len(h_lower)):
        cdf_lower[idx] = cdf_lower[idx - 1] + h_lower[idx]

    # cdf_upper = norm.cdf(h_upper)
    cdf_upper = np.zeros(shape=h_upper.shape)
    cdf_upper[0] = h_upper[0]
    for idx in range(1, len(h_upper)):
        cdf_upper[idx] = cdf_upper[idx - 1] + h_upper[idx]

    half_low = 0
    for idx in range(0, int(m)):
        if cdf_lower[idx + 1] > 0.5:
            half_low = idx
            break
    half_up = 0
    for idx in range(int(m), 256):
        if cdf_upper[idx + 1] > 0.5:
            half_up = idx
            break

    # sigmoid CDF creation
    tones_low = np.arange(0, m, 1)
    x_low = 5.0 * (tones_low - half_low) / m  # shift & scale intensity x to place sigmoid [-2.5, 2.5
    # lower sigmoid
    s_low = 1 / (1 + np.exp(-gamma * x_low))

    tones_up = np.arange(m, 255, 1)
    x_up = 5.0 * (tones_up - half_up) / (255 - m)  # shift & scale intensity x to place sigmoid [-2.5, 2.5
    # upper sigmoid
    s_up = 1 / (1 + np.exp(-gamma * x_up))

    mapping_vector = np.zeros(shape=(256,))
    for idx in range(0, m):
        mapping_vector[idx] = np.int32(m * s_low[idx])

    minimum = mapping_vector[0]
    maximum = mapping_vector[m]
    for idx in range(0, m):
        mapping_vector[idx] = np.int32((m / (maximum - minimum)) * (mapping_vector[idx] - minimum))
    for idx in range(m + 1, 256):
        mapping_vector[idx] = np.int32(m + (255 - m) * s_up[idx - m - 1])

    minimum = mapping_vector[m + 1]
    maximum = mapping_vector[255]
    for idx in range(m + 1, 256):
        mapping_vector[idx] = (255 - m) * (mapping_vector[idx] - minimum) / (maximum - minimum) + m

    result = copy.deepcopy(image)
    result[:, :] = mapping_vector[image[:, :]]
    return result
