import cv2
import numpy as np


def image_rgb_to_grayscale(image):
    tmp = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)


def image_rgb_to_ycbcr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)


def equalize_histogram(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def blur_image(image):
    return cv2.GaussianBlur(image, (3, 3), 0)


def adjust_gamma(image, gamma=1.5):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
