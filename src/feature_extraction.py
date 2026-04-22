import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(img):

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    return features


def extract_color_histogram(img):

    hist_features = []

    for i in range(3):  # BGR
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)

    return np.array(hist_features)


def extract_features(gray_img, color_img):

    hog_feat = extract_hog_features(gray_img)
    color_feat = extract_color_histogram(color_img)

    # 🔥 Balance أفضل (تجريبياً)
    hog_feat *= 0.8
    color_feat *= 0.2

    return np.hstack((hog_feat, color_feat))