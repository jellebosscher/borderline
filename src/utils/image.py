import cv2
import imutils
import numpy as np


def augment_image(image):
    # apply different augmentations to the image: rotate, resize, skew
    width = image.shape[1]
    height = image.shape[0]

    images = []
    images.append(image)

    # rotate
    images.append(imutils.rotate(image, 180))

    # resize
    images.append(imutils.resize(image, width=600))

    # change perspective
    pts1 = np.float32([[0, 0], [width, 0], [width, 387], [0, height]])
    pts2 = np.float32([[0, 10], [width, 10], [width, 300], [50, height - 10]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    images.append(cv2.warpPerspective(image, M, image.shape[:2][::-1]))

    return images
