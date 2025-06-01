# ---------------------------------------------------------
# Portions of this file are adapted from Microsoft Genalog
# (https://github.com/microsoft/genalog) © Microsoft Corp.
# MIT License – see original repository for details.
# ---------------------------------------------------------
from math import floor
import cv2
import numpy as np

def blur(src, radius=5):
    return cv2.GaussianBlur(src, (radius | 1, radius | 1), cv2.BORDER_DEFAULT)

def overlay_weighted(src, background, alpha, beta, gamma=0):
    return cv2.addWeighted(src, alpha, background, beta, gamma).astype(np.uint8)

def overlay(src, background):
    return cv2.bitwise_and(src, background).astype(np.uint8)

def translation(src, offset_x, offset_y):
    """
    Shift the image in x, y direction, preserving multi-channel images.
    """
    rows, cols = src.shape[:2]
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    dst = cv2.warpAffine(src, M, (cols, rows), borderValue=255)
    return dst.astype(np.uint8)

def bleed_through(src, background=None, alpha=0.8, gamma=0, offset_x=0, offset_y=5):
    if background is None:
        background = src.copy()
    background = cv2.flip(background, 1)
    background = translation(background, offset_x, offset_y)
    return overlay_weighted(src, background, alpha, 1 - alpha, gamma)

def pepper(src, amount=0.05):
    dst = src.copy()
    noise = np.random.random(src.shape[:2])
    # broadcast noise mask across channels if needed
    mask = noise < amount
    if dst.ndim == 3:
        dst[mask] = 0
    else:
        dst[mask] = 0
    return dst.astype(np.uint8)

def salt(src, amount=0.05):
    dst = src.copy()
    noise = np.random.random(src.shape[:2])
    mask = noise < amount
    if dst.ndim == 3:
        dst[mask] = 255
    else:
        dst[mask] = 255
    return dst.astype(np.uint8)

def salt_then_pepper(src, salt_amount=0.1, pepper_amount=0.05):
    return pepper(salt(src, salt_amount), pepper_amount)

def pepper_then_salt(src, pepper_amount=0.05, salt_amount=0.1):
    return salt(pepper(src, pepper_amount), salt_amount)

# morphology helpers

def create_2D_kernel(shape, ktype="ones"):
    r, c = shape
    if ktype == "ones":
        k = np.ones(shape)
    elif ktype == "upper_triangle":
        k = np.triu(np.ones(shape))
    elif ktype == "lower_triangle":
        k = np.tril(np.ones(shape))
    elif ktype == "x":
        k = np.eye(r, c) + np.fliplr(np.eye(r, c))
        k[k > 1] = 1
    elif ktype == "plus":
        k = np.zeros(shape)
        k[:, c // 2] = 1
        k[r // 2, :] = 1
    elif ktype == "ellipse":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, shape)
    else:
        raise ValueError("Bad kernel_type")
    return k.astype(np.uint8)


def open_morph(src, kernel):   return cv2.morphologyEx(src, cv2.MORPH_OPEN,  kernel)
def close_morph(src, kernel):  return cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
def dilate(src, kernel):       return cv2.dilate(src, kernel)
def erode(src, kernel):        return cv2.erode(src, kernel)

def morphology(src, operation="open", kernel_shape=(3,3), kernel_type="ones"):
    k = create_2D_kernel(kernel_shape, kernel_type)
    if operation == "open":   return open_morph(src, k)
    if operation == "close":  return close_morph(src, k)
    if operation == "dilate": return dilate(src, k)
    if operation == "erode":  return erode(src, k)
    raise ValueError("Unknown morphology op")
