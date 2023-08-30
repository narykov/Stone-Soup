import cv2
import numpy as np


def read_raw(path, h=4096, w=4096):
    imgs = []

    with open(path, 'rb') as fid:
        while 1:
            vector = np.fromfile(fid, count=h*w, dtype=np.uint8)
            if vector.size == h*w:
                img = vector.reshape(h, w)
                img_interest = img[0:2048, 0:1000]
                img_interest = cv2.rotate(img_interest, cv2.ROTATE_90_COUNTERCLOCKWISE)
                imgs.append(img_interest[np.newaxis, :, :])
            else:
                break
    imgs = np.concatenate(imgs, axis=0)
    return imgs
