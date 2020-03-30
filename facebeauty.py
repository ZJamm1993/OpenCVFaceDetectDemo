from cv2 import cv2
import numpy as np 

def beautyface(img):
    value1 = 5
    value2 = 3
    dx = value1 * 5
    fc = value1 * 12.5
    p = 0.5

    temp1 = cv2.bilateralFilter(img, dx, fc, fc)
    # cv2.imshow('temp1', temp1)

    temp2 = temp1 - img + 128
    # cv2.imshow('temp2', temp2)

    temp3 = cv2.GaussianBlur(temp2, (2 * value2 - 1, 2 * value2 - 1), 0)
    # cv2.imshow('temp3', temp3)

    floatimg = img.astype(np.float)
    floattemp3 = temp3.astype(np.float)
    temp4 = floatimg + 2.0 * floattemp3 - 255.0
    temp4[temp4 > 255] = 255
    temp4 = temp4.astype(np.uint8)
    # cv2.imshow('temp4', temp4)

    res = cv2.addWeighted(img, p, temp4, 1.0 - p, 1)
    return res

if __name__ == "__main__":
    img = cv2.imread('face/testface7.jpg')

    res = beautyface(img)

    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()