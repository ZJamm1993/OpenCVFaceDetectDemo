# coding:utf-8
import numpy as np
import cv2
import zzimgtool

def swapfaceimg(container, face):
    sizeWidth = 600
    sizeHeight = 600
    container = cv2.resize(container, (sizeWidth, sizeHeight))
    faceimg = cv2.resize(face, (sizeWidth, sizeHeight))

    points1 = zzimgtool.facesLandmarks(cv2.cvtColor(container, cv2.COLOR_BGR2GRAY))[0]
    points2 = zzimgtool.facesLandmarks(faceimg)[0]

    hullIndexes = cv2.convexHull(np.array(points1), returnPoints=False)
    hullpoints1 = []
    hullpoints2 = []
    for index in hullIndexes.flatten():
        hullpoints1.append(points1[index])
        hullpoints2.append(points2[index])
    
    imgtargetmask = np.zeros_like(container)
    cv2.fillConvexPoly(imgtargetmask, np.array(hullpoints1), (255, 255, 255))

    imgWarp = zzimgtool.warpImage(faceimg, hullpoints2, hullpoints1)

    rx, ry, rw, rh = cv2.boundingRect(np.array(hullpoints1))
    center = (rx + rw / 2, ry + rh / 2)
    # BLUR mask
    imgtargetmask = cv2.GaussianBlur(imgtargetmask, (3, 3), 0)
    output = cv2.seamlessClone(imgWarp, container, imgtargetmask, center, cv2.NORMAL_CLONE)
    return output

def testwarp():
    faceimg = cv2.imread('face/testface.jpg')
    # faceimg = cv2.flip(faceimg, 1)
    container = cv2.imread('face/baby/baby6.png')

    output = swapfaceimg(container, faceimg)

    # cv2.imshow("src1", container)
    # cv2.imshow("src2", faceimg)
    cv2.imshow("warp", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__' :
    testwarp()