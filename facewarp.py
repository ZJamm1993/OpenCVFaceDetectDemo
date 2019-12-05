# coding:utf-8
import numpy as np
import cv2
import zzimgtool

def testwarp():
    img1 = cv2.imread('/Users/zjj/Desktop/people1.png')
    img2 = cv2.imread('/Users/zjj/Desktop/people2.png')

    sizeWidth = 600
    sizeHeight = 800
    img1 = cv2.resize(img1, (sizeWidth, sizeHeight))
    img2 = cv2.resize(img2, (sizeWidth, sizeHeight))

    points1 = zzimgtool.faceLandmarks(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    points2 = zzimgtool.faceLandmarks(img2)


    hullIndexes = cv2.convexHull(np.array(points1), returnPoints=False)
    hullpoints1 = []
    hullpoints2 = []
    for index in hullIndexes.flatten():
        hullpoints1.append(points1[index])
        hullpoints2.append(points2[index])
    
    imgtargetmask = np.zeros_like(img1)
    cv2.fillConvexPoly(imgtargetmask, np.array(hullpoints1), (255, 255, 255))

    imgWarp = zzimgtool.warpImage(img2, hullpoints2, hullpoints1)

    rx, ry, rw, rh = cv2.boundingRect(np.array(hullpoints1))
    center = (rx + rw / 2, ry + rh / 2)
    output = cv2.seamlessClone(imgWarp, img1, imgtargetmask, center, cv2.NORMAL_CLONE)

    cv2.imshow("src1", img1)
    cv2.imshow("src2", img2)
    cv2.imshow("warp", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__' :
    testwarp()